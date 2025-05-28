import numpy as np
import torch
from torch_geometric.data import Dataset, Data, Batch

def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data

def lanegcn_collate_fn(batch):
    # batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch

def traj_collate(batch):
    batch_size = len(batch)
    graphs = []

    ctrs = []
    actor_idcs = []
    count = 0

    actor_paths = []
    for i in range(batch_size):
        idcs = torch.arange(count, count + len(batch[i].veh_xseq))
        actor_idcs.append(idcs.cuda())
        count += len(batch[i].veh_xseq)
        ctrs.append(batch[i].veh_xseq[:,-1,:2].cuda())

        # add the path edge_indexs
        graph = {"ctrs":batch[i].lane_ctrs.cuda(),
                 "feats":torch.cat([batch[i].lane_vecs, batch[i].lane_widths],-1).cuda(),
                 "pris":batch[i].lane_pris.cuda(),
                 "ids":batch[i].lane_id.cuda(),
                 "num_nodes":batch[i].lane_ctrs.size(0),
                 "num_paths":batch[i].path_num_nodes.size(0),
                 "pre":{k:v.cuda() for k, v in batch[i].lane_pre_edge_index.items()},
                 "suc":{k:v.cuda() for k, v in batch[i].lane_suc_edge_index.items()},
                 "path":{k:v.cuda() for k, v in batch[i].lane_path.items()},
                 #"path_weight":{k:v.cuda() for k, v in batch[i].lane_path_weight.items()},
                 "left":batch[i].lane_left_edge_index.cuda(),
                 "right":batch[i].lane_right_edge_index.cuda(),
                 "start":batch[i].lane_start.cuda(),

                 "path->node":batch[i].path_lane_edge_index.cuda(),
                 "path-node->path-node":[e.cuda() for e in batch[i].path_node_node_edge_index],
                 "veh->path":batch[i].veh_path_edge_index.cuda(),
                 #"target":batch[i].lane_target.cuda(),
                 }
        graphs.append(graph)
        batch[i].lane_suc_edge_index = None
        batch[i].lane_pre_edge_index = None
        batch[i].lane_left_edge_index = None
        batch[i].lane_left_edge_index = None
        batch[i].lane_right_edge_index = None

        #batch[i].path_lane_edge_index = None
        batch[i].path_node_node_edge_index = None
        #batch[i].veh_path_edge_index = None

        batch[i].lane_path = None
        #batch[i].lane_path_weight = None
        batch[i].lane_start = None
        #batch[i].lane_target = None
        #batch[i].lane_ctrs = None
        batch[i].lane_vecs = None
        batch[i].lane_pris = None
        batch[i].lane_id = None
    batch = Batch.from_data_list(batch)

    dpos_xy1 = batch.veh_xseq[:, -2, 2:4]
    dpos_xy0 = batch.veh_xseq[:, -1, 2:4]

    d1 = torch.sqrt(torch.sum(dpos_xy1**2, dim=-1))
    d0 = torch.sqrt(torch.sum(dpos_xy0**2, dim=-1))
    info1 = batch.veh_xseq[:,-2,-1]

    batch['obs_head'] = torch.atan2(dpos_xy0[:,1], dpos_xy0[:,0])
    batch['obs_acc'] = (d0-d1)/0.04
    batch['obs_acc'][info1==0] = 0

    batch['obs_traj'] = batch.veh_xseq[:,:,:2].permute(1, 0, 2).cuda()
    batch['obs_traj_rel'] = batch.veh_xseq[:,:,2:4].permute(1, 0, 2).cuda()
    batch['obs_info'] = batch.veh_xseq[:,:,4:].permute(1, 0, 2).cuda()
    batch['obs_shape'] = batch.veh_shape.cuda()
    batch['fut_traj'] = batch.veh_yseq[:,:,:2].permute(1, 0, 2).cuda()
    batch['fut_traj_rel'] = batch.veh_yseq[:,:,2:].permute(1, 0, 2).cuda()
    batch['fut_traj_fre'] = batch.veh_yfre.permute(1, 0, 2).type('torch.cuda.FloatTensor')
    batch['fut_path'] = batch.veh_path.squeeze().cuda()
    batch.veh_full_path = batch.veh_full_path.cuda()
    batch['fut_target'] = batch.veh_target.squeeze().cuda()
    batch['has_preds'] = batch.veh_has_preds.permute(1, 0).cuda()
    #batch['obs_linear'] = batch.veh_x.cuda()
    batch['edge_index'] = batch.veh_edge_index.cuda()
    batch['veh_batch'] = actor_idcs
    batch['graphs'] = graphs
    batch['veh_ctrs'] = ctrs
    return batch

def vector_collate(batch):
    batch_size = len(batch)
    seq_len = batch[0].veh_xseq.size(1)
    num_preds = batch[0].veh_yseq.size(1)

    lane_feats = []
    lane_clusters = []
    batch_lane = []
    veh_feats = []
    veh_clusters = []
    batch_veh = []

    candidate, offset_gt, target_gt = [], [], []
    candidate_mask = []
    candidate_gt = []

    lane_counts = 0
    veh_counts = 0
    valid_lens = []
    veh_lens = []

    actor_ctrs = []
    actor_idcs = []
    count = 0
    graphs = []
    for i in range(batch_size):
        idcs = torch.arange(count, count + len(batch[i].veh_xseq))
        actor_idcs.append(idcs.cuda())
        count += len(batch[i].veh_xseq)
        actor_ctrs.append(batch[i].veh_xseq[:,-1,:2].cuda())
        graph = {"ctrs":batch[i].lane_ctrs.cuda(),
                 "feats":torch.cat([batch[i].lane_vecs, batch[i].lane_widths],-1).cuda(),
                 "pris":batch[i].lane_pris.cuda(),
                 "ids":batch[i].lane_id.cuda(),
                 "num_nodes":batch[i].lane_ctrs.size(0),
                 "num_paths":batch[i].path_num_nodes.size(0),
                 "pre":{k:v.cuda() for k, v in batch[i].lane_pre_edge_index.items()},
                 "suc":{k:v.cuda() for k, v in batch[i].lane_suc_edge_index.items()},
                 "path":{k:v.cuda() for k, v in batch[i].lane_path.items()},
                 #"path_weight":{k:v.cuda() for k, v in batch[i].lane_path_weight.items()},
                 "left":batch[i].lane_left_edge_index.cuda(),
                 "right":batch[i].lane_right_edge_index.cuda(),
                 "start":batch[i].lane_start.cuda(),

                 "path->node":batch[i].path_lane_edge_index.cuda(),
                 "path-node->path-node":[e.cuda() for e in batch[i].path_node_node_edge_index],
                 "veh->path":batch[i].veh_path_edge_index.cuda(),
                 #"target":batch[i].lane_target.cuda(),
                 }
        graphs.append(graph)
        lane_feat = torch.cat([batch[i].lane_ctrs,
                          batch[i].lane_vecs,
                          batch[i].lane_widths,
                          batch[i].lane_pris], -1)
        lane_feats.append(lane_feat)
        lane_cluster = torch.zeros_like(batch[i].lane_id, dtype=torch.long)
        unique_lane_ids = batch[i].lane_id.unique()
        for j, id in enumerate(unique_lane_ids):
            lane_cluster[batch[i].lane_id==id] = j+lane_counts
        lane_counts += len(unique_lane_ids)
        lane_clusters.append(lane_cluster)
        batch_lane.append(torch.ones_like(unique_lane_ids, dtype=torch.long)*i)


        veh_info = batch[i].veh_xseq[:,:,4]
        num_vehs = batch[i].veh_xseq.size(0)
        timestamps = torch.arange(seq_len).float().repeat(num_vehs).reshape(num_vehs, seq_len, 1)
        veh_cluster = torch.arange(num_vehs).repeat(seq_len).reshape(seq_len, num_vehs).permute(1,0) + veh_counts
        veh_feat = torch.cat([batch[i].veh_xseq[:, :, :2],
                              timestamps], -1)
        veh_feats.append(veh_feat[veh_info==1])
        veh_counts += num_vehs
        veh_clusters.append(veh_cluster[veh_info==1])
        batch_veh.append(torch.ones(num_vehs, dtype=torch.long)*i)

        valid_lens.append(num_vehs+len(unique_lane_ids))
        veh_lens.append(num_vehs)

        last = batch[i].veh_has_preds.float() + 0.1 * torch.arange(num_preds).float() / float(num_preds)
        max_last, last_idcs = last.max(1)
        row_idcs = torch.arange(len(last_idcs)).long()

        veh_target_ctrs = batch[i].veh_yseq[row_idcs,last_idcs]
        veh_target_ctrs = veh_target_ctrs[:,:2]
        lane_ctrs = batch[i].lane_ctrs

        for v in range(num_vehs):
            veh_path_idcs = batch[i].veh_path_edge_index[1][batch[i].veh_path_edge_index[0] == v]
            lctrs = []
            unique_lane_idcs = []
            for p in veh_path_idcs:
                path = lane_ctrs[batch[i].path_lane_edge_index[1][batch[i].path_lane_edge_index[0]==p]]
                lane_idcs = batch[i].path_lane_edge_index[1][batch[i].path_lane_edge_index[0]==p].numpy()
                if path.size(0) < 15:
                    delta_xy = path[-1] - path[-2]
                    add_points = torch.cumsum(delta_xy.repeat(15-path.size(0),1), dim=0) + path[-1]
                    lane_idcs = np.concatenate([lane_idcs, np.ones(15-path.size(0))*lane_idcs[-1]], axis=0)
                    path = torch.cat([path, add_points], dim=0)
                else:
                    path = path[:15]
                    lane_idcs = lane_idcs[:15]

                if len(unique_lane_idcs) == 0:
                    lctrs.append(path)
                    lctrs.append((path[1:] - path[:-1] )*1/5 +path[:-1])
                    lctrs.append((path[1:] - path[:-1] )*2/5 +path[:-1])
                    lctrs.append((path[1:] - path[:-1] )*3/5 +path[:-1])
                    lctrs.append((path[1:] - path[:-1] )*4/5 +path[:-1])
                else:
                    mask = torch.ones(15, dtype=torch.bool)
                    for lane_idc in unique_lane_idcs:
                        mask = mask & (lane_idcs != lane_idc)
                    mask = mask.bool()
                    lctrs.append(path[mask])
                    lctrs.append(((path[1:] - path[:-1] )*1/5 +path[:-1])[mask[:14]])
                    lctrs.append(((path[1:] - path[:-1] )*2/5 +path[:-1])[mask[:14]])
                    lctrs.append(((path[1:] - path[:-1] )*3/5 +path[:-1])[mask[:14]])
                    lctrs.append(((path[1:] - path[:-1] )*4/5 +path[:-1])[mask[:14]])
                for lane_idc in lane_idcs:
                    if lane_idc not in unique_lane_idcs:
                        unique_lane_idcs.append(lane_idc)

            lctrs = torch.cat(lctrs, dim=0)
            #lctrs = lane_ctrs[des[src==v]]
            candidate.append(lctrs)
            dist = (lctrs - veh_target_ctrs[v])**2
            dist = dist.sum(-1)
            candidate_gt.append(dist.argmin())
            offset_gt.append(veh_target_ctrs[v] - lctrs[dist.argmin()])

            candidate_mask.append(len(candidate[-1]))

        target_gt.append(veh_target_ctrs)

        batch[i].lane_suc_edge_index = None
        batch[i].lane_pre_edge_index = None
        batch[i].lane_left_edge_index = None
        batch[i].lane_left_edge_index = None
        batch[i].lane_right_edge_index = None

        batch[i].path_lane_edge_index = None
        batch[i].path_node_node_edge_index = None
        batch[i].veh_path_edge_index = None

        batch[i].lane_path = None
        batch[i].lane_start = None
        batch[i].lane_ctrs = None
        batch[i].lane_vecs = None
        batch[i].lane_pris = None
        batch[i].lane_id = None
        #batch[i].veh_xseq = None

        batch[i].veh_yfre = None
        batch[i].veh_path = None
        batch[i].veh_full_path = None
        batch[i].veh_target = None
        batch[i].veh_x = None
        batch[i].veh_edge_index = None


    batch = Batch.from_data_list(batch)

    max_candidate_len = max(candidate_mask)
    for i in range(len(candidate)):
        zeros = torch.zeros(max_candidate_len-len(candidate[i]),2, dtype=torch.float)
        candidate[i] = torch.cat([candidate[i], zeros], 0)

    batch['candidate'] = torch.stack(candidate, 0).cuda()
    batch['offset_gt'] = torch.stack(offset_gt, 0).cuda()
    batch['target_gt'] = torch.cat(target_gt, 0).cuda()
    batch['candidate_gt'] = torch.stack(candidate_gt, 0).cuda()
    batch['candidate_mask'] = candidate_mask

    batch['actor_ctrs'] = torch.cat(actor_ctrs, dim=0).cuda()
    batch['obs_traj'] = batch.veh_xseq[:,:,:2].permute(1, 0, 2).cuda()
    batch['veh_feat'] = torch.cat(veh_feats, 0).cuda()
    batch['veh_cluster'] = torch.cat(veh_clusters, 0).cuda()
    batch['batch_veh'] = torch.cat(batch_veh, 0).cuda()
    batch['lane_feat'] = torch.cat(lane_feats, 0).cuda()
    batch['lane_cluster'] = torch.cat(lane_clusters, 0).cuda()
    batch['batch_lane'] = torch.cat(batch_lane, 0).cuda()
    batch['fut_traj'] = batch.veh_yseq[:,:,:2].permute(1, 0, 2).cuda()
    batch['fut_traj_rel'] = batch.veh_yseq[:,:,2:].permute(1, 0, 2).cuda()
    batch['has_preds'] = batch.veh_has_preds.permute(1, 0).cuda()
    batch['valid_lens'] = valid_lens
    batch['veh_lens'] = veh_lens
    batch['obs_traj_rel'] = batch.veh_xseq[:,:,2:4].permute(1, 0, 2).cuda()
    batch['obs_info'] = batch.veh_xseq[:,:,4:].permute(1, 0, 2).cuda()
    batch['veh_batch'] = actor_idcs
    batch['graphs'] = graphs

    return batch
