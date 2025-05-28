import torch
import torch.nn as nn
import math
from torch_scatter import scatter_max, scatter_add, scatter_softmax
from models.ftt.utils.losses import GraphCrossEntropyLoss
from models.ftt.utils.util import relative_to_curve_with_curvature 
from models.laneGCN.layers import ActorNet, MapNet, A2M, M2M, M2A, A2A
from models.ftt.modules import PathPredNet, TargetsPredAttNet


def graph_gather(graphs, veh_batch):
    batch_size = len(graphs)
    graph = dict()
    veh_count = 0
    veh_counts = []
    for i in range(batch_size):
        veh_counts.append(veh_count)
        veh_count = veh_count + veh_batch[i].size(0)

    graph["num_vehs"] = veh_count
    node_idcs = []
    node_count = 0
    node_counts = []
    for i in range(batch_size):
        node_counts.append(node_count)
        idcs = torch.arange(node_count, node_count + graphs[i]["num_nodes"]).to(
            graphs[i]["feats"].device
        )
        node_idcs.append(idcs)
        node_count = node_count + graphs[i]["num_nodes"]

    graph["num_nodes"] = node_count
    graph["idcs"] = node_idcs
    graph["ctrs"] = [x["ctrs"] for x in graphs]
    for key in ["feats", "pris", "start"]:
        graph[key] = torch.cat([x[key] for x in graphs], 0)
    # graph["obs_idcs"] = [x["start"] for x in graphs]

    path_idcs = []
    path_count = 0
    path_counts = []
    for i in range(batch_size):
        path_counts.append(path_count)
        idcs = torch.arange(path_count, path_count + graphs[i]["num_paths"]).to(
            graphs[i]["feats"].device
        )
        path_idcs.append(idcs)
        path_count = path_count + graphs[i]["num_paths"]

    graph["num_paths"] = path_count
    graph["path_idcs"] = path_idcs # number of batch size with path indices
    # edges between path and nodes
    graph["path->node"] = dict()
    graph["path->node"]["u"] = torch.cat(
            [graphs[j]["path->node"][0] + path_counts[j] for j in range(batch_size)], 0
        )
    graph["path->node"]["v"] = torch.cat(
        [graphs[j]["path->node"][1] + node_counts[j] for j in range(batch_size)], 0
    )

    graph["path-node->path-node"] = []
    for g in graphs:
        graph["path-node->path-node"].extend(g["path-node->path-node"])

    graph["veh->path"] = dict()
    graph["veh->path"]["u"] = torch.cat(
        [graphs[j]["veh->path"][0] + veh_counts[j] for j in range(batch_size)], 0
    )
    graph["veh->path"]["v"] = torch.cat(
        [graphs[j]["veh->path"][1] + path_counts[j] for j in range(batch_size)], 0
    )
    graph["veh_path_idcs"] = []
    for i in range(graph["num_vehs"]):
        graph["veh_path_idcs"].append(graph["veh->path"]["v"][graph["veh->path"]["u"]==i])

    for k1 in ["pre", "suc"]:
        graph[k1] = []
        for i in range(len(graphs[0]["pre"])):
            graph[k1].append(dict())
            for ki, k2 in enumerate(["u", "v"]):
                graph[k1][i][k2] = torch.cat(
                    [graphs[j][k1][i][ki] + node_counts[j] for j in range(batch_size)], 0
                )

    for k1 in ["left", "right"]:
        graph[k1] = dict()
        for ki, k2 in enumerate(["u", "v"]):
            temp = [graphs[i][k1][ki] + node_counts[i] for i in range(batch_size)]
            temp = [
                x if x.dim() > 0 else graph["pre"][0]["u"].new().resize_(0)
                for x in temp
            ]
            graph[k1][k2] = torch.cat(temp)
    return graph

class FrenetPathMultiTargetGCN(nn.Module):
    def __init__(self, config):
        super(FrenetPathMultiTargetGCN, self).__init__()
        self.config = config
        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)

        self.a2m = A2M(config)
        self.m2m = M2M(config)
        self.m2a = M2A(config)
        self.a2a = A2A(config)
        self.path_net = PathPredNet(config, latent=False)
        self.tar_traj_net = TargetsPredAttNet(config)

        self.path_loss = GraphCrossEntropyLoss()
        self.target_cls_loss = nn.CrossEntropyLoss(reduction="mean")
        self.target_offset_loss = nn.SmoothL1Loss(reduction="mean")
        self.traj_loss = nn.SmoothL1Loss(reduction="mean")

        # self.device = torch.device('cuda')
        # if "save_path" in config:
        #     self.writer = SummaryWriter(config["save_path"])
        # else:
        #     self.writer = None

    def select_paths(self, veh_path_idcs, path_num_nodes, nodes, graph):
        hi = []
        wi = []
        p_hi = []
        p_wi = []
        count = 0

        div_term = torch.exp(torch.arange(0, self.config["n_map"], 2) * (-math.log(10000.0) / self.config["n_map"]))
        pe = torch.zeros(40, self.config["n_map"]).cuda()
        position = torch.arange(40).unsqueeze(1)
        pe[:,  0::2] = torch.sin(position * div_term)
        pe[:,  1::2] = torch.cos(position * div_term)

        PE = []

        traj_edge_index = []
        node_traj_edge_index = []
        actor_traj_index = []

        for i in range(veh_path_idcs.size(0)):
            w = graph["path->node"]["v"][graph["path->node"]["u"]==veh_path_idcs[i,0]]
            PE.append(pe[:w.size(0)])

            wi.append(w)
            hi.append(torch.ones_like(w)*i)
            try:
                node_node_edge_index = graph["path-node->path-node"][veh_path_idcs[i,0]]
            except:
                print(len(graph["path-node->path-node"]), i, veh_path_idcs[i,0], veh_path_idcs)
                raise
            p_hi.append(node_node_edge_index[0]+count)
            p_wi.append(node_node_edge_index[1]+count)


            traj_sender, traj_receiver = [], []
            for pn in range(self.config["num_preds"]):
                for ppn in range(pn, self.config["num_preds"]):
                    traj_sender.append(pn + i*self.config["num_preds"])
                    traj_receiver.append(ppn + i*self.config["num_preds"])
            traj_edge_index.append(torch.tensor([traj_sender,
                                                 traj_receiver],
                                                 dtype=torch.long).cuda()
                                    )
            node_traj_sender, node_traj_receiver = [], []
            for pn in range(w.size(0)):
                for ppn in range(self.config["num_preds"]):
                    node_traj_sender.append(pn + count)
                    node_traj_receiver.append(ppn + i*self.config["num_preds"])
            node_traj_edge_index.append(torch.tensor([node_traj_sender,
                                                 node_traj_receiver],
                                                 dtype=torch.long).cuda()
                                    )
            actor_traj_index.append(i*torch.ones(self.config["num_preds"],dtype=torch.long).cuda())

            count += path_num_nodes[veh_path_idcs[i,0]]

        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)
        p_hi = torch.cat(p_hi, 0)
        p_wi = torch.cat(p_wi, 0)

        PE = torch.cat(PE, dim=0)
        target_paths = nodes[wi] + PE

        traj_edge_index = torch.cat(traj_edge_index, dim=1)
        node_traj_edge_index = torch.cat(node_traj_edge_index, dim=1)
        actor_traj_index = torch.cat(actor_traj_index)

        initial_poses = graph["start"][veh_path_idcs].squeeze(1)
        return target_paths, \
               initial_poses, \
               torch.stack([p_hi, p_wi]), \
               torch.stack([torch.arange(0, len(wi)).to(nodes.device), hi]), \
               traj_edge_index,\
               node_traj_edge_index,\
               actor_traj_index

    def update_summary(self, name, value, step):
        if isinstance(value, int) or isinstance(value, float):
            self.writer.add_scalar(name, value, step)
        else:
            self.writer.add_scalars(name, value, step)

    def forward(self, batch):
        # obs_traj_rel:[S, N8, 2] S:10
        # obs_info: [S, NB, 1] S:10
        actor_idcs = batch.veh_batch
        actor_ctrs = batch.veh_ctrs
        batch_size = len(actor_idcs)

        actors_x =  torch.cat([batch.obs_traj_rel, batch.obs_info], -1).permute(1, 2, 0)
        actors_x = self.actor_net(actors_x)

        # contruct traffic features
        graph = graph_gather(batch.graphs, batch.veh_batch)
        nodes, node_idcs, node_ctrs = self.map_net(graph)
        lane_ctrs = torch.cat(node_ctrs, 0)

        nodes = self.a2m(nodes, graph, actors_x, actor_idcs, actor_ctrs)
        nodes = self.m2m(nodes, graph)


        actors_x = self.m2a(actors_x, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)
        actors_x = self.a2a(actors_x, actor_idcs, actor_ctrs)
        #actor_edge_index = self.actor_edges(actor_idcs, actor_ctrs)

        gt_preds = batch.fut_traj_fre[:,:,:2].permute(1, 0, 2)
        gt_targets = torch.stack([batch.fut_target[:,0],
                                  batch.fut_target[:,2]], dim=-1)

        S = torch.arange(0,60,3.)
        D = torch.arange(-3,4,3.)
        tar_candidate = torch.stack([S.unsqueeze(0).repeat(3,1),
                                     D.unsqueeze(1).repeat(1,20)], dim=-1).to(gt_targets.device)
        tar_candidate = tar_candidate.reshape(1, -1, 2).repeat(gt_targets.size(0), 1, 1)
        offsets = gt_targets.unsqueeze(1) - tar_candidate
        _, gt_target_idcs = (offsets**2).sum(-1).min(-1)
        row_idcs = torch.arange(gt_target_idcs.size(0)).to(gt_targets.device)
        gt_target_offsets = offsets[row_idcs, gt_target_idcs]

        has_preds = batch.has_preds.permute(1, 0)
        last = has_preds.float() + 0.1 * torch.arange(self.config['num_preds']).float().to(
            has_preds.device
        ) / float(self.config['num_preds'])
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        if self.config["num_mods"] == 1:
            if self.config["sample_mode"] == "ground_truth":
                veh_path_idcs = veh_path_idcs_gt = torch.arange(0, graph["num_paths"]).unsqueeze(1)[batch.veh_full_path == 1].unsqueeze(1).cuda()

            else:
                paths, _ = scatter_max(nodes[graph["path->node"]["v"]], graph["path->node"]["u"], dim=0, dim_size=graph["num_paths"])
                path_logits = self.path_net(actors_x, paths, None, None, graph)
                path_probs = scatter_softmax(path_logits, graph['veh->path']['u'], dim=0).squeeze()
                path_loss = self.path_loss(path_logits, batch.veh_full_path, graph["veh->path"]["u"], has_preds.size(0), mask)
                _, veh_path_idcs_pred = scatter_max(path_logits, graph["veh->path"]["u"], dim=0, dim_size=graph["num_vehs"])
                veh_path_idcs_gt = torch.arange(0, graph["num_paths"]).unsqueeze(1)[batch.veh_full_path == 1].unsqueeze(1).cuda()

            #if self.training:
            veh_path_idcs = veh_path_idcs_gt

            curves = [lane_ctrs[graph['path->node']["v"][graph['path->node']["u"]==pidx]] for pidx in veh_path_idcs]
            path_nodes, initial_poses,\
            node_edge_index, \
            node_target_edge_index, \
            traj_edge_index, \
            node_traj_edge_index,\
            actor_traj_index = self.select_paths(veh_path_idcs, batch.path_num_nodes, nodes, graph)
            traj_pred, target_pred, target_prob, target_logits, target_offset, _ = self.tar_traj_net(actors_x, path_nodes, initial_poses,
                                      gt_targets,
                                      tar_candidate,
                                      node_edge_index,
                                      node_target_edge_index,
                                      traj_edge_index,
                                      node_traj_edge_index,
                                      actor_traj_index )

        else:
            # Generation of K paths
            frenet_preds = []
            frenet_curves = []
            preds = []
            curvatures = []
            target_preds = []
            all_target_probs = []

            paths, _ = scatter_max(nodes[graph["path->node"]["v"]], graph["path->node"]["u"], dim=0, dim_size=graph["num_paths"])
            path_logits = self.path_net(actors_x, paths, None, None, graph)
            path_probs = scatter_softmax(path_logits, graph['veh->path']['u'], dim=0).squeeze()

            if self.config["sample_mode"] == "ucb" or self.config["sample_mode"] == "bias":
                if self.config["sample_mode"] == "ucb":
                    veh_path_idcs, ranks = ucb_sample_path(graph, path_logits, num_samples=self.config['num_mods'])
                    #print(veh_path_idcs.size(), ranks.size())
                else:
                    veh_path_idcs, ranks = bias_sample_path(graph, path_logits, num_samples=self.config['num_mods'])
                    #print(veh_path_idcs.size(), ranks.size())
                path_loss = self.path_loss(path_logits, batch.veh_full_path, graph["veh->path"]["u"], has_preds.size(0), mask)

            else:
                veh_path_idcs, ranks = uniform_sample_path(graph, num_samples=self.config['num_mods'])

            scores = []
            for m in range(self.config['num_mods']):
                # the first version doesn't consider the joint path distribution
                path_nodes, initial_poses,\
                node_edge_index, \
                node_target_edge_index, \
                traj_edge_index, \
                node_traj_edge_index,\
                actor_traj_index = self.select_paths(veh_path_idcs[m].unsqueeze(1), batch.path_num_nodes, nodes, graph)
                traj_pred, target_pred, target_prob, _, _, target_probs = self.tar_traj_net(actors_x, path_nodes, initial_poses,
                                          gt_targets,
                                          tar_candidate,
                                          node_edge_index,
                                          node_target_edge_index,
                                          traj_edge_index,
                                          node_traj_edge_index,
                                          actor_traj_index, ranks[m])
                scores.append(path_probs[veh_path_idcs[m]]*target_prob)
                all_target_probs.append(target_probs)
                frenet_preds.append(traj_pred)

                curves = [lane_ctrs[graph['path->node']["v"][graph['path->node']["u"]==pidx]] for pidx in veh_path_idcs[m]]
                frenet_curves.append(curves)
                converted_pred, curvature = relative_to_curve_with_curvature(frenet_preds[-1].permute(1, 0, 2), batch.obs_traj[-1], curves)
                preds.append(converted_pred.permute(1, 0, 2))
                curvatures.append(curvature)
                target_preds.append(target_pred)

            scores = torch.stack(scores, 1)
            frenet_preds = torch.stack(frenet_preds, 1)
            #score_out = self.score_net(actors_x, actor_idcs, actor_ctrs, preds)
            #score_loss = self.score_loss(score_out, batch.fut_traj, batch.has_preds)
            #score_loss = score_loss['cls_loss']/(score_loss['num_cls']+1e-10)
            #frenet_preds_sorted = frenet_preds[score_out["row_idcs"], score_out["sort_idcs"]].view(score_out["cls_sorted"].size(0),
            #                                                                                       score_out["cls_sorted"].size(1), -1, 2)

            #if self.training:
            veh_path_idcs_gt = torch.arange(0, graph["num_paths"]).unsqueeze(1)[batch.veh_full_path == 1].unsqueeze(1).cuda()
            path_nodes, initial_poses,\
            node_edge_index, \
            node_target_edge_index, \
            traj_edge_index, \
            node_traj_edge_index,\
            actor_traj_index = self.select_paths(veh_path_idcs_gt, batch.path_num_nodes, nodes, graph)
            traj_pred, target_pred, _, target_logits, target_offset, _= self.tar_traj_net(actors_x, path_nodes, initial_poses,
                                          gt_targets,
                                          tar_candidate,
                                          node_edge_index,
                                          node_target_edge_index,
                                          traj_edge_index,
                                          node_traj_edge_index,
                                          actor_traj_index )
            curves = [lane_ctrs[graph['path->node']["v"][graph['path->node']["u"]==pidx]] for pidx in veh_path_idcs_gt]
            converted_pred, converted_curvature = relative_to_curve_with_curvature(traj_pred.permute(1, 0, 2), batch.obs_traj[-1], curves)
            #else:
            #    traj_pred = frenet_preds_sorted[:, 0]
            #    converted_pred = score_out['pred'].permute(1, 0, 2)

        traj_loss = self.traj_loss(traj_pred[mask][has_preds[mask]], gt_preds[mask][has_preds[mask]])
        target_cls_loss = self.target_cls_loss(target_logits[mask], gt_target_idcs[mask])
        target_offset_loss = self.target_offset_loss(target_offset[row_idcs, gt_target_idcs][mask], gt_target_offsets[mask])

        tar = batch.veh_full_path==1
        path_idx = graph['veh->path']["v"][tar.squeeze()]
        curves_gt = []
        for idx in range(path_idx.size(0)):
            curves_gt.append(lane_ctrs[graph['path->node']["v"][graph['path->node']["u"]==path_idx[idx]]])

        if self.config["num_mods"] == 1:
            ret = {'frenet_pred': traj_pred.permute(1, 0, 2),
                   'curves':curves,
                   'curves_gt':curves_gt,
                   'converted_gt':batch.fut_traj_fre[:,:,2:],
                   'loss':traj_loss + 0.5*path_loss + 0.1*(target_cls_loss+target_offset_loss),
                   'tloss':traj_loss,
                   'ploss':path_loss,
                   'target':target_pred,
                   'tar_cls_loss':target_cls_loss,
                   'tar_offset_loss':target_offset_loss,
                   'path': veh_path_idcs_pred,
                   'gt_path': veh_path_idcs_gt}

        else:
            ret = {'frenet_pred': traj_pred.permute(1, 0, 2),
                   'converted_pred': converted_pred,
                   'converted_curvature': converted_curvature,
                   'curves':frenet_curves,
                   'curves_gt':curves_gt,
                   'converted_gt':batch.fut_traj_fre[:,:,2:],
                   'loss':traj_loss + 0.5*path_loss + 0.1*(target_cls_loss+target_offset_loss),
                   'tloss':traj_loss,
                   'ploss':path_loss,
                   'tar_cls_loss':target_cls_loss,
                   'tar_offset_loss':target_offset_loss,
                   'paths': veh_path_idcs,
                   'path_probs':path_probs,
                   'targets': target_preds,
                   'target_probs':all_target_probs,
                   'scores': scores,
                   'reg':preds,
                   'reg_frenets':frenet_preds,
                   'reg_curvatures':curvatures}

        return ret

    def predict(self, batch):
        # obs_traj_rel:[S, N8, 2] S:10
        # obs_info: [S, NB, 1] S:10
        actor_idcs = batch.veh_batch
        actor_ctrs = batch.veh_ctrs
        batch_size = len(actor_idcs)

        actors_x =  torch.cat([batch.obs_traj_rel, batch.obs_info], -1).permute(1, 2, 0)
        actors_x = self.actor_net(actors_x)

        # contruct traffic features
        graph = graph_gather(batch.graphs, batch.veh_batch)
        #lane_ids = graph["ids"]
        nodes, node_idcs, node_ctrs = self.map_net(graph)
        lane_ctrs = torch.cat(node_ctrs, 0)

        nodes = self.a2m(nodes, graph, actors_x, actor_idcs, actor_ctrs)
        nodes = self.m2m(nodes, graph)

        actors_x = self.m2a(actors_x, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)
        actors_x = self.a2a(actors_x, actor_idcs, actor_ctrs)
        #actor_edge_index = self.actor_edges(actor_idcs, actor_ctrs)

        #gt_preds = batch.fut_traj_fre[:,:,:2].permute(1, 0, 2)
        #gt_targets = torch.stack([batch.fut_target[:,0],
        #                          batch.fut_target[:,2]], dim=-1)

        S = torch.arange(0,60,3.)
        D = torch.arange(-3,4,3.)
        tar_candidate = torch.stack([S.unsqueeze(0).repeat(3,1),
                                     D.unsqueeze(1).repeat(1,20)], dim=-1).to(actors_x.device)
        tar_candidate = tar_candidate.reshape(1, -1, 2).repeat(actors_x.size(0), 1, 1)
        #offsets = gt_targets.unsqueeze(1) - tar_candidate
        #_, gt_target_idcs = (offsets**2).sum(-1).min(-1)
        #row_idcs = torch.arange(gt_target_idcs.size(0)).to(gt_targets.device)
        #gt_target_offsets = offsets[row_idcs, gt_target_idcs]

        #has_preds = batch.has_preds.permute(1, 0)
        #last = has_preds.float() + 0.1 * torch.arange(self.config['num_preds']).float().to(
        #    has_preds.device
        #) / float(self.config['num_preds'])
        #max_last, last_idcs = last.max(1)
        #mask = max_last > 1.0


        # Generation of K paths
        frenet_preds = []
        frenet_curves = []
        preds = []
        target_preds = []
        all_target_probs = []

        paths, _ = scatter_max(nodes[graph["path->node"]["v"]], graph["path->node"]["u"], dim=0, dim_size=graph["num_paths"])
        path_logits = self.path_net(actors_x, paths, None, None, graph)
        path_probs = scatter_softmax(path_logits, graph['veh->path']['u'], dim=0).squeeze()

        if self.config["sample_mode"] == "ucb" or self.config["sample_mode"] == "bias":
            if self.config["sample_mode"] == "ucb":
                veh_path_idcs, ranks = ucb_sample_path(graph, path_logits, num_samples=self.config['num_mods'])
            else:
                veh_path_idcs, ranks = bias_sample_path(graph, path_logits, num_samples=self.config['num_mods'])
        else:
            veh_path_idcs, ranks = uniform_sample_path(graph, path_logits, num_samples=self.config['num_mods'])

        scores = []
        #predictions = dict()
        #veh_ids = batch.veh_id.numpy().astype(int)
        for m in range(self.config['num_mods']):
            path_nodes, initial_poses,\
            node_edge_index, \
            node_target_edge_index, \
            traj_edge_index, \
            node_traj_edge_index,\
            actor_traj_index = self.select_paths(veh_path_idcs[m].unsqueeze(1), batch.path_num_nodes, nodes, graph)
            traj_pred, target_pred, target_prob, _, _, target_probs = self.tar_traj_net(actors_x, path_nodes, initial_poses,
                                      None,
                                      tar_candidate,
                                      node_edge_index,
                                      node_target_edge_index,
                                      traj_edge_index,
                                      node_traj_edge_index,
                                      actor_traj_index, ranks[m])
            scores.append(path_probs[veh_path_idcs[m]]*target_prob)
            all_target_probs.append(target_probs)
            frenet_preds.append(traj_pred)

            curves = [lane_ctrs[graph['path->node']["v"][graph['path->node']["u"]==pidx]] for pidx in veh_path_idcs[m]]
            #lane_ids = [lane_ids[graph['path->node']["v"][graph['path->node']["u"]==pidx]] for pidx in veh_path_idcs[m]]
            frenet_curves.append(curves)
            #pred_traj_rel = relative_to_curve(frenet_preds[-1].permute(1, 0, 2), batch.obs_traj[-1], curves)
            #pred_traj = relative_to_abs(pred_traj_rel, batch.obs_traj[-1])

            #target_preds.append(target_pred)
            # x, y, h, phi, s, d, sa, da

        scores = torch.stack(scores, 1)
        frenet_preds = torch.stack(frenet_preds, 1)

        ret = {'curves':frenet_curves,
               'paths': veh_path_idcs,
               'path_probs':path_probs,
               'targets': target_preds,
               'target_probs':all_target_probs,
               'scores': scores,
               'trajs':frenet_preds}

        return ret

def ucb_sample_path(graph, logits, num_samples=6, c=0.2):
    logits = logits.squeeze()
    idcs = []
    n = torch.zeros_like(logits)
    exp_logits = scatter_softmax(logits, graph["veh->path"]["u"], dim=0)
    _, idc = scatter_max(exp_logits, graph["veh->path"]["u"], dim=0)
    idcs.append(idc.view(1,-1))
    ranks = []
    ranks.append(n[idc])
    n[idc] += 1

    for s in range(num_samples-1):
        sn = scatter_add(n, graph["veh->path"]["u"], dim=0) + 0.01
        sn = sn[graph["veh->path"]["u"]]
        _, idc = scatter_max(exp_logits + c*torch.sqrt(torch.log(sn)/n), graph["veh->path"]["u"], dim=0)
        idcs.append(idc.view(1,-1))
        ranks.append(n[idc])
        n[idc] += 1
    return torch.cat(idcs, dim=0), torch.stack(ranks, dim=0).long()

def bias_sample_path(graph, logits, num_samples=6):
    idcs = []
    ranks = []
    n = torch.zeros_like(logits)
    for path_idcs in graph["veh_path_idcs"]:
        m = torch.distributions.categorical.Categorical(logits=logits.squeeze()[path_idcs])
        idcs.append(path_idcs[m.sample((num_samples,))])

    idcs = torch.stack(idcs, dim=1)
    for s in range(num_samples):
        idc = idcs[s]
        ranks.append(n[idc])
        n[idc] += 1
    return idcs.to(logits.device), torch.stack(ranks, dim=0).squeeze().long()

def uniform_sample_path(graph, logits, num_samples=6):
    idcs = []
    ranks = []
    n = torch.zeros_like(logits)
    for path_idcs in graph["veh_path_idcs"]:
        idcs.append(torch.randint(path_idcs[0], path_idcs[-1]+1, (num_samples,1)))
    idcs = torch.cat(idcs, dim=1)
    for s in range(num_samples):
        idc = idcs[s]
        ranks.append(n[idc])
        n[idc] += 1
    return idcs.to(logits.device), torch.stack(ranks, dim=0)