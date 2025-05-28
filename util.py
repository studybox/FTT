# import pickle
# import os
# import os.path as osp
# from collections import defaultdict
# import numpy as np
# import torch

# from scipy.sparse import csr_matrix
# import scipy
# from scipy import sparse
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib import collections  as mc
# from matplotlib.widgets import Button, Slider
# from matplotlib.patches import Circle, Arc, Polygon
# import networkx as nx

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from commonroad.scenario.lanelet import LineMarking, LaneletType
# from commonroad.scenario.trajectorycomplement import FrenetState, Frenet
from torch_geometric.data import Dataset, Data, Batch
# from commonroad.scenario.scenariocomplement import MultiFeatureExtractor, CoreFeatureExtractor1, NeighborFeatureExtractor, ScenarioWrapper, LaneletToLaneletFeatureExtractor1, LaneletFeatureExtractor1, VehicleToLaneletFeatureExtractor1
# from commonroad.scenario.obstacle import StaticObstacle, ObstacleType, DynamicObstacle
# from commonroad.common.file_reader_complement import LaneletCurveNetworkReader
# from commonroad.geometry.shape import Rectangle

# from commonroad.scenario.trajectorycomplement import FrenetState, Frenet, move_along_curve
# from commonroad.scenario.laneletcomplement import *

# from commonroad.visualization.draw_dispatch_cr import draw_object
# from commonroad.scenario.laneletcomplement import make_lanelet_curve_network
# from commonroad.common.file_reader import CommonRoadFileReader
# from torch_geometric.utils import degree, add_self_loops
# from torch_scatter import scatter_mean, scatter_max, scatter_add
# #from config import basic_shape_parameters_ego, basic_shape_parameters_nei, basic_shape_parameters_obs, draw_params_neighbor, draw_params_ego, draw_params_obstacle
# #from motion_prediction.losses import l2_loss, displacement_error, final_displacement_error

# from losses import l2_loss, displacement_error, final_displacement_error

# MAX_SPEED = 45  # maximum speed [m/s]
# MAX_ACCEL = 3.0  # maximum acceleration [m/ss]
# MAX_DECEL = 8.0 # maximum dec[m/ss]
# MAX_CURVATURE = 1.0  # maximum curvature [1/m]

class ScenarioData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
# class ScenarioData(Data):
#     def __init__(self, veh_id=None,
#                  veh_t=None, veh_x=None, veh_xseq=None, veh_shape=None, veh_has_preds=None,
#                  veh_yseq=None, veh_yfre= None, veh_edge_index=None, veh_edge_attr=None,
#                  lane_ctrs=None, lane_vecs=None, lane_pris=None, lane_suc_edge_index=None,
#                  lane_pre_edge_index=None, lane_left_edge_index=None, lane_right_edge_index=None,
#                  veh_path=None, veh_target=None, lane_start=None, lane_path=None, lane_id = None,
#                  veh_full_path=None, veh_path_edge_index=None, path_lane_edge_index=None,
#                  lane_widths=None, path_num_nodes=None, path_node_node_edge_index=None):
#         super(ScenarioData, self).__init__()
#         self.veh_id = veh_id
#         self.veh_t = veh_t
#         self.veh_x = veh_x
#         self.veh_xseq = veh_xseq
#         self.veh_shape = veh_shape
#         self.veh_yseq = veh_yseq
#         self.veh_yfre = veh_yfre
#         self.veh_path = veh_path
#         self.veh_target = veh_target
#         self.veh_has_preds = veh_has_preds
#         self.veh_edge_index = veh_edge_index
#         self.veh_edge_attr = veh_edge_attr

#         self.lane_id = lane_id
#         self.lane_ctrs = lane_ctrs
#         self.lane_vecs = lane_vecs
#         self.lane_pris = lane_pris
#         self.lane_suc_edge_index = lane_suc_edge_index
#         self.lane_pre_edge_index = lane_pre_edge_index
#         self.lane_left_edge_index = lane_left_edge_index
#         self.lane_right_edge_index = lane_right_edge_index
#         self.lane_start = lane_start
#         self.lane_path = lane_path

#         self.veh_full_path = veh_full_path
#         self.veh_path_edge_index = veh_path_edge_index
#         self.path_lane_edge_index = path_lane_edge_index
#         self.lane_widths=lane_widths
#         self.path_num_nodes = path_num_nodes
#         self.path_node_node_edge_index = path_node_node_edge_index

    def __inc__(self, key, value, *args):
        if "index" in key or "face" in key:
            if "veh" in key:
                return self.veh_shape.size(0)
            elif "lane" in key:
                return self.lane_ctrs.size(0)
            elif "path" in key:
                return self.path_num_nodes.size(0)
            else:
                return super(ScenarioData, self).__inc__(key, value)
        else:
            return 0

# from datasets.seen_data_files import all_val_files_2
# def mix_dataset(Config):
#     train_files, val_files_1, val_files_2 = [], [], []
#     for dataset in Config["datasets"]:
#         data_root_dir = Config[dataset]["data_dir"]
#         data_files = [osp.join(data_root_dir, "processed", d) for d in  os.listdir(osp.join(data_root_dir, "processed")) if 'data' in d]
#         if dataset in ["EP", "mcity"]:
#             val_files_1.extend(data_files)
#         else:
#             # tf, vf = train_test_split(data_files, test_size=0.15, random_state=42)
#             for file_name in data_files:
#                 if file_name in all_val_files_2:
#                     val_files_2.append(file_name)
#                 else:
#                     train_files.append(file_name)
#             # train_files.extend(tf)
#             # val_files_2.extend(vf)
#     return train_files, val_files_1, val_files_2

# def load_ngsim_scenarios(lanelet_network_filepaths, trajectory_filepaths):

#     lanelet_networks = {}
#     # load lanelet_networks
#     for fp in lanelet_network_filepaths:
#         if "i80" in fp:
#             lanelet_networks["i80"] = LaneletCurveNetworkReader(fp).lanelet_network
#         elif "u101" in fp:
#             u101_lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
#             speed_limits = {ln.lanelet_id:39.89 for ln in u101_lanelet_network.lanelets}
#             lanelet_networks["u101"] = make_lanelet_curve_network(u101_lanelet_network, speed_limits)
#         elif "highD2" in fp:
#             lanelet_networks["highD2"] = LaneletCurveNetworkReader(fp).lanelet_network
#         elif "highD3" in fp:
#             lanelet_networks["highD3"] = LaneletCurveNetworkReader(fp).lanelet_network
#         elif "rounD-plus" in fp:
#             rounD_lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
#             speed_limits = {ln.lanelet_id:13.89 for ln in rounD_lanelet_network.lanelets}
#             if "00" in fp:
#                 lanelet_networks["rounD-plus00"] = make_lanelet_curve_network(rounD_lanelet_network, speed_limits)
#             else:
#                 lanelet_networks["rounD-plus01"] = make_lanelet_curve_network(rounD_lanelet_network, speed_limits)
#         elif "rounD" in fp:
#             rounD_lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
#             speed_limits = {ln.lanelet_id:13.89 for ln in rounD_lanelet_network.lanelets}
#             lanelet_networks["rounD"] = make_lanelet_curve_network(rounD_lanelet_network, speed_limits)
#         elif "CHN" in fp:
#             lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
#             speed_limits = {ln.lanelet_id:17.89 for ln in lanelet_network.lanelets}
#             lanelet_networks["CHN"] = make_lanelet_curve_network(lanelet_network, speed_limits)
#         elif "DEU" in fp:
#             lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
#             speed_limits = {ln.lanelet_id:17.89 for ln in lanelet_network.lanelets}
#             lanelet_networks["DEU"] = make_lanelet_curve_network(lanelet_network, speed_limits)
#         elif "SR" in fp:
#             lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
#             speed_limits = {ln.lanelet_id:17.89 for ln in lanelet_network.lanelets}
#             lanelet_networks["SR"] = make_lanelet_curve_network(lanelet_network, speed_limits)
#         elif "EP" in fp:
#             lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
#             speed_limits = {ln.lanelet_id:17.89 for ln in lanelet_network.lanelets}
#             lanelet_networks["EP"] = make_lanelet_curve_network(lanelet_network, speed_limits)
#         elif "FT" in fp:
#             lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
#             speed_limits = {ln.lanelet_id:17.89 for ln in lanelet_network.lanelets}
#             lanelet_networks["FT"] = make_lanelet_curve_network(lanelet_network, speed_limits)
#         elif "mcity" in fp:
#             mcity_lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
#             speed_limits = {ln.lanelet_id:13.89 for ln in mcity_lanelet_network.lanelets}
#             lanelet_networks["mcity"] = make_lanelet_curve_network(mcity_lanelet_network, speed_limits)
#         else:
#             raise ValueError("Can not identify lanelet_network in {}".format(fp))

#     trajectories = []
#     vehicleinfo = []
#     # load trajectories
#     for fp in trajectory_filepaths:
#         trajdata = pickle.load(open(fp, "rb"))

#         obstacle_infos = {}
#         obstacle_states = defaultdict(lambda:dict())
#         for d in trajdata['def']:
#             carid, length, width, tp, f_lo, f_hi = d
#             obstacle_infos[int(carid)] = {"shape":Rectangle(length,width),
#                                         "type":tp, "frames":(int(f_lo),int(f_hi))}
#         for d in trajdata['state']:
#             step, carid, x, y, ori, v, i, t, lid, s, d, phi = d
#             state = FrenetState(position=np.array([x,y]), orientation=ori, velocity=v, time_step = int(step))
#             posF = Frenet(None, None, (i, t, lid, s, d, phi))
#             state.posF = posF
#             obstacle_states[int(step)][int(carid)] = state

#         trajectories.append(obstacle_states)
#         if "i80" in fp:
#             lanelane_network_id = "i80"
#         elif "u101" in fp:
#             lanelane_network_id = "u101"
#         elif "highD2" in fp:
#             lanelane_network_id = "highD2"
#         elif "highD3" in fp:
#             lanelane_network_id = "highD3"
#         elif "rounD-plus" in fp:
#             if "00" in fp:
#                 lanelane_network_id = "rounD-plus00"
#             else:
#                 lanelane_network_id = "rounD-plus01"
#         elif "rounD" in fp:
#             lanelane_network_id = "rounD"
#         elif "mcity" in fp:
#             lanelane_network_id = "mcity"
#         elif "CHN" in fp:
#             lanelane_network_id = "CHN"
#         elif "DEU" in fp:
#             lanelane_network_id = "DEU"
#         elif "SR" in fp:
#             lanelane_network_id = "SR"
#         elif "EP" in fp:
#             lanelane_network_id = "EP"
#         elif "FT" in fp:
#             lanelane_network_id = "FT"
#         else:
#             raise ValueError("Can not identify trajectory in {}".format(fp))

#         vehicleinfo.append((lanelane_network_id, obstacle_infos))

#     return trajectories, vehicleinfo, lanelet_networks



# def route_collate(batch):
#     batch_size = len(batch)
#     graphs = []

#     ctrs = []
#     actor_idcs = []
#     count = 0
#     for i in range(batch_size):
#         idcs = torch.arange(count, count + len(batch[i].veh_xseq))
#         actor_idcs.append(idcs.cuda())
#         count += len(batch[i].veh_xseq)
#         ctrs.append(batch[i].veh_xseq[:,-1,:2].cuda())

#         graph = {"ctrs":batch[i].lane_ctrs.cuda(),
#                  "feats":batch[i].lane_vecs.cuda(),
#                  "pris":batch[i].lane_pris.cuda(),
#                  "ids":batch[i].lane_id.cuda(),
#                  "num_nodes":batch[i].lane_ctrs.size(0),
#                  "pre":{k:v.cuda() for k, v in batch[i].lane_pre_edge_index.items()},
#                  "suc":{k:v.cuda() for k, v in batch[i].lane_suc_edge_index.items()},
#                  "left":batch[i].lane_left_edge_index.cuda(),
#                  "right":batch[i].lane_right_edge_index.cuda()}
#         graphs.append(graph)
#         batch[i].lane_suc_edge_index = None
#         batch[i].lane_pre_edge_index = None
#         batch[i].lane_left_edge_index = None
#         batch[i].lane_left_edge_index = None
#         batch[i].lane_right_edge_index = None
#         batch[i].lane_ctrs = None
#         batch[i].lane_vecs = None
#         batch[i].lane_pris = None
#         batch[i].lane_id = None
#     batch = Batch.from_data_list(batch)

#     batch['obs_traj'] = batch.veh_xseq[:,:,:2].permute(1, 0, 2).cuda()
#     batch['obs_traj_rel'] = batch.veh_xseq[:,:,2:4].permute(1, 0, 2).cuda()
#     batch['obs_info'] = batch.veh_xseq[:,:,4:].permute(1, 0, 2).cuda()
#     batch['obs_shape'] = batch.veh_shape.cuda()
#     batch['fut_traj'] = batch.veh_yseq[:,:,:2].permute(1, 0, 2).cuda()
#     batch['fut_traj_rel'] = batch.veh_yseq[:,:,2:].permute(1, 0, 2).cuda()
#     batch['has_preds'] = batch.veh_has_preds.permute(1, 0).cuda()
#     batch['obs_linear'] = batch.veh_x.cuda()
#     batch['edge_index'] = batch.veh_edge_index.cuda()
#     batch['veh_batch'] = actor_idcs
#     batch['graphs'] = graphs
#     batch['veh_ctrs'] = ctrs
#     return batch

# def local_search(nodes, edges, mode):
#     g = nx.DiGraph()
#     for n in nodes:
#         g.add_node(n)
#     for j in range(edges.size(-1)):
#         src, des = edges[0][j].item(), edges[1][j].item()
#         if g.has_node(src) and g.has_node(des):
#             g.add_edge(src, des)
#     #print(g.edges)
#     cands = []
#     for n in nodes:
#         if len(cands) == 0:
#             cands.append(n)
#             continue
#         else:
#             if mode == "start":
#                 path_lengths = nx.multi_source_dijkstra_path_length(g, cands)
#                 if n in path_lengths:
#                     continue
#                 else:
#                     cands.append(n)

#             else:
#                 path_lengths = nx.single_source_dijkstra_path_length(g, n)
#                 for s in cands:
#                     if s in path_lengths:
#                         break
#                 else:
#                     cands.append(n)
#     return cands

# def vector_collate(batch):
#     batch_size = len(batch)
#     seq_len = batch[0].veh_xseq.size(1)
#     num_preds = batch[0].veh_yseq.size(1)

#     lane_feats = []
#     lane_clusters = []
#     batch_lane = []
#     veh_feats = []
#     veh_clusters = []
#     batch_veh = []

#     candidate, offset_gt, target_gt = [], [], []
#     candidate_mask = []
#     candidate_gt = []

#     lane_counts = 0
#     veh_counts = 0
#     valid_lens = []
#     veh_lens = []

#     actor_ctrs = []
#     actor_idcs = []
#     count = 0
#     graphs = []
#     for i in range(batch_size):
#         idcs = torch.arange(count, count + len(batch[i].veh_xseq))
#         actor_idcs.append(idcs.cuda())
#         count += len(batch[i].veh_xseq)
#         actor_ctrs.append(batch[i].veh_xseq[:,-1,:2].cuda())
#         graph = {"ctrs":batch[i].lane_ctrs.cuda(),
#                  "feats":torch.cat([batch[i].lane_vecs, batch[i].lane_widths],-1).cuda(),
#                  "pris":batch[i].lane_pris.cuda(),
#                  "ids":batch[i].lane_id.cuda(),
#                  "num_nodes":batch[i].lane_ctrs.size(0),
#                  "num_paths":batch[i].path_num_nodes.size(0),
#                  "pre":{k:v.cuda() for k, v in batch[i].lane_pre_edge_index.items()},
#                  "suc":{k:v.cuda() for k, v in batch[i].lane_suc_edge_index.items()},
#                  "path":{k:v.cuda() for k, v in batch[i].lane_path.items()},
#                  #"path_weight":{k:v.cuda() for k, v in batch[i].lane_path_weight.items()},
#                  "left":batch[i].lane_left_edge_index.cuda(),
#                  "right":batch[i].lane_right_edge_index.cuda(),
#                  "start":batch[i].lane_start.cuda(),

#                  "path->node":batch[i].path_lane_edge_index.cuda(),
#                  "path-node->path-node":[e.cuda() for e in batch[i].path_node_node_edge_index],
#                  "veh->path":batch[i].veh_path_edge_index.cuda(),
#                  #"target":batch[i].lane_target.cuda(),
#                  }
#         graphs.append(graph)
#         lane_feat = torch.cat([batch[i].lane_ctrs,
#                           batch[i].lane_vecs,
#                           batch[i].lane_widths,
#                           batch[i].lane_pris], -1)
#         lane_feats.append(lane_feat)
#         lane_cluster = torch.zeros_like(batch[i].lane_id, dtype=torch.long)
#         unique_lane_ids = batch[i].lane_id.unique()
#         for j, id in enumerate(unique_lane_ids):
#             lane_cluster[batch[i].lane_id==id] = j+lane_counts
#         lane_counts += len(unique_lane_ids)
#         lane_clusters.append(lane_cluster)
#         batch_lane.append(torch.ones_like(unique_lane_ids, dtype=torch.long)*i)


#         veh_info = batch[i].veh_xseq[:,:,4]
#         num_vehs = batch[i].veh_xseq.size(0)
#         timestamps = torch.arange(seq_len).float().repeat(num_vehs).reshape(num_vehs, seq_len, 1)
#         veh_cluster = torch.arange(num_vehs).repeat(seq_len).reshape(seq_len, num_vehs).permute(1,0) + veh_counts
#         veh_feat = torch.cat([batch[i].veh_xseq[:, :, :2],
#                               timestamps], -1)
#         veh_feats.append(veh_feat[veh_info==1])
#         veh_counts += num_vehs
#         veh_clusters.append(veh_cluster[veh_info==1])
#         batch_veh.append(torch.ones(num_vehs, dtype=torch.long)*i)

#         valid_lens.append(num_vehs+len(unique_lane_ids))
#         veh_lens.append(num_vehs)

#         last = batch[i].veh_has_preds.float() + 0.1 * torch.arange(num_preds).float() / float(num_preds)
#         max_last, last_idcs = last.max(1)
#         row_idcs = torch.arange(len(last_idcs)).long()

#         veh_target_ctrs = batch[i].veh_yseq[row_idcs,last_idcs]
#         veh_target_ctrs = veh_target_ctrs[:,:2]
#         lane_ctrs = batch[i].lane_ctrs

#         #src = [batch[i].lane_path[k][0] for k in range(5)]
#         #des = [batch[i].lane_path[k][1] for k in range(5)]
#         #src = torch.cat(src,0)
#         #des = torch.cat(des,0)



#         for v in range(num_vehs):
#             veh_path_idcs = batch[i].veh_path_edge_index[1][batch[i].veh_path_edge_index[0] == v]
#             lctrs = []
#             unique_lane_idcs = []
#             for p in veh_path_idcs:
#                 path = lane_ctrs[batch[i].path_lane_edge_index[1][batch[i].path_lane_edge_index[0]==p]]
#                 lane_idcs = batch[i].path_lane_edge_index[1][batch[i].path_lane_edge_index[0]==p].numpy()
#                 if path.size(0) < 15:
#                     delta_xy = path[-1] - path[-2]
#                     add_points = torch.cumsum(delta_xy.repeat(15-path.size(0),1), dim=0) + path[-1]
#                     lane_idcs = np.concatenate([lane_idcs, np.ones(15-path.size(0))*lane_idcs[-1]], axis=0)
#                     path = torch.cat([path, add_points], dim=0)
#                 else:
#                     path = path[:15]
#                     lane_idcs = lane_idcs[:15]

#                 '''
#                 lctrs.append(path)
#                 lctrs.append((path[1:] - path[:-1] )*1/5 +path[:-1])
#                 lctrs.append((path[1:] - path[:-1] )*2/5 +path[:-1])
#                 lctrs.append((path[1:] - path[:-1] )*3/5 +path[:-1])
#                 lctrs.append((path[1:] - path[:-1] )*4/5 +path[:-1])
#                 '''
#                 if len(unique_lane_idcs) == 0:
#                     lctrs.append(path)
#                     lctrs.append((path[1:] - path[:-1] )*1/5 +path[:-1])
#                     lctrs.append((path[1:] - path[:-1] )*2/5 +path[:-1])
#                     lctrs.append((path[1:] - path[:-1] )*3/5 +path[:-1])
#                     lctrs.append((path[1:] - path[:-1] )*4/5 +path[:-1])
#                 else:
#                     mask = torch.ones(15, dtype=torch.bool)
#                     for lane_idc in unique_lane_idcs:
#                         mask = mask & (lane_idcs != lane_idc)
#                     mask = mask.bool()
#                     lctrs.append(path[mask])
#                     lctrs.append(((path[1:] - path[:-1] )*1/5 +path[:-1])[mask[:14]])
#                     lctrs.append(((path[1:] - path[:-1] )*2/5 +path[:-1])[mask[:14]])
#                     lctrs.append(((path[1:] - path[:-1] )*3/5 +path[:-1])[mask[:14]])
#                     lctrs.append(((path[1:] - path[:-1] )*4/5 +path[:-1])[mask[:14]])
#                 for lane_idc in lane_idcs:
#                     if lane_idc not in unique_lane_idcs:
#                         unique_lane_idcs.append(lane_idc)

#             lctrs = torch.cat(lctrs, dim=0)
#             #lctrs = lane_ctrs[des[src==v]]
#             candidate.append(lctrs)
#             dist = (lctrs - veh_target_ctrs[v])**2
#             dist = dist.sum(-1)
#             candidate_gt.append(dist.argmin())
#             offset_gt.append(veh_target_ctrs[v] - lctrs[dist.argmin()])

#             candidate_mask.append(len(candidate[-1]))

#         target_gt.append(veh_target_ctrs)

#         batch[i].lane_suc_edge_index = None
#         batch[i].lane_pre_edge_index = None
#         batch[i].lane_left_edge_index = None
#         batch[i].lane_left_edge_index = None
#         batch[i].lane_right_edge_index = None

#         batch[i].path_lane_edge_index = None
#         batch[i].path_node_node_edge_index = None
#         batch[i].veh_path_edge_index = None

#         batch[i].lane_path = None
#         batch[i].lane_start = None
#         batch[i].lane_ctrs = None
#         batch[i].lane_vecs = None
#         batch[i].lane_pris = None
#         batch[i].lane_id = None
#         #batch[i].veh_xseq = None

#         batch[i].veh_yfre = None
#         batch[i].veh_path = None
#         batch[i].veh_full_path = None
#         batch[i].veh_target = None
#         batch[i].veh_x = None
#         batch[i].veh_edge_index = None


#     batch = Batch.from_data_list(batch)

#     max_candidate_len = max(candidate_mask)
#     for i in range(len(candidate)):
#         zeros = torch.zeros(max_candidate_len-len(candidate[i]),2, dtype=torch.float)
#         candidate[i] = torch.cat([candidate[i], zeros], 0)

#     batch['candidate'] = torch.stack(candidate, 0).cuda()
#     batch['offset_gt'] = torch.stack(offset_gt, 0).cuda()
#     batch['target_gt'] = torch.cat(target_gt, 0).cuda()
#     batch['candidate_gt'] = torch.stack(candidate_gt, 0).cuda()
#     batch['candidate_mask'] = candidate_mask

#     batch['veh_ctrs'] = actor_ctrs #torch.cat(actor_ctrs, dim=0).cuda()
#     batch['obs_traj'] = batch.veh_xseq[:,:,:2].permute(1, 0, 2).cuda()
#     batch['veh_feat'] = torch.cat(veh_feats, 0).cuda()
#     batch['veh_cluster'] = torch.cat(veh_clusters, 0).cuda()
#     batch['batch_veh'] = torch.cat(batch_veh, 0).cuda()
#     batch['lane_feat'] = torch.cat(lane_feats, 0).cuda()
#     batch['lane_cluster'] = torch.cat(lane_clusters, 0).cuda()
#     batch['batch_lane'] = torch.cat(batch_lane, 0).cuda()
#     batch['fut_traj'] = batch.veh_yseq[:,:,:2].permute(1, 0, 2).cuda()
#     batch['fut_traj_rel'] = batch.veh_yseq[:,:,2:].permute(1, 0, 2).cuda()
#     batch['has_preds'] = batch.veh_has_preds.permute(1, 0).cuda()
#     batch['valid_lens'] = valid_lens
#     batch['veh_lens'] = veh_lens
#     batch['obs_traj_rel'] = batch.veh_xseq[:,:,2:4].permute(1, 0, 2).cuda()
#     batch['obs_info'] = batch.veh_xseq[:,:,4:].permute(1, 0, 2).cuda()
#     batch['veh_batch'] = actor_idcs
#     batch['graphs'] = graphs

#     return batch

# def traj_collate(batch):
#     batch_size = len(batch)
#     graphs = []

#     ctrs = []
#     actor_idcs = []
#     count = 0

#     actor_paths = []
#     for i in range(batch_size):
#         idcs = torch.arange(count, count + len(batch[i].veh_xseq))
#         actor_idcs.append(idcs.cuda())
#         count += len(batch[i].veh_xseq)
#         ctrs.append(batch[i].veh_xseq[:,-1,:2].cuda())

#         # add the path edge_indexs
#         graph = {"ctrs":batch[i].lane_ctrs.cuda(),
#                  "feats":torch.cat([batch[i].lane_vecs, batch[i].lane_widths],-1).cuda(),
#                  "pris":batch[i].lane_pris.cuda(),
#                  "ids":batch[i].lane_id.cuda(),
#                  "num_nodes":batch[i].lane_ctrs.size(0),
#                  "num_paths":batch[i].path_num_nodes.size(0),
#                  "pre":{k:v.cuda() for k, v in batch[i].lane_pre_edge_index.items()},
#                  "suc":{k:v.cuda() for k, v in batch[i].lane_suc_edge_index.items()},
#                  "path":{k:v.cuda() for k, v in batch[i].lane_path.items()},
#                  #"path_weight":{k:v.cuda() for k, v in batch[i].lane_path_weight.items()},
#                  "left":batch[i].lane_left_edge_index.cuda(),
#                  "right":batch[i].lane_right_edge_index.cuda(),
#                  "start":batch[i].lane_start.cuda(),

#                  "path->node":batch[i].path_lane_edge_index.cuda(),
#                  "path-node->path-node":[e.cuda() for e in batch[i].path_node_node_edge_index],
#                  "veh->path":batch[i].veh_path_edge_index.cuda(),
#                  #"target":batch[i].lane_target.cuda(),
#                  }
#         graphs.append(graph)
#         batch[i].lane_suc_edge_index = None
#         batch[i].lane_pre_edge_index = None
#         batch[i].lane_left_edge_index = None
#         batch[i].lane_left_edge_index = None
#         batch[i].lane_right_edge_index = None

#         #batch[i].path_lane_edge_index = None
#         batch[i].path_node_node_edge_index = None
#         #batch[i].veh_path_edge_index = None

#         batch[i].lane_path = None
#         #batch[i].lane_path_weight = None
#         batch[i].lane_start = None
#         #batch[i].lane_target = None
#         #batch[i].lane_ctrs = None
#         batch[i].lane_vecs = None
#         batch[i].lane_pris = None
#         batch[i].lane_id = None
#     batch = Batch.from_data_list(batch)

#     dpos_xy1 = batch.veh_xseq[:, -2, 2:4]
#     dpos_xy0 = batch.veh_xseq[:, -1, 2:4]

#     d1 = torch.sqrt(torch.sum(dpos_xy1**2, dim=-1))
#     d0 = torch.sqrt(torch.sum(dpos_xy0**2, dim=-1))
#     info1 = batch.veh_xseq[:,-2,-1]

#     batch['obs_head'] = torch.atan2(dpos_xy0[:,1], dpos_xy0[:,0])
#     batch['obs_acc'] = (d0-d1)/0.04
#     batch['obs_acc'][info1==0] = 0

#     batch['obs_traj'] = batch.veh_xseq[:,:,:2].permute(1, 0, 2).cuda()
#     batch['obs_traj_rel'] = batch.veh_xseq[:,:,2:4].permute(1, 0, 2).cuda()
#     batch['obs_info'] = batch.veh_xseq[:,:,4:].permute(1, 0, 2).cuda()
#     batch['obs_shape'] = batch.veh_shape.cuda()
#     batch['fut_traj'] = batch.veh_yseq[:,:,:2].permute(1, 0, 2).cuda()
#     batch['fut_traj_rel'] = batch.veh_yseq[:,:,2:].permute(1, 0, 2).cuda()
#     batch['fut_traj_fre'] = batch.veh_yfre.permute(1, 0, 2).type('torch.cuda.FloatTensor')
#     batch['fut_path'] = batch.veh_path.squeeze().cuda()
#     batch.veh_full_path = batch.veh_full_path.cuda()
#     batch['fut_target'] = batch.veh_target.squeeze().cuda()
#     batch['has_preds'] = batch.veh_has_preds.permute(1, 0).cuda()
#     #batch['obs_linear'] = batch.veh_x.cuda()
#     batch['edge_index'] = batch.veh_edge_index.cuda()
#     batch['veh_batch'] = actor_idcs
#     batch['graphs'] = graphs
#     batch['veh_ctrs'] = ctrs
#     return batch

# def gpu(data):
#     """
#     Transfer tensor in `data` to gpu recursively
#     `data` can be dict, list or tuple
#     """
#     if isinstance(data, list) or isinstance(data, tuple):
#         data = [gpu(x) for x in data]
#     elif isinstance(data, dict):
#         data = {key:gpu(_data) for key,_data in data.items()}
#     elif isinstance(data, torch.Tensor):
#         data = data.contiguous().cuda(non_blocking=True)
#     return data


# def to_long(data):
#     if isinstance(data, dict):
#         for key in data.keys():
#             data[key] = to_long(data[key])
#     if isinstance(data, list) or isinstance(data, tuple):
#         data = [to_long(x) for x in data]
#     if torch.is_tensor(data) and data.dtype == torch.int16:
#         data = data.long()
#     return data

# def transform(data, config=None):
#     data.x -= torch.cat([config["veh_x_mean"], config["veh_x_mean"], config["veh_x_mean"]], -1)
#     data.x /= torch.cat([config["veh_x_std"],config["veh_x_std"],config["veh_x_std"]],-1)
#     data.xseq -= config["veh_x_mean"]
#     data.xseq /= config["veh_x_std"]

#     data.y -= config["veh_y_mean"]
#     data.y /= config["veh_y_std"]

#     return data

# def relative_to_curve_with_curvature(traj_fre, last_pos, vertices):
#         # traj_fre #(S, A, 2)
#     last_pos = last_pos.cpu().numpy()

#     traj_pred_rel = []
#     traj_pred_curv = []
#     for idx in range(traj_fre.size(1)):
#         target_curve = make_curve(vertices[idx].cpu().numpy())
#         start_pos = VecSE2(last_pos[idx, 0], last_pos[idx, 1], 0.0)
#         if len(target_curve) > 10:
#             bound_ind = len(target_curve)//3
#         else:
#             bound_ind = len(target_curve)
#         closest_ind = start_pos.index_closest_to_point(target_curve[:bound_ind])
#         while closest_ind >= bound_ind-1:
#             bound_ind +=1
#             closest_ind = start_pos.index_closest_to_point(target_curve[:bound_ind])

#         start_proj = start_pos.proj_on_curve(target_curve[:bound_ind], clamped=False)
#         start_s = lerp_curve_with_ind(target_curve, start_proj.ind)
#         start_ind = start_proj.ind
#         traj = []
#         traj_curv = []
#         for t in range(traj_fre.size(0)):
#             next_ind, next_pos, curvature  = move_along_curve(start_ind, target_curve, traj_fre[t, idx, 0].cpu().item(), traj_fre[t, idx, 1].cpu().item(), include_curvature=True)
#             traj.append([next_pos.x-start_pos.x, next_pos.y-start_pos.y])
#             traj_curv.append(curvature)
#             start_ind = next_ind
#             start_pos = next_pos
#         traj_pred_rel.append(traj)
#         traj_pred_curv.append(traj_curv)

#     return torch.tensor(traj_pred_rel, dtype=torch.float32).cuda().permute(1, 0, 2), torch.tensor(traj_pred_curv, dtype=torch.float32).cuda().permute(1, 0)

# def relative_to_curve(traj_fre, last_pos, vertices):
#     # traj_fre #(S, A, 2)
#     last_pos = last_pos.cpu().numpy()

#     traj_pred_rel = []
#     for idx in range(traj_fre.size(1)):
#         target_curve = make_curve(vertices[idx].cpu().numpy())
#         start_pos = VecSE2(last_pos[idx, 0], last_pos[idx, 1], 0.0)
#         if len(target_curve) > 10:
#             bound_ind = len(target_curve)//3
#         else:
#             bound_ind = len(target_curve)
#         closest_ind = start_pos.index_closest_to_point(target_curve[:bound_ind])
#         while closest_ind >= bound_ind-1:
#             bound_ind +=1
#             closest_ind = start_pos.index_closest_to_point(target_curve[:bound_ind])

#         start_proj = start_pos.proj_on_curve(target_curve[:bound_ind], clamped=False)
#         start_s = lerp_curve_with_ind(target_curve, start_proj.ind)
#         start_ind = start_proj.ind
#         traj = []
#         for t in range(traj_fre.size(0)):
#             next_ind, next_pos = move_along_curve(start_ind, target_curve, traj_fre[t, idx, 0].cpu().item(), traj_fre[t, idx, 1].cpu().item())
#             traj.append([next_pos.x-start_pos.x, next_pos.y-start_pos.y])
#             start_ind = next_ind
#             start_pos = next_pos
#         traj_pred_rel.append(traj)

#     return torch.tensor(traj_pred_rel, dtype=torch.float32).cuda().permute(1, 0, 2)

# def relative_to_abs(rel_traj, start_pos):
#     """
#     Inputs:
#     - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
#     - start_pos: pytorch tensor of shape (batch, 2)
#     Outputs:
#     - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
#     """
#     # batch, seq_len, 2
#     rel_traj = rel_traj.permute(1, 0, 2)
#     displacement = torch.cumsum(rel_traj, dim=1)
#     start_pos = torch.unsqueeze(start_pos, dim=1)
#     abs_traj = displacement + start_pos
#     return abs_traj.permute(1, 0, 2)


# def check_accuracy(dataloader, config, model):
#     d_losses = []
#     metrics = {}
#     g_l2_losses_abs, g_l2_losses_rel = [], []
#     disp_error, f_disp_error, miss_rate = [[] for _ in range(3)], [[] for _ in range(3)], [[] for _ in range(3)]

#     total_final_preds, total_preds = [0 for _ in range(3)], [0 for _ in range(3)]

#     best_disp_error, best_f_disp_error = [[] for _ in range(3)], [[] for _ in range(3)]


#     K_total_preds, K_total_final_preds = [0 for _ in range(3)], [0 for _ in range(3)]
#     K_disp_error, K_f_disp_error, K_miss_rate =  [[] for _ in range(3)], [[] for _ in range(3)], [[] for _ in range(3)]
#     diversities = 0.0
#     diversities_N = 0
#     path_acc, target_errors, slc_acc = [], [], []

#     loss_mask_sum = 0
#     model.eval()
#     with torch.no_grad():
#         for batch in dataloader:
#             ret = model(batch)
#             if "frenet_pred" in ret:
#                 frenet_pred = ret["frenet_pred"]
#                 curves = ret["curves"]
#                 curves_gt = ret["curves_gt"]
#                 if "converted_pred" in ret:
#                     pred_traj_fake_rel = ret["converted_pred"]
#                 else:
#                     pred_traj_fake_rel = relative_to_curve(frenet_pred, batch.obs_traj[-1], curves)
#                 pred_traj_fake = relative_to_abs(pred_traj_fake_rel, batch.obs_traj[-1])
#                 if "converted_gt" in ret:
#                     pred_traj_gt_rel = ret["converted_gt"]
#                 else:
#                     pred_traj_gt_rel = relative_to_curve(batch.fut_traj_fre, batch.obs_traj[-1], curves_gt)
#                 pred_traj_gt = relative_to_abs(pred_traj_gt_rel, batch.obs_traj[-1])

#                 g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
#                     pred_traj_gt, pred_traj_gt_rel, batch.has_preds, pred_traj_fake, pred_traj_fake_rel
#                 )
#                 g_l2_losses_abs.append(g_l2_loss_abs.item())
#                 g_l2_losses_rel.append(g_l2_loss_rel.item())


#                 for tt in range(3):
#                     ade = cal_ade(
#                         pred_traj_gt[:config["num_preds"]//3*(tt+1)-1],
#                         pred_traj_fake[:config["num_preds"]//3*(tt+1)-1],
#                         batch.has_preds[:config["num_preds"]//3*(tt+1)-1]
#                     )
#                     fde = cal_fde(
#                         pred_traj_gt[:config["num_preds"]//3*(tt+1)-1],
#                         pred_traj_fake[:config["num_preds"]//3*(tt+1)-1],
#                         batch.has_preds[:config["num_preds"]//3*(tt+1)-1]
#                     )
#                     mr = cal_mr(
#                         pred_traj_gt[:config["num_preds"]//3*(tt+1)-1],
#                         pred_traj_fake[:config["num_preds"]//3*(tt+1)-1],
#                         batch.has_preds[:config["num_preds"]//3*(tt+1)-1]
#                     )
#                     disp_error[tt].append(ade.item())
#                     f_disp_error[tt].append(fde.item())
#                     miss_rate[tt].append(mr.item())

#                     total_preds[tt] += batch.has_preds[:config["num_preds"]//3*(tt+1)-1].sum()
#                     total_final_preds[tt] += batch.has_preds[config["num_preds"]//3*(tt+1)-1].sum()



#             if "pred" in ret:
#                 pred_traj_fake_rel = ret['pred']
#                 pred_traj_fake = relative_to_abs(pred_traj_fake_rel, batch.obs_traj[-1])

#                 g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
#                     batch.fut_traj, batch.fut_traj_rel, batch.has_preds, pred_traj_fake, pred_traj_fake_rel
#                 )
#                 g_l2_losses_abs.append(g_l2_loss_abs.item())
#                 g_l2_losses_rel.append(g_l2_loss_rel.item())

#                 for tt in range(3):
#                     ade = cal_ade(
#                         batch.fut_traj[:config["num_preds"]//3*(tt+1)-1],
#                         pred_traj_fake[:config["num_preds"]//3*(tt+1)-1],
#                         batch.has_preds[:config["num_preds"]//3*(tt+1)-1]
#                     )
#                     fde = cal_fde(
#                         batch.fut_traj[:config["num_preds"]//3*(tt+1)-1],
#                         pred_traj_fake[:config["num_preds"]//3*(tt+1)-1],
#                         batch.has_preds[:config["num_preds"]//3*(tt+1)-1]
#                     )
#                     mr = cal_mr(
#                         batch.fut_traj[:config["num_preds"]//3*(tt+1)-1],
#                         pred_traj_fake[:config["num_preds"]//3*(tt+1)-1],
#                         batch.has_preds[:config["num_preds"]//3*(tt+1)-1]
#                     )
#                     disp_error[tt].append(ade.item())
#                     f_disp_error[tt].append(fde.item())
#                     miss_rate[tt].append(mr.item())
#                     total_preds[tt] += batch.has_preds[:config["num_preds"]//3*(tt+1)-1].sum()
#                     total_final_preds[tt] += batch.has_preds[config["num_preds"]//3*(tt+1)-1].sum()

#             if "best" in ret:
#                 best_pred_traj_fake_rel = ret['best']
#                 best_pred_traj_fake = relative_to_abs(best_pred_traj_fake_rel, batch.obs_traj[-1])

#                 for tt in range(3):
#                     best_ade = cal_ade(
#                         batch.fut_traj[:config["num_preds"]//3*(tt+1)-1],
#                         best_pred_traj_fake[:config["num_preds"]//3*(tt+1)-1],
#                         batch.has_preds[:config["num_preds"]//3*(tt+1)-1]
#                     )
#                     best_fde = cal_fde(
#                         batch.fut_traj[:config["num_preds"]//3*(tt+1)-1],
#                         best_pred_traj_fake[:config["num_preds"]//3*(tt+1)-1],
#                         batch.has_preds[:config["num_preds"]//3*(tt+1)-1]
#                     )
#                     best_disp_error[tt].append(best_ade.item())
#                     best_f_disp_error[tt].append(best_fde.item())

#             if "path" in ret:
#                 path = ret["path"]
#                 acc_score = cal_acc(ret["path"][batch.has_preds[-1]], ret["gt_path"][batch.has_preds[-1]])
#                 path_acc.append(acc_score)

#             if "target" in ret:

#                 #path = ret["path"]
#                 #print("has ", batch.has_preds[-1].sum())
#                 #acc_score = cal_min_acc(path[batch.has_preds[-1]], batch.fut_path[batch.has_preds[-1]])
#                 #path_acc.append(acc_score)

#                 #target_pred = torch.stack([ret["target"][:, 0]*50,
#                 #                      ret["target"][:, 1]*13.5-5.5], dim=-1)
#                 target_pred = torch.stack([ret["target"][:, 0],
#                                       ret["target"][:, 1]], dim=-1)


#                 target_gt = batch.fut_target[:, [0,2]]
#                 target_error = target_gt[batch.has_preds[-1]] - target_pred[batch.has_preds[-1]]
#                 target_error = target_error**2
#                 target_error = torch.sqrt(target_error.sum(dim=-1)).sum()
#                 target_errors.append(target_error.item())
#                 #acc_score = cal_min_acc(target[batch.has_preds[-1]], batch.fut_target[batch.has_preds[-1]])
#                 #target_acc.append(acc_score)

#             #if "score" in ret and "target" in ret:
#             #    slc = ret['score']
#             #    target = ret["target"]
#             #    _, slc_idcs = slc.max(1)
#             #    row_idcs = torch.arange(len(slc_idcs)).long().to(slc_idcs.device)

#             #    slc_score = cal_acc(target[row_idcs, slc_idcs][batch.has_preds[-1]], batch.fut_target[batch.has_preds[-1]])
#             #    slc_acc.append(slc_score)

#             if "reg" in ret:
#                 # K = 6
#                 K_pred_traj_fake_rel = ret['reg']
#                 ades = [[], [], []]
#                 fdes = [[], [], []]
#                 mrs = [[], [], []]
#                 K_pred_traj_final = []
#                 if "curves_gt" in ret:
#                     curves_gt = ret["curves_gt"]
#                     pred_traj_gt_rel = relative_to_curve(batch.fut_traj_fre, batch.obs_traj[-1], curves_gt)
#                     pred_traj_gt = relative_to_abs(pred_traj_gt_rel, batch.obs_traj[-1])
#                 else:
#                     pred_traj_gt = batch.fut_traj

#                 for k in range(config['num_mods']):
#                     pred_rel = K_pred_traj_fake_rel[k].permute(1, 0, 2)
#                     pred_traj = relative_to_abs(pred_rel, batch.obs_traj[-1])

#                     K_pred_traj_final.append(pred_traj[-1])
#                     for tt in range(3):
#                         ade = cal_ade(
#                             pred_traj_gt[:config["num_preds"]//3*(tt+1)-1],
#                             pred_traj[:config["num_preds"]//3*(tt+1)-1],
#                             batch.has_preds[:config["num_preds"]//3*(tt+1)-1], mode='raw'
#                         )
#                         fde = cal_fde(
#                             pred_traj_gt[:config["num_preds"]//3*(tt+1)-1],
#                             pred_traj[:config["num_preds"]//3*(tt+1)-1],
#                             batch.has_preds[:config["num_preds"]//3*(tt+1)-1], mode='raw'
#                         )
#                         mr = cal_mr(
#                             pred_traj_gt[:config["num_preds"]//3*(tt+1)-1],
#                             pred_traj[:config["num_preds"]//3*(tt+1)-1],
#                             batch.has_preds[:config["num_preds"]//3*(tt+1)-1], mode='raw'
#                         )
#                         ades[tt].append(ade)
#                         fdes[tt].append(fde)
#                         mrs[tt].append(mr)
#                         K_total_preds[tt] += batch.has_preds[:config["num_preds"]//3*(tt+1)-1].sum()
#                         K_total_final_preds[tt] += batch.has_preds[config["num_preds"]//3*(tt+1)-1].sum()
#                 for tt in range(3):
#                     K_disp_error[tt].append(torch.stack(ades[tt], 1))
#                     K_f_disp_error[tt].append(torch.stack(fdes[tt], 1))
#                     K_miss_rate[tt].append(torch.stack(mrs[tt], 1))

#                 diversities += cal_diversity(torch.stack(K_pred_traj_final, 0))
#                 diversities_N += batch.has_preds.size(1)

#         if "best" in ret:
#             metrics['best_ade'] = [sum(best_disp_error[tt]) / total_preds[tt] for tt in range(3)]
#             metrics['best_fde'] = [sum(best_f_disp_error[tt]) / total_final_preds[tt] for tt in range(3)]


#         if "pred" in ret or "frenet_pred" in ret:
#             metrics['g_l2_loss_abs']  = [sum(g_l2_losses_abs) / total_preds[-1]]
#             metrics['ade'] = [sum(disp_error[tt]) / total_preds[tt] for tt in range(3)]
#             metrics['fde'] = [sum(f_disp_error[tt]) / total_final_preds[tt] for tt in range(3)]
#             metrics['mr'] = [sum(miss_rate[tt]) / total_final_preds[tt] for tt in range(3)]

#         if "path" in ret:
#             metrics['path'] = sum(path_acc) / total_final_preds[-1]

#         if "target" in ret:
#             metrics['target'] = sum(target_errors) / total_final_preds[-1]
#             #metrics['path'] = sum(path_acc) / total_final_preds[-1]
#             #metrics['target'] = sum(target_acc) / total_final_preds[-1]
#             #print("p", sum(path_acc), path[:5], batch.fut_path[:5])
#             #print("t", sum(target_acc), target[:5], batch.fut_target[:5])
#             #print("t", total_final_preds[2])
#             #print("metric ", metrics['path'])
#             #print("logits ", path[batch.has_preds[-1]][:5], path[batch.has_preds[-1]][:5].max(1))
#             #print("gt ", batch.fut_path[batch.has_preds[-1]][:5])
#         #if "score" in ret and "target" in ret:
#         #    metrics['score'] = sum(slc_acc) / total_final_preds[-1]
#         #    print("s", sum(slc_acc), slc[:5], slc_idcs[:5])

#         if "reg" in ret:

#             metrics['kade'] = [torch.cat(K_disp_error[tt], 0).sum().item() / K_total_preds[tt] for tt in range(3)]
#             metrics['kfde'] = [torch.cat(K_f_disp_error[tt], 0).sum().item() / K_total_final_preds[tt] for tt in range(3)]

#             metrics['minkade'] = [torch.cat(K_disp_error[tt], 0).min(1)[0].sum().item()  / total_preds[tt] for tt in range(3)]
#             metrics['minkfde'] = [torch.cat(K_f_disp_error[tt], 0).min(1)[0].sum().item() / total_final_preds[tt] for tt in range(3)]

#             metrics['minkmr'] = [torch.cat(K_miss_rate[tt], 0).min(1)[0].sum().item() / total_final_preds[tt] for tt in range(3)]
#             metrics['diversity'] = diversities/diversities_N
#         model.train()
#         return metrics

# def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, has_preds, pred_traj_fake, pred_traj_fake_rel):
#     g_l2_loss_abs = l2_loss(
#         pred_traj_fake, pred_traj_gt, has_preds, mode='sum'
#     )
#     g_l2_loss_rel = l2_loss(
#         pred_traj_fake_rel, pred_traj_gt_rel, has_preds, mode='sum'
#     )
#     return g_l2_loss_abs, g_l2_loss_rel

# def cal_acc(pred, gt):
#     score = 0
#     #print(pred, pred.size())

#     #print(gt, gt.size())
#     #_, pred_indexs = pred.max(1)
#     #print(pred_indexs.size(), gt.size())
#     for pred_idx, gt_idx in zip(pred, gt):
#         if gt_idx == pred_idx:
#             score += 1
#     #print(score)
#     return score

# def cal_min_acc(pred, gt):
#     score = 0
#     for pred_idx, gt_idx in zip(pred, gt):
#         if gt_idx in pred_idx:
#             score += 1
#     return score


# def cal_ade(pred_traj_gt, pred_traj_fake, has_preds, mode='sum'):
#     ade = displacement_error(pred_traj_fake, pred_traj_gt, has_preds, mode=mode)
#     return ade
# def cal_fde(pred_traj_gt, pred_traj_fake, has_preds, mode='sum'):
#     fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], has_preds[-1], mode=mode)
#     return fde
# def cal_ede(pred_traj_gt, pred_traj_fake, has_preds):
#     seq_len = pred_traj_gt.size(0)
#     ede = []
#     for t in range(seq_len):
#         de = final_displacement_error(pred_traj_fake[t], pred_traj_gt[t], has_preds[t])
#         ede.append(de)
#     return ede

# def cal_mr(pred_traj_gt, pred_traj_fake, has_preds, mode='sum'):
#     loss = pred_traj_gt[-1][has_preds[-1]] - pred_traj_fake[-1][has_preds[-1]]
#     loss = loss**2
#     loss = torch.sqrt(loss.sum(dim=-1))
#     if mode == 'raw':
#         return loss > 2.0
#     else:
#         return torch.sum(loss > 2.0)

# def cal_diversity(K_pred_traj_final):
#     k, n, _  = K_pred_traj_final.size()
#     diversity = K_pred_traj_final.view(k, 1, n, 2) - K_pred_traj_final.view(1, k, n, 2)
#     diversity = diversity**2
#     diversity = torch.sqrt(diversity.sum(dim=-1))
#     i = torch.arange(k).view(1,k).repeat(k,1) > torch.arange(k).view(k,1).repeat(1,k)
#     return torch.sum(diversity[i].mean(0)).item()

# '''
# def cal_kde_nll(K_pred_traj, pred_traj_gt):
#     k, n, l,_ = K_pred_traj.size()
#     for node in range(n):
#         nll = 0.0
#         for timestep in range(l):
#             curr_gt = pred_traj_gt[time_step, node]
# '''

# def dilated_nbrs(sender, receiver, num_nodes, num_scales):
#     data = np.ones(len(sender), np.bool)
#     csr = sparse.csr_matrix((data, (sender, receiver)), shape=(num_nodes, num_nodes))

#     mat = csr
#     nbrs = []
#     for i in range(1, num_scales):
#         mat = mat * mat

#         nbr = dict()
#         coo = mat.tocoo()
#         new_sender = coo.row.astype(np.int64)
#         new_receiver = coo.col.astype(np.int64)
#         nbrs.append([new_sender,new_receiver])
#     return nbrs

# def make_grids(scenario, ego_id, startframe, grid_length, max_disp_front=55, max_disp_rear=30, max_radius=55, cross_dist=6.0, num_scales=6):
#     ego_posG = scenario.obstacles[startframe][ego_id].initial_state.get_posG()
#     ego_lanelet_id = scenario.obstacles[startframe][ego_id].initial_state.posF.ind[1]
#     ego_lanelet = scenario.lanelet_network.find_lanelet_by_id(ego_lanelet_id)

#     selected_lanelets = []
#     for lanelet in scenario.lanelet_network.lanelets:
#         if lanelet.lanelet_id == ego_lanelet_id:
#             selected_lanelets.append(lanelet)
#             continue
#         if len(lanelet.successor)==0 and len(lanelet.predecessor)==0:
#             # this is case for isolated lanelets
#             continue
#         s_pos_x, s_pos_y = lanelet.center_curve[0].pos.x, lanelet.center_curve[0].pos.y
#         if (s_pos_x - ego_posG.x)**2+(s_pos_y - ego_posG.y)**2 <= max_radius**2:
#             selected_lanelets.append(lanelet)
#             continue
#         e_pos_x, e_pos_y = lanelet.center_curve[-1].pos.x, lanelet.center_curve[-1].pos.y
#         if (e_pos_x - ego_posG.x)**2+(e_pos_y - ego_posG.y)**2 <= max_radius**2:
#             selected_lanelets.append(lanelet)
#             continue
#         for curvept in lanelet.center_curve[1:-1]:
#             if (curvept.pos.x - ego_posG.x)**2+(curvept.pos.y - ego_posG.y)**2 <= max_radius**2:
#                 selected_lanelets.append(lanelet)
#                 break

#     ctrs = []
#     vecs = []
#     pris = []
#     lrdists = []

#     suc_edges, suc_edges = {}, {}
#     pre_edges, pre_edges = {}, {}
#     lane_ids = [lanelet.lanelet_id for lanelet in selected_lanelets]
#     grids = []
#     for lanelet in selected_lanelets:
#         start_curvePt = lanelet.center_curve[0]
#         start_curveInd = CurveIndex(0, 0.0)
#         nodes = []

#         lanelet_leftmost = lanelet
#         while lanelet_leftmost.adj_left is not None and lanelet_leftmost.adj_left_same_direction:
#             lanelet_leftmost = scenario.lanelet_network.find_lanelet_by_id(lanelet_leftmost.adj_left)
#         lanelet_rightmost = lanelet
#         while lanelet_rightmost.adj_right is not None and lanelet_rightmost.adj_right_same_direction:
#             lanelet_rightmost = scenario.lanelet_network.find_lanelet_by_id(lanelet_rightmost.adj_right)


#         if start_curvePt.s + grid_length*1.5 > lanelet.center_curve[-1].s:
#             # at least make two nodes from a single lanelet
#             ds = (lanelet.center_curve[-1].s - start_curvePt.s) * 0.5
#             center_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, ds*0.5)
#             center_curvePt = lanelet.get_curvePt_by_curveid(center_curveInd)
#             end_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, ds)
#             end_curvePt = lanelet.get_curvePt_by_curveid(end_curveInd)
#             nodes.append([(start_curvePt,start_curveInd),
#                       (center_curvePt,center_curveInd),
#                       (end_curvePt, end_curveInd)])

#             start_curvePt = end_curvePt
#             start_curveInd = end_curveInd
#             center_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, ds*0.5)
#             center_curvePt = lanelet.get_curvePt_by_curveid(center_curveInd)
#             end_curveInd = CurveIndex(len(lanelet.center_curve)-1, 1.0)
#             end_curvePt = lanelet.center_curve[-1]
#             nodes.append([(start_curvePt,start_curveInd),
#                       (center_curvePt,center_curveInd),
#                       (end_curvePt, end_curveInd)])

#             #continue
#         else:
#             while start_curvePt.s + grid_length <= lanelet.center_curve[-1].s:
#                 center_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, grid_length*0.5)
#                 center_curvePt = lanelet.get_curvePt_by_curveid(center_curveInd)
#                 end_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, grid_length)
#                 end_curvePt = lanelet.get_curvePt_by_curveid(end_curveInd)
#                 nodes.append([(start_curvePt,start_curveInd),
#                               (center_curvePt,center_curveInd),
#                               (end_curvePt, end_curveInd)])

#                 start_curvePt = end_curvePt
#                 start_curveInd = end_curveInd
#                 if lanelet.center_curve[-1].s - start_curvePt.s < 0.5*grid_length:
#                     start_curvePt, start_curveInd = nodes[-1][0]
#                     ds = lanelet.center_curve[-1].s - start_curvePt.s
#                     center_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, ds*0.5)
#                     center_curvePt = lanelet.get_curvePt_by_curveid(center_curveInd)
#                     end_curveInd = CurveIndex(len(lanelet.center_curve)-1, 1.0)
#                     end_curvePt = lanelet.center_curve[-1]
#                     nodes[-1] = [(start_curvePt,start_curveInd),
#                               (center_curvePt,center_curveInd),
#                               (end_curvePt, end_curveInd)]
#                     break
#                 elif lanelet.center_curve[-1].s - start_curvePt.s >= 0.5*grid_length and \
#                       lanelet.center_curve[-1].s - start_curvePt.s < grid_length:
#                     ds = lanelet.center_curve[-1].s - start_curvePt.s
#                     center_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, ds*0.5)
#                     center_curvePt = lanelet.get_curvePt_by_curveid(center_curveInd)
#                     end_curveInd = CurveIndex(len(lanelet.center_curve)-1, 1.0)
#                     end_curvePt = lanelet.center_curve[-1]
#                     nodes.append([(start_curvePt,start_curveInd),
#                               (center_curvePt,center_curveInd),
#                               (end_curvePt, end_curveInd)])
#                     break

#         for n in nodes:
#             grids.append(Grid(len(grids)))
#             grids[-1].add_pos(n[1][0].pos)
#             grids[-1].add_ind((n[1][1], lanelet.lanelet_id))

#         ctr = []
#         vec = []
#         origin_ctr = []
#         for n in nodes:
#             sta_proj = n[0][0].pos.inertial2body(ego_posG)
#             cen_proj = n[1][0].pos.inertial2body(ego_posG)
#             las_proj = n[2][0].pos.inertial2body(ego_posG)
#             ctr.append([cen_proj.x, cen_proj.y])
#             vec.append([las_proj.x-sta_proj.x, las_proj.y-sta_proj.y])
#             origin_ctr.append([n[1][0].pos.x, n[1][0].pos.y])

#         ctrs.append(np.array(ctr))

#         vecs.append(np.array(vec))

#         lrdists.append(np.concatenate([lanelet_leftmost.distance_line2line(np.array(origin_ctr), line="left"),
#                                  lanelet_rightmost.distance_line2line(np.array(origin_ctr), line="right")]
#                                  ,-1))
#         #ctrs.append(np.array([[n[1][0].pos.x, n[1][0].pos.y] for n in nodes]))
#         #vecs.append(np.array([[n[2][0].pos.x-n[0][0].pos.x, n[2][0].pos.y-n[0][0].pos.y] for n in nodes]))
#         if LaneletType.ACCESS_RAMP in lanelet.lanelet_type:
#             pris.append(np.array([[1,0] for _ in range(len(nodes))]))
#         elif LaneletType.EXIT_RAMP in lanelet.lanelet_type:
#             pris.append(np.array([[0,1] for _ in range(len(nodes))]))
#         else:
#             pris.append(np.array([[1,1] for _ in range(len(nodes))]))


#     node_idcs = []
#     count = 0
#     node2lane = {}
#     lane2node = {lanelet.lanelet_id:[] for lanelet in selected_lanelets}
#     for lanelet, ctr in zip(selected_lanelets, ctrs):
#         node_idcs.append(range(count, count + len(ctr)))
#         lane2node[lanelet.lanelet_id] = range(count, count + len(ctr))
#         for idx in node_idcs[-1]:
#             node2lane[idx] = lanelet.lanelet_id
#         count += len(ctr)
#     num_nodes = count

#     pre_sender, pre_receiver, suc_sender, suc_receiver = [], [], [], []
#     for i, lane in enumerate(selected_lanelets):
#         idcs = node_idcs[i]
#         pre_sender += idcs[1:]
#         pre_receiver += idcs[:-1]
#         if len(lane.predecessor) > 0:
#             for nbr_id in lane.predecessor:
#                 if nbr_id in lane_ids:
#                     j = lane_ids.index(nbr_id)
#                     pre_sender.append(idcs[0])
#                     pre_receiver.append(node_idcs[j][-1])


#         suc_sender += idcs[:-1]
#         suc_receiver += idcs[1:]

#         if len(lane.successor) > 0:
#             for nbr_id in lane.successor:
#                 if nbr_id in lane_ids:
#                     j = lane_ids.index(nbr_id)
#                     suc_sender.append(idcs[-1])
#                     suc_receiver.append(node_idcs[j][0])

#     suc_edges[0] = torch.tensor([suc_sender,suc_receiver], dtype=torch.long)
#     pre_edges[0] = torch.tensor([pre_sender,pre_receiver], dtype=torch.long)

#     i = 1
#     for edges in dilated_nbrs(suc_sender, suc_receiver, num_nodes, num_scales):
#         suc_edges[i] = torch.tensor(edges, dtype=torch.long)
#         i += 1
#     i = 1
#     for edges in dilated_nbrs(pre_sender, pre_receiver, num_nodes, num_scales):
#         pre_edges[i] = torch.tensor(edges, dtype=torch.long)
#         i += 1

#     ctrs = torch.tensor(np.concatenate(ctrs, 0), dtype=torch.float)
#     vecs = torch.tensor(np.concatenate(vecs, 0), dtype=torch.float)
#     pris = torch.tensor(np.concatenate(pris, 0), dtype=torch.float)

#     lrdists = torch.tensor(np.concatenate(lrdists, 0), dtype=torch.float)

#     lane_idcs = []
#     for i, idcs in enumerate(node_idcs):
#         lane_idcs.append(i * np.ones(len(idcs), np.int64))
#     lane_idcs = np.concatenate(lane_idcs, 0)

#     pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
#     for i, lane in enumerate(selected_lanelets):
#         nbr_ids = lane.predecessor
#         for nbr_id in nbr_ids:
#             if nbr_id in lane_ids:
#                 j = lane_ids.index(nbr_id)
#                 pre_pairs.append([i, j])

#         nbr_ids = lane.successor
#         for nbr_id in nbr_ids:
#             if nbr_id in lane_ids:
#                 j = lane_ids.index(nbr_id)
#                 suc_pairs.append([i, j])

#         nbr_id = lane.adj_left
#         if nbr_id is not None and lane.adj_left_same_direction:
#             if nbr_id in lane_ids:
#                 j = lane_ids.index(nbr_id)
#                 left_pairs.append([i, j])

#         nbr_id = lane.adj_right
#         if nbr_id is not None and lane.adj_right_same_direction:
#             if nbr_id in lane_ids:
#                 j = lane_ids.index(nbr_id)
#                 right_pairs.append([i, j])

#     pre_pairs = torch.tensor(pre_pairs, dtype=torch.long)
#     suc_pairs = torch.tensor(suc_pairs, dtype=torch.long)
#     left_pairs = torch.tensor(left_pairs, dtype=torch.long)
#     right_pairs = torch.tensor(right_pairs, dtype=torch.long)

#     num_lanes = len(selected_lanelets)
#     dist = ctrs.unsqueeze(1) - ctrs.unsqueeze(0)
#     dist = torch.sqrt((dist ** 2).sum(2))
#     hi = torch.arange(num_nodes).long().to(dist.device).view(-1, 1).repeat(1, num_nodes).view(-1)
#     wi = torch.arange(num_nodes).long().to(dist.device).view(1, -1).repeat(num_nodes, 1).view(-1)
#     row_idcs = torch.arange(num_nodes).long().to(dist.device)

#     pre = pre_pairs.new().float().resize_(num_lanes, num_lanes).zero_()
#     pre[pre_pairs[:, 0], pre_pairs[:, 1]] = 1
#     suc = suc_pairs.new().float().resize_(num_lanes, num_lanes).zero_()
#     suc[suc_pairs[:, 0], suc_pairs[:, 1]] = 1

#     pairs = left_pairs
#     if len(pairs) > 0:
#         mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
#         mat[pairs[:, 0], pairs[:, 1]] = 1
#         mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

#         left_dist = dist.clone()
#         mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
#         left_dist[hi[mask], wi[mask]] = 1e6

#         min_dist, min_idcs = left_dist.min(1)
#         mask = min_dist < cross_dist
#         ui = row_idcs[mask]
#         vi = min_idcs[mask]
#         f1 = vecs[ui]
#         f2 = vecs[vi]
#         t1 = torch.atan2(f1[:, 1], f1[:, 0])
#         t2 = torch.atan2(f2[:, 1], f2[:, 0])
#         dt = torch.abs(t1 - t2)
#         m = dt > np.pi
#         dt[m] = torch.abs(dt[m] - 2 * np.pi)
#         m = dt < 0.25 * np.pi

#         ui = ui[m]
#         vi = vi[m]

#         left_edges = torch.stack([ui, vi])
#     else:
#         left_edges = torch.tensor([np.zeros(0, np.int16), np.zeros(0, np.int16)], dtype=torch.long)

#     pairs = right_pairs
#     if len(pairs) > 0:
#         mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
#         mat[pairs[:, 0], pairs[:, 1]] = 1
#         mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

#         right_dist = dist.clone()
#         mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
#         right_dist[hi[mask], wi[mask]] = 1e6

#         min_dist, min_idcs = right_dist.min(1)
#         mask = min_dist < cross_dist
#         ui = row_idcs[mask]
#         vi = min_idcs[mask]
#         f1 = vecs[ui]
#         f2 = vecs[vi]
#         t1 = torch.atan2(f1[:, 1], f1[:, 0])
#         t2 = torch.atan2(f2[:, 1], f2[:, 0])
#         dt = torch.abs(t1 - t2)
#         m = dt > np.pi
#         dt[m] = torch.abs(dt[m] - 2 * np.pi)
#         m = dt < 0.25 * np.pi

#         ui = ui[m]
#         vi = vi[m]

#         right_edges = torch.stack([ui, vi])
#     else:
#         right_edges = torch.tensor([np.zeros(0, np.int16), np.zeros(0, np.int16)], dtype=torch.long)

#     return grids, ctrs,  vecs, pris, lrdists, suc_edges, pre_edges, left_edges, right_edges, node2lane, lane2node

# def generate_frenet_path(curve, s0, vs0, d0, vd0, sT, vsT, dT, vdT, T, DT):
#     path = calc_frenet_path(s0, vs0, d0, vd0, sT, vsT, dT, vdT, T, DT)
#     #path = calc_global_path(path, curve)
#     return path

# def calc_frenet_path(s0, vs0, d0, vd0, sT, vsT, dT, vdT, T, DT):
#     fp = FrenetPath()
#     lat = QuinticPolynomial(d0, vd0, 0.0, dT, vdT, 0.0, T)
#     lon = QuinticPolynomial(s0, vs0, 0.0, sT, vsT, 0.0, T)

#     fp.t = [t for t in np.arange(0.0, T, DT)]
#     fp.d = [lat.calc_point(t) for t in fp.t]
#     fp.s = [lon.calc_point(t) for t in fp.t]

# class FrenetPath:
#     def __init__(self):
#         self.t = []
#         self.d = []
#         self.s = []


# class QuinticPolynomial:
#     def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
#         # calc coefficient of quintic polynomial
#         self.a0 = xs
#         self.a1 = vxs
#         self.a2 = axs / 2.0

#         A = np.array([[time ** 3, time ** 4, time ** 5],
#                       [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
#                       [6 * time, 12 * time ** 2, 20 * time ** 3]])
#         b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
#                       vxe - self.a1 - 2 * self.a2 * time,
#                       axe - 2 * self.a2])
#         x = np.linalg.solve(A, b)

#         self.a3 = x[0]
#         self.a4 = x[1]
#         self.a5 = x[2]

#     def calc_point(self, t):
#         xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
#              self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

#         return xt

#     def calc_first_derivative(self, t):
#         xt = self.a1 + 2 * self.a2 * t + \
#              3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

#         return xt

#     def calc_second_derivative(self, t):
#         xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

#         return xt

#     def calc_third_derivative(self, t):
#         xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

#         return xt
