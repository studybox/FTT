import os
import os.path as osp
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle, Arc, Polygon
from copy import copy
import torch
import gc
from torch_geometric.data import Dataset, Data, Batch
from torch.utils.data import DataLoader
from torch_geometric.utils import degree, add_self_loops
from commonroad.scenario.scenariocomplement import MultiFeatureExtractor, CoreFeatureExtractor0, NeighborFeatureExtractor0, NeighborFeatureExtractor, ScenarioWrapper
#from commonroad.scenario.scenariocomplement import LaneletToLaneletFeatureExtractor0, LaneletFeatureExtractor0, VehicleToLaneletFeatureExtractor0
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType, DynamicObstacle
from commonroad.common.file_reader_complement import LaneletCurveNetworkReader
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LineMarking, LaneletType
from commonroad.scenario.trajectorycomplement import FrenetState, Frenet, move_along_curve

from commonroad.scenario.laneletcomplement import *
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad.scenario.laneletcomplement import make_lanelet_curve_network
from commonroad.common.file_reader import CommonRoadFileReader


from datasets.utils import make_grids, local_search
import networkx as nx
from config import basic_shape_parameters_ego, basic_shape_parameters_nei, basic_shape_parameters_obs, draw_params_neighbor, draw_params_ego, draw_params_obstacle
# data in the format of list [[[x_n,x_e, edge_index, y_n], next_step, next_step], next_scene]

class ScenarioData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        
class ScenarioDataset(Dataset):
    def __init__(self, config, processed_files=None, transform=None, pre_transform=None, process_route=False):
        self.config = config
        if isinstance(config["data_dir"], str):
            self.root = root = config["data_dir"]
            self.raw_files = [f for f in os.listdir(osp.join(root, "raw")) if 'dump' in f]
            self.map_files = [f for f in os.listdir(osp.join(root, "raw")) if 'lanelet' in f]
        else:
            self.root = root = config["data_dir"][0]
            self.raw_files = [f for f in os.listdir(osp.join(root, "raw")) if 'dump' in f]
            self.map_files = [f for f in os.listdir(osp.join(root, "raw")) if 'lanelet' in f]
        if processed_files is None:
            self.vehicle_exts = MultiFeatureExtractor([CoreFeatureExtractor0(self.config),
                                              NeighborFeatureExtractor0(self.config),
                                              NeighborFeatureExtractor()])

            self.processed_files = [osp.join(root, "processed", f) for f in os.listdir(osp.join(root, "processed")) if 'data' in f]
        else:
            self.processed_files = processed_files
        
        super(ScenarioDataset, self).__init__(root, transform, pre_transform)
        if process_route:
            self.load_route = True
            self.add_routes(self.processed_file_names)
        
    @property
    def raw_file_names(self):
        return self.raw_files
    @property
    def processed_paths(self):
        return self.processed_files
    @property
    def processed_file_names(self):
        return self.processed_files
    @property
    def lanelet_file_names(self):
        return self.map_files
    @property
    def lanelet_paths(self):
        files = self.lanelet_file_names
        return [osp.join(self.raw_dir, f) for f in files]
    def generate(self, expert_cr_scenario, egoid, time_steps):
        record = True
        all_grids,\
        ctrs, \
        vecs, \
        pris, \
        widths, \
        suc_edges, \
        pre_edges, left_edges, right_edges,\
        node2lane, lane2node  = make_grids(expert_cr_scenario,
                                                            egoid,
                                                            time_steps[-1],
                                                            self.config["grid_length"],
                                                            max_disp_front=self.config["max_grid_disp_front"],
                                                            max_disp_rear=self.config["max_grid_disp_rear"],
                                                            max_radius=self.config["max_grid_radius"],
                                                            cross_dist=self.config["cross_dist"],
                                                            num_scales=self.config["num_scales"])
        #print(" |||after grid")
        expert_cr_scenario.add_grids(all_grids)
        neighbors = expert_cr_scenario.get_neighbors(egoid,
                                                     time_steps[-1],
                                                     along_lanelet=self.config["veh_along_lane"],
                                                     max_radius=self.config["max_veh_radius"],
                                                     front=self.config["max_veh_disp_front"],
                                                     side=self.config["max_veh_disp_side"],
                                                     rear=self.config["max_veh_disp_rear"])
        self.vehicle_exts.set_origin(expert_cr_scenario, egoid, time_steps[-1])
        x_dict,y_dict,lx_dict,ly_dict,w_dict,edge_dict,edge_dict2 = {},{},{},{},{},{},{}
        ego_feat = self.vehicle_exts.get_features(expert_cr_scenario, egoid, time_steps[-1])
        ego_x = self.vehicle_exts.get_features_by_name(ego_feat, ['obs_xs', 'obs_ys',
                                                                  'obs_rel_xs','obs_rel_ys', 'masked',
                                                                  #'lane_lefts', 'lane_rights',
                                                                  'length', 'width'])
        ego_lx = self.vehicle_exts.get_features_by_name(ego_feat, ['obs_lanelet'])
        ego_y = self.vehicle_exts.get_features_by_name(ego_feat, ['fut_xs', 'fut_ys', 'fut_rel_xs', 'fut_rel_ys', 'has_preds'])
        ego_ly = self.vehicle_exts.get_features_by_name(ego_feat, ['fut_lanelet'])
        #print(" |||after ego")

        if ego_y[0] == -100 or ego_x[0] == -100:
            record = False
        max_vel_nei_x = np.max(self.vehicle_exts.get_features_by_name(ego_feat,['obs_rel_xs']))/(self.config["delta_step"]*self.config["dt"])
        max_vel_nei_y = np.max(self.vehicle_exts.get_features_by_name(ego_feat,['obs_rel_ys']))/(self.config["delta_step"]*self.config["dt"])
        if max_vel_nei_x >= 40 or max_vel_nei_y >= 40:
            record = False
        x_dict[egoid] = ego_x
        y_dict[egoid] = ego_y
        lx_dict[egoid] = ego_lx
        ly_dict[egoid] = ego_ly
        w_dict[egoid] = 1.0

        for name in self.vehicle_exts.extractors[1].neighbor_names:
            veh_id = int(self.vehicle_exts.get_features_by_name(ego_feat, [name+'_veh_id'])[0])
            if veh_id in neighbors:
                edge_dict[(egoid, name, 1)] = self.vehicle_exts.get_features_by_name(ego_feat, [name+'_veh_id', name+'_dist_ins',name+'_offset_ins',
                                                                                                                     name+'_vel_x', name+"_vel_y",
                                                                                                                     name+"_length", name+"_width"])
            else:
                edge_dict[(egoid, name, 0)] = [0.0, 100., 100., 0.0, 0.0, 0.0, 0.0]
        close_neighbors = []
        for name in self.vehicle_exts.extractors[2].neighbor_names:
            veh_id = int(self.vehicle_exts.get_features_by_name(ego_feat, [name+'_veh_id'])[0])
            if veh_id in neighbors:
                edge_dict2[(egoid,veh_id)] = self.vehicle_exts.get_features_by_name(ego_feat, [name+'_rel_x',name+'_rel_y'])
                w_dict[int(veh_id)] = 0.8
                close_neighbors.append(veh_id)
        remaining_neighbors = []
        connected_neighbors = []
        for neighbor_id in close_neighbors:
            neighbors_feat = self.vehicle_exts.get_features(expert_cr_scenario, neighbor_id, time_steps[-1])
            nei_x = self.vehicle_exts.get_features_by_name(neighbors_feat,['obs_xs','obs_ys',
                                                                           'obs_rel_xs','obs_rel_ys', 'masked',
                                                                           #'lane_lefts', 'lane_rights',
                                                                           'length', 'width'])
            nei_lx = self.vehicle_exts.get_features_by_name(neighbors_feat, ['obs_lanelet'])

            nei_y = self.vehicle_exts.get_features_by_name(neighbors_feat, ['fut_xs','fut_ys', 'fut_rel_xs', 'fut_rel_ys', 'has_preds'])
            nei_ly = self.vehicle_exts.get_features_by_name(neighbors_feat, ['fut_lanelet'])

            if nei_y[0] == -100 or nei_x[0] == -100:
                record = False
            max_vel_nei_x = np.max(self.vehicle_exts.get_features_by_name(neighbors_feat,['obs_rel_xs']))/(self.config["delta_step"]*self.config["dt"])
            max_vel_nei_y = np.max(self.vehicle_exts.get_features_by_name(neighbors_feat,['obs_rel_ys']))/(self.config["delta_step"]*self.config["dt"])
            if max_vel_nei_x >= 40 or max_vel_nei_y >= 40:
                record = False
            x_dict[neighbor_id] = nei_x
            y_dict[neighbor_id] = nei_y

            lx_dict[neighbor_id] = nei_lx
            ly_dict[neighbor_id] = nei_ly


            for name in self.vehicle_exts.extractors[1].neighbor_names:
                veh_id = int(self.vehicle_exts.get_features_by_name(neighbors_feat, [name+'_veh_id'])[0])
                if veh_id in neighbors:
                    edge_dict[(neighbor_id, name, 1)] = self.vehicle_exts.get_features_by_name(neighbors_feat, [name+'_veh_id', name+'_dist_ins',name+'_offset_ins',
                                                                                                                                     name+'_vel_x', name+'_vel_y',
                                                                                                                                     name+"_length", name+"_width"])
                else:
                    edge_dict[(neighbor_id, name, 0)] = [0.0, 100., 100., 0.0, 0.0, 0.0, 0.0]
            for name in self.vehicle_exts.extractors[2].neighbor_names:
                veh_id = int(self.vehicle_exts.get_features_by_name(neighbors_feat, [name+'_veh_id'])[0])
                if veh_id in neighbors:
                    edge_dict2[(neighbor_id, veh_id)] = self.vehicle_exts.get_features_by_name(neighbors_feat, [name+'_rel_x',name+'_rel_y'])
                    if veh_id not in close_neighbors and veh_id != egoid and veh_id not in remaining_neighbors:
                        remaining_neighbors.append(veh_id)
                        connected_neighbors.append(veh_id)
        for neighbor_id in neighbors:
            if neighbor_id not in close_neighbors and neighbor_id not in remaining_neighbors:
                remaining_neighbors.append(neighbor_id)
        for neighbor_id in remaining_neighbors:
            neighbors_feat = self.vehicle_exts.get_features(expert_cr_scenario, neighbor_id, time_steps[-1])
            for name in self.vehicle_exts.extractors[1].neighbor_names:
                veh_id = int(self.vehicle_exts.get_features_by_name(neighbors_feat, [name+'_veh_id'])[0])
                if veh_id == egoid or veh_id in close_neighbors or veh_id in remaining_neighbors:
                    edge_dict[neighbor_id, name, 1] = self.vehicle_exts.get_features_by_name(neighbors_feat, [name+'_veh_id', name+'_dist_ins',name+'_offset_ins',
                                                                                                                                   name+'_vel_x', name+'_vel_y',
                                                                                                                                   name+"_length", name+"_width"])
                else:
                    edge_dict[(neighbor_id, name, 0)] = [0.0, 100., 100., 0.0, 0.0, 0.0, 0.0]
            for name in self.vehicle_exts.extractors[2].neighbor_names:
                veh_id = int(self.vehicle_exts.get_features_by_name(neighbors_feat, [name+'_veh_id'])[0])
                if veh_id == egoid or veh_id in close_neighbors or veh_id in remaining_neighbors:
                    edge_dict2[(neighbor_id, veh_id)] = self.vehicle_exts.get_features_by_name(neighbors_feat, [name+'_rel_x',name+'_rel_y'])
                    if veh_id in remaining_neighbors and veh_id not in connected_neighbors:
                        connected_neighbors.append(veh_id)
                    if neighbor_id not in connected_neighbors:
                        connected_neighbors.append(neighbor_id)
        for neighbor_id in connected_neighbors:
            neighbors_feat = self.vehicle_exts.get_features(expert_cr_scenario, neighbor_id, time_steps[-1])
            nei_x = self.vehicle_exts.get_features_by_name(neighbors_feat, ['obs_xs','obs_ys',
                                                                            'obs_rel_xs','obs_rel_ys','masked',
                                                                            #'lane_lefts', 'lane_rights',
                                                                            'length', 'width'])
            nei_lx = self.vehicle_exts.get_features_by_name(neighbors_feat, ['obs_lanelet'])

            nei_y = self.vehicle_exts.get_features_by_name(neighbors_feat, ['fut_xs','fut_ys', 'fut_rel_xs', 'fut_rel_ys', 'has_preds'])
            nei_ly = self.vehicle_exts.get_features_by_name(neighbors_feat, ['fut_lanelet'])

            if nei_y[0] == -100 or nei_x[0] == -100 :
                record = False
            max_vel_nei_x = np.max(self.vehicle_exts.get_features_by_name(neighbors_feat,['obs_rel_xs']))/(self.config["delta_step"]*self.config["dt"])
            max_vel_nei_y = np.max(self.vehicle_exts.get_features_by_name(neighbors_feat,['obs_rel_ys']))/(self.config["delta_step"]*self.config["dt"])
            if max_vel_nei_x >= 40 or max_vel_nei_y >= 40:
                record = False
            x_dict[neighbor_id] = nei_x
            y_dict[neighbor_id] = nei_y

            lx_dict[neighbor_id] = nei_lx
            ly_dict[neighbor_id] = nei_ly


            w_dict[neighbor_id] = 0.5
        if record == False or len(x_dict)==1:
            return None
        #print(" |||after nei")

        # convert to linear data
        x, xseq, y, yseq, w, id, shape, has_preds = [], [], [], [], [], [], [], []
        lxseq, lyseq = [], []
        id_to_idx = {}
        sender, receiver, edge_attr = [], [], []
        pred_steps =  self.config["prediction_steps"]//self.config["delta_step"]
        for idx, vehid in enumerate(x_dict.keys()):
            xx = np.array(copy(x_dict[vehid]))
            shape.append(xx[-2:])
            #xx = np.concatenate([xx[:-2], np.repeat(xx[-2:], len(time_steps))])
            yy = np.array(copy(y_dict[vehid])[:-pred_steps]).reshape((4, pred_steps)).T
            xseq.append(xx[:-2].reshape((5, len(time_steps))).T) # change

            lxseq.append(np.array(lx_dict[vehid]))
            x.append(copy(x_dict[vehid]))
            yseq.append(yy)

            lyseq.append(np.array(ly_dict[vehid]))

            has_preds.append(copy(y_dict[vehid])[-pred_steps:])
            y.append(copy(y_dict[vehid]))
            w.append(w_dict[vehid])
            id.append(vehid)
            id_to_idx[vehid] = idx
            for name in self.vehicle_exts.extractors[1].neighbor_names:
                if (vehid, name, 1) in edge_dict:
                    x[-1].append(1)
                    x[-1].extend(edge_dict[(vehid, name, 1)][1:])
                else:
                    x[-1].append(0)
                    x[-1].extend(edge_dict[(vehid, name, 0)][1:])
            edge_attr.append([0.0, 0.0])
            sender.append(idx)
            receiver.append(idx)

        for s, r in edge_dict2.keys():
            edge_attr.append(edge_dict2[(s,r)])
            sender.append(id_to_idx[s])
            receiver.append(id_to_idx[r])

        has_preds = torch.tensor(has_preds, dtype=torch.bool)
        num_preds = has_preds.size(1)
        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
                  has_preds.device
               ) / float(num_preds)
        max_last, last_idcs = last.max(1)
        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)

        veh_xseq = torch.tensor(xseq, dtype=torch.float)
        veh_yseq = torch.tensor(yseq, dtype=torch.float)

        veh_lxseq = torch.tensor(lxseq, dtype=torch.long)
        veh_lyseq = torch.tensor(lyseq, dtype=torch.long)

        veh_start_ctrs = veh_xseq[:,-1,:2]
        veh_start_lanelet = veh_lxseq[:,-1].numpy()
        veh_end_ctrs = veh_yseq[:,:,:2]
        veh_end_ctrs = veh_end_ctrs[row_idcs, last_idcs]
        veh_end_lanelet = veh_lyseq[row_idcs, last_idcs].numpy()

        lane_ctrs = ctrs

        path_DG = nx.DiGraph()
        lanelet_DG = nx.DiGraph()
        for lanelet_id in lane2node.keys():
            lanelet = expert_cr_scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
            if LaneletType.ACCESS_RAMP in lanelet.lanelet_type or \
               LaneletType.EXIT_RAMP in lanelet.lanelet_type:
                pri = 1
            else:
                pri = 2

            is_left = lanelet.adj_left is None
            path_DG.add_node(lanelet_id, pri=pri, left=is_left)
            for succ in lanelet.successor:
                if succ in lane2node:
                    path_DG.add_edge(lanelet_id, succ)
                    lanelet_DG.add_edge(lanelet_id, succ)
            if lanelet.adj_left is not None:
                lanelet_DG.add_edge(lanelet_id, lanelet.adj_left)
            if lanelet.adj_right is not None:
                lanelet_DG.add_edge(lanelet_id, lanelet.adj_right)


        dist = veh_start_ctrs.view(-1,1,2) - lane_ctrs.view(1,-1,2)
        start_dist = torch.sqrt((dist**2).sum(2))
        start_mask = start_dist <= self.config["actor2map_dist"]
        start_dist_idcs = torch.nonzero(start_mask, as_tuple=False)

        sorted_start_dist, sorted_start_idcs = start_dist.sort(1, descending=False)

        dist = veh_end_ctrs.view(-1,1,2) - lane_ctrs.view(1,-1,2)
        target_dist = torch.sqrt((dist**2).sum(2))
        target_mask = target_dist <=self.config["map2actor_dist"]
        target_dist_idcs = torch.nonzero(target_mask, as_tuple=False)

        sorted_target_dist, sorted_target_idcs = target_dist.sort(1, descending=False)

        # find the shortest paths

        node_DG = nx.DiGraph()
        for n in range(lane_ctrs.size(0)):
            node_DG.add_node(n)

        flr_edge_index = torch.cat([suc_edges[0],
                                    left_edges,
                                    right_edges], -1)
        for j in range(flr_edge_index.size(-1)):
            length = torch.sqrt(((ctrs[flr_edge_index[0][j]]-ctrs[flr_edge_index[1][j]])**2).sum()).item()
            node_DG.add_edge(flr_edge_index[0][j].item(), flr_edge_index[1][j].item(),
                             length=length)
        obs_veh_index, fut_veh_index = [], []
        obs_lane_index, fut_lane_index = [], []
        route_index = {r:[] for r in range(5)}
        actor_routes = []
        actor_targets = []
        veh_full_path = []

        veh_path_sender, veh_path_receiver = [], []
        path_node_sender, path_node_receiver = [], []
        path_num_nodes, path_node_node = [], []

        delta_s, next_d = [], []
        for v in range(veh_start_ctrs.size(0)):
            start_vs = start_dist_idcs[:,0] == v
            if start_vs.sum() == 0:
                first_ns = list(sorted_start_idcs[v, :4].numpy())
            else:
                first_ns = list(start_dist_idcs[:, 1][start_vs].numpy())
            #target_vs = target_dist_idcs[:,0] == v
            #if target_vs.sum() == 0:
            #    last_ns = sorted_target_idcs[v, :4].numpy()
            #else:
            #    last_ns = target_dist_idcs[:, 1][target_vs].numpy()

            last_ns = sorted_target_idcs[v, :8].numpy()
            last_dists = sorted_target_dist[v, :8].numpy()

            # lanelet level
            #start_position = veh_start_ctrs[v].numpy()
            #start_lanelet_cands = expert_cr_scenario.lanelet_network.find_lanelet_by_position([start_position])[0]

            #dest_position = veh_end_ctrs[v].numpy()
            #dest_lanelet_cands = expert_cr_scenario.lanelet_network.find_lanelet_by_position([dest_position])[0]

            # it is possible that the vehicle is outside any lanelets
            #if len(start_lanelet_cands) == 0:
            start_lanelet_cands = [node2lane[n] for n in first_ns]
            start_lanelet_cands = set(start_lanelet_cands)
            #if len(dest_lanelet_cands) == 0:
            dest_lanelet_cands = [node2lane[n] for n in last_ns]
            dest_lanelet_cands = set(dest_lanelet_cands)
            # need to find the ground-truth start_lanelet, and dest_lanelet
            best_start_lanelet_id = veh_start_lanelet[v]
            best_dest_lanelet_id = veh_end_lanelet[v]

            # TODO hard coding for SR
            '''
            if best_dest_lanelet_id == 138:
                best_dest_lanelet_id = 136
            if best_dest_lanelet_id == 115 or best_dest_lanelet_id == 118:
                best_dest_lanelet_id = 117
            #
            '''
            assert(best_start_lanelet_id is not None)
            assert(best_dest_lanelet_id is not None)

            #print("start_cand ", start_lanelet_cands, " dest_cand ", dest_lanelet_cands)
            #print("||best start", best_start_lanelet_id, "||best_dest ", best_dest_lanelet_id)
            #print("best start", best_start_lanelet_id, "best_dest ", best_dest_lanelet_id)

            if best_start_lanelet_id not in start_lanelet_cands or best_dest_lanelet_id not in dest_lanelet_cands:
                record = False
                print("start_cand ", start_lanelet_cands, " dest_cand ", dest_lanelet_cands)
                print("||best start", best_start_lanelet_id, "||best_dest ", best_dest_lanelet_id)
                all_lanelet_paths = nx.multi_source_dijkstra_path(lanelet_DG, set(start_lanelet_cands))

                best_path_length = float('inf')
                for dest_lanelet_id in dest_lanelet_cands:
                    if dest_lanelet_id not in all_lanelet_paths:
                        continue
                    path_length = len(all_lanelet_paths[dest_lanelet_id])
                    if path_length < best_path_length:
                        best_path_length = path_length
                        best_start_lanelet_id = all_lanelet_paths[dest_lanelet_id][0]
                        best_dest_lanelet_id = dest_lanelet_id
                print("best start", best_start_lanelet_id, "best_dest ", best_dest_lanelet_id)

            #
            #    for n in lane2node[best_start_lanelet_id]:
            #        first_ns.append(n)
            # find the end lanelets each lanelet forms a path

            # start edges
            obs_veh_index.extend([v for _ in range(len(first_ns))])
            obs_lane_index.extend(first_ns)

            #last_ns = sorted_target_idcs[v, :8].numpy()
            #last_dists = sorted_target_dist[v, :8].numpy()

            # compute all route lengths
            path_lengths = nx.multi_source_dijkstra_path_length(node_DG, set(first_ns), weight="length")
            for n, d in zip(last_ns, last_dists):
                if n in path_lengths and path_lengths[n] < 40:
                   actor_targets.append(n)
                   break
            else:
                print(v, last_ns, last_dists)
                raise
            # path edges
            r = {kk:[] for kk in range(5)}
            #
            possible_lanelets = [best_start_lanelet_id, best_dest_lanelet_id]
            for n, path_len in path_lengths.items():
                lid = node2lane[n]
                if path_len < 10:#/self.config["grid_length"]:
                    r[0].append([v, n])
                    if n == actor_targets[-1]:
                        actor_routes.append(0)
                    if lid not in possible_lanelets:
                        possible_lanelets.append(lid)
                elif path_len < 20:#/self.config["grid_length"]:
                    r[1].append([v, n])
                    if n == actor_targets[-1]:
                        actor_routes.append(1)
                    if lid not in possible_lanelets:
                        possible_lanelets.append(lid)
                elif path_len < 30:#/self.config["grid_length"]:
                    r[2].append([v, n])
                    if n == actor_targets[-1]:
                        actor_routes.append(2)
                    if lid not in possible_lanelets:
                        possible_lanelets.append(lid)
                elif path_len < 40:#/self.config["grid_length"]:
                    r[3].append([v, n])
                    if n == actor_targets[-1]:
                        actor_routes.append(3)
                    if lid not in possible_lanelets:
                        possible_lanelets.append(lid)
                elif path_len < 50:#/self.config["grid_length"]:
                    r[4].append([v, n])
                    if n == actor_targets[-1]:
                        actor_routes.append(4)
                    #if lid not in possible_lanelets:
                    #    possible_lanelets.append(lid)
                    #if lid not in possible_lanelets:
                    #    possible_lanelets.append(lid)

            assert(len(r[0]) > 0) # the closest
            route_index[0].extend(r[0])
            if len(r[1]) == 0:
                assert(len(r[2]) ==0 and len(r[3]) ==0 and len(r[4]) ==0)
                route_index[1].extend(r[0][-2:])
                route_index[2].extend(r[0][-2:])
                route_index[3].extend(r[0][-2:])
                route_index[4].extend(r[0][-2:])
            elif len(r[2]) == 0:
                route_index[1].extend(r[1])
                assert(len(r[3]) ==0 and len(r[4]) ==0)
                route_index[2].extend(r[1][-2:])
                route_index[3].extend(r[1][-2:])
                route_index[4].extend(r[1][-2:])
            elif len(r[3]) == 0:
                route_index[1].extend(r[1])
                route_index[2].extend(r[2])
                assert(len(r[4]) ==0)
                route_index[3].extend(r[2][-2:])
                route_index[4].extend(r[2][-2:])
            elif len(r[4]) == 0:
                route_index[1].extend(r[1])
                route_index[2].extend(r[2])
                route_index[3].extend(r[3])
                route_index[4].extend(r[3][-2:])
            else:
                route_index[1].extend(r[1])
                route_index[2].extend(r[2])
                route_index[3].extend(r[3])
                route_index[4].extend(r[4])

            #possible starts and possible targets
            possible_sources = []
            possible_targets = []
            for lid in possible_lanelets:
                lanelet = expert_cr_scenario.lanelet_network.find_lanelet_by_id(lid)
                is_source = True
                for pred in lanelet.predecessor:
                    if pred in possible_lanelets:
                        is_source = False
                        break
                is_target = True
                for succ in lanelet.successor:
                    if succ in possible_lanelets:
                        is_target = False
                        break
                if is_source:
                    possible_sources.append(lid)
                if is_target:
                    possible_targets.append(lid)
            possible_sources = [best_start_lanelet_id]
            lanelet_leftmost = expert_cr_scenario.lanelet_network.find_lanelet_by_id(best_start_lanelet_id)
            while lanelet_leftmost.adj_left is not None and lanelet_leftmost.adj_left_same_direction:
                lanelet_leftmost = expert_cr_scenario.lanelet_network.find_lanelet_by_id(lanelet_leftmost.adj_left)
                if lanelet_leftmost.lanelet_id not in possible_sources:
                    possible_sources.append(lanelet_leftmost.lanelet_id)
            lanelet_rightmost = expert_cr_scenario.lanelet_network.find_lanelet_by_id(best_start_lanelet_id)
            while lanelet_rightmost.adj_right is not None and lanelet_rightmost.adj_right_same_direction:
                lanelet_rightmost = expert_cr_scenario.lanelet_network.find_lanelet_by_id(lanelet_rightmost.adj_right)
                if lanelet_rightmost.lanelet_id not in possible_sources:
                    possible_sources.append(lanelet_rightmost.lanelet_id)

            # let's get all possible paths
            #shortest_lanelet_paths = nx.multi_source_dijkstra_path(path_DG, set(possible_sources))

            possible_paths = []
            for sou in possible_sources:
                shortest_lanelet_paths = nx.single_source_dijkstra_path(path_DG, sou)
                for tar in possible_targets:
                    if tar in shortest_lanelet_paths:
                        is_path = True
                        for llid in shortest_lanelet_paths[tar]:
                            if llid not in possible_lanelets:
                                is_path = False
                        if is_path:
                            possible_paths.append(shortest_lanelet_paths[tar])
            #print("pos sta", possible_sources, "pos tat", possible_targets,  "pos lan", possible_lanelets, "pos path", possible_paths)
            has_target = False
            target_path_inds = []
            for pind, path in enumerate(possible_paths):
                # a candidate path
                if best_dest_lanelet_id in path:
                    #veh_full_path.append([1])
                    target_path_inds.append(pind)
                    has_target = True
                #else:
                    #veh_full_path.append([0])
                src, tar = path[0], path[-1]
                veh_path_sender.append(v)
                veh_path_receiver.append(len(path_num_nodes))
                num_path_nodes = 0
                for llid in path:
                    for nid in lane2node[llid]:
                        path_node_sender.append(len(path_num_nodes))
                        path_node_receiver.append(nid)
                        num_path_nodes += 1
                path_node_node_sender, path_node_node_receiver = [], []
                for pn in range(num_path_nodes):
                    # a path is forward connected
                    for ppn in range(pn, num_path_nodes):
                        path_node_node_sender.append(pn)
                        path_node_node_receiver.append(ppn)
                path_num_nodes.append(num_path_nodes)
                path_node_node.append(torch.tensor([path_node_node_sender,
                                                    path_node_node_receiver], dtype=torch.long))

            assert(has_target)
            if len(target_path_inds) == 1:
                target_path_ind = target_path_inds[0]
            else:
                select_main_inds = []
                for pind in target_path_inds:
                    path = possible_paths[pind]
                    if path_DG.nodes[path[-1]]["pri"] == 2:
                        select_main_inds.append(pind)

                if len(select_main_inds) == 1:
                    target_path_ind = select_main_inds[0]
                elif len(select_main_inds) == 0:
                    select_left_inds = []
                    for pind in target_path_inds:
                        path = possible_paths[pind]
                        if path_DG.nodes[path[-1]]["left"]:
                            select_left_inds.append(pind)
                    if len(select_left_inds) == 0:
                        target_path_ind = target_path_inds[0]
                    else:
                        target_path_ind = select_left_inds[0]
                else:
                    select_left_inds = []
                    for pind in select_main_inds:
                        path = possible_paths[pind]
                        if path_DG.nodes[path[-1]]["left"]:
                            select_left_inds.append(pind)
                    if len(select_left_inds) == 0:
                        target_path_ind = target_path_inds[0]
                    else:
                        target_path_ind = select_left_inds[0]
            for pind in range(len(possible_paths)):
                if pind == target_path_ind:
                    veh_full_path.append([1])
                else:
                    veh_full_path.append([0])

            target_path = possible_paths[target_path_ind]

            vertices = []
            for i, lid in enumerate(target_path):
                #lanelet = expert_cr_scenario.lanelet_network.find_lanelet_by_id(lid)
                for n in lane2node[lid]:
                    vertices.append(ctrs[n].numpy())
            vertices = np.array(vertices)

            target_curve = make_curve(vertices)

            ds, nd = [], []
            for i in range(veh_yseq[v,:,:2].size(0)):
                if i == 0:
                    start_pos = veh_start_ctrs[v].numpy()
                    tar_pos = veh_yseq[v, i, :2].numpy()
                else:
                    start_pos = veh_yseq[v, i-1, :2].numpy()
                    tar_pos = veh_yseq[v, i, :2].numpy()
                if has_preds[v, i]:
                    start_pos = VecSE2(start_pos[0], start_pos[1], 0.0)
                    tar_pos = VecSE2(tar_pos[0], tar_pos[1], 0.0)
                    start_proj = start_pos.proj_on_curve(target_curve, clamped=False)
                    tar_proj = tar_pos.proj_on_curve(target_curve, clamped=False)
                    ds.append(lerp_curve_with_ind(target_curve, tar_proj.ind) - \
                              lerp_curve_with_ind(target_curve, start_proj.ind))
                    nd.append(tar_proj.d)
                else:
                    ds.append(0.0)
                    nd.append(0.0)
            delta_s.append(ds)
            next_d.append(nd)
        last_ds = [d[-1] for d in next_d]
        #if np.max(np.abs(last_ds)) > 5:
        if np.max(np.abs(next_d)) > 6:
            print("next_d", next_d)
            record = False

        if record == False:
            return None

        lane_start = torch.tensor([obs_veh_index,obs_lane_index], dtype=torch.long)
        lane_path = {k:torch.tensor(v, dtype=torch.long).transpose(1, 0) for k,v in route_index.items()}
        veh_target = torch.tensor(actor_targets, dtype=torch.long).unsqueeze(1)
        veh_path = torch.tensor(actor_routes, dtype=torch.long).unsqueeze(1)
        veh_full_path = torch.tensor(veh_full_path, dtype=torch.float)
        path_num_nodes = torch.tensor(path_num_nodes, dtype=torch.long)
        veh_path_edge = torch.tensor([veh_path_sender, veh_path_receiver], dtype=torch.long)
        path_node_edge = torch.tensor([path_node_sender, path_node_receiver], dtype=torch.long)


        delta_s = torch.tensor(delta_s, dtype=torch.float)
        next_d = torch.tensor(next_d, dtype=torch.float)

        veh_yfrenet = torch.stack([delta_s, next_d], -1)

        lane_ids = torch.tensor([node2lane[n] for n in range(ctrs.size(0))], dtype=torch.long)
        data = ScenarioData(veh_t=torch.tensor(time_steps, dtype=torch.long),
                                                veh_x=torch.tensor(x, dtype=torch.float),
                                                veh_xseq=veh_xseq,
                                                veh_shape=torch.tensor(shape, dtype=torch.float),
                                                veh_yseq=veh_yseq,
                                                veh_yfre = veh_yfrenet,
                                                veh_path=veh_path,
                                                veh_full_path=veh_full_path,
                                                veh_target=veh_target,
                                                veh_has_preds=has_preds,
                                                veh_edge_index=torch.tensor([sender,receiver], dtype=torch.long),
                                                veh_edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                                                veh_id=torch.tensor(id, dtype=torch.float),

                                                veh_path_edge_index=veh_path_edge,
                                                path_lane_edge_index=path_node_edge,
                                                path_num_nodes=path_num_nodes,
                                                path_node_node_edge_index=path_node_node,
                                                lane_id = lane_ids,
                                                lane_ctrs = ctrs,
                                                lane_vecs = vecs,
                                                lane_pris = pris,
                                                lane_widths = widths,
                                                lane_suc_edge_index = suc_edges,
                                                lane_pre_edge_index = pre_edges,
                                                lane_left_edge_index = left_edges,
                                                lane_right_edge_index = right_edges,
                                                lane_start = lane_start,
                                                lane_path = lane_path,
                                                )
        return data

    def process(self):
        trajectories, vehicleinfos, lanelet_networks = self.trajectories, self.vehicleinfos, self.lanelet_networks
        true_ego_vehs = defaultdict(lambda:dict())
        i = 0
        offset = self.config["primesteps"] + self.config["horizon_steps"] + self.config["total_prediction_steps"]#self.config["prediction_steps"]
        #offset = self.config["primesteps"] + self.config["horizon_steps"]
        for traj_idx in range(len(vehicleinfos)):
            scene = vehicleinfos[traj_idx][0]
            #self.vehicle_exts.set_step_length_and_maneuver_horizon(self.config["step_length"],
            #                                                       self.config["maneuver_horizon"])
            vehids = list(vehicleinfos[traj_idx][1].keys())
            print("{} Number of vehicles: {}".format(self.raw_paths[traj_idx], len(vehids)))
            lanelet_network = lanelet_networks[vehicleinfos[traj_idx][0]]
            # select ego ids & time period
            for ego_index, egoid in enumerate(vehids):
                ts, te = vehicleinfos[traj_idx][1][egoid]["frames"]
                if ts >= te - offset or (scene == "u101" and egoid <=50):
                    continue
                ego_class = vehicleinfos[traj_idx][1][egoid]["type"]
                if ego_class==ObstacleType.TRAILER or ego_class==ObstacleType.PEDESTRIAN or ego_class==ObstacleType.BICYCLE or ego_class==ObstacleType.MOTORCYCLE:
                    continue
                if egoid not in trajectories[traj_idx][te].keys():
                    continue
                if scene == "rounD" and trajectories[traj_idx][te][egoid].posF.ind[1] in [104, 109, 114, 115, 132]:
                    continue
                max_frame = np.amax(list(trajectories[traj_idx].keys()))
                if scene == "rounD" and te >= max_frame - 300:
                    continue
                if ego_index%100 == 0:
                    print("current ego_vehicle, {}".format(egoid))
                true_ego_vehs = defaultdict(lambda:dict())
                for t in range(ts, te+1):
                    for vehid in trajectories[traj_idx][t].keys():
                        true_ego_vehs[t][vehid] = DynamicObstacle(obstacle_id=vehid,
                                obstacle_type=vehicleinfos[traj_idx][1][vehid]["type"],
                                initial_state=trajectories[traj_idx][t][vehid],
                                obstacle_shape=vehicleinfos[traj_idx][1][vehid]["shape"])
                expert_cr_scenario = ScenarioWrapper(self.config["dt"], lanelet_network,
                                                      vehicleinfos[traj_idx][0], true_ego_vehs, None)
                expert_cr_scenario.set_sensor_range(self.config["max_veh_radius"],
                                                    self.config["max_veh_disp_front"],
                                                    self.config["max_veh_disp_rear"],
                                                    self.config["max_veh_disp_side"])
                np.random.seed(0)
                #tc = np.random.randint(ts+self.config["delta_step"], ts + self.config["primesteps"]+self.config["delta_step"])
                tc = ts + self.config["primesteps"]
                #tc = ts
                while tc + self.config["horizon_steps"] + self.config["prediction_steps"] <= te:
                    time_steps = list(range(tc, tc+self.config["horizon_steps"], self.config["delta_step"]))
                    data = self.generate(expert_cr_scenario, egoid, time_steps)
                    if data is None:
                        break

                    torch.save(data, osp.join(self.processed_dir, 'data_{}_{}.pt'.format(traj_idx, i)))
                    tc += self.config["skip_steps"]
                    i += 1

    def len(self):
        return len(self.processed_file_names)

    def add_routes(self, files):
        self.lane_start, self.lane_target = [], []
        self.lane_path, self.lane_path_weight = [], []
        self.veh_path = []
        for f in files:
            data = torch.load(f)
            has_preds = data.veh_has_preds
            num_preds = has_preds.size(1)
            last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
                has_preds.device
            ) / float(num_preds)
            max_last, last_idcs = last.max(1)
            row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
            veh_start_ctrs = data.veh_xseq[:,-1,:2]
            veh_end_ctrs = data.veh_yseq[:,:,:2]
            veh_end_ctrs = veh_end_ctrs[row_idcs, last_idcs]
            lane_ctrs = data.lane_ctrs
            veh_speeds = data.veh_xseq[:,-1,2:4]/0.2
            dist = veh_start_ctrs.view(-1,1,2) - lane_ctrs.view(1,-1,2)
            dist = torch.sqrt((dist**2).sum(2))
            sorted_dist, sorted_idcs = dist.sort(1, descending=False)
            min_start_idcs = sorted_idcs[:,:8]

            dist = veh_end_ctrs.view(-1,1,2) - lane_ctrs.view(1,-1,2)
            dist = torch.sqrt((dist**2).sum(2))
            sorted_dist, sorted_idcs = dist.sort(1, descending=False)
            min_end_idcs = sorted_idcs[:,:8]
            # find the shortest paths
            DG = nx.DiGraph()
            for n in range(lane_ctrs.size(0)):
                DG.add_node(n)

            flr_edge_index = torch.cat([data.lane_suc_edge_index[0],
                                        data.lane_left_edge_index,
                                        data.lane_right_edge_index], -1)
            for j in range(flr_edge_index.size(-1)):
                DG.add_edge(flr_edge_index[0][j].item(), flr_edge_index[1][j].item(), t=1.0)

            ii = torch.tensor([np.arange(lane_ctrs.size(0)),
                               np.arange(lane_ctrs.size(0))], dtype=torch.float)
            new_edge_indexs, _ = add_self_loops(flr_edge_index)
            AI = torch.sparse_coo_tensor(new_edge_indexs,
                                        torch.ones_like(new_edge_indexs[0], dtype=torch.float),
                                        (lane_ctrs.size(0),lane_ctrs.size(0)))

            D = degree(new_edge_indexs[0], lane_ctrs.size(0), dtype=torch.float)
            D2 = torch.sparse_coo_tensor(ii, D**(-0.5), dtype=torch.float)
            S = torch.mm(torch.mm(D2, AI.to_dense()), D2.to_dense())

            SS = S
            S_mask = SS > 0
            SS_max, _ = SS.max(1)
            SS_norm = SS / SS_max.unsqueeze(1)
            idcs = torch.nonzero(S_mask, as_tuple=False)
            paths = {0:idcs.T}
            path_weights = {0:SS_norm[S_mask].unsqueeze(1)}
            selection = {5:1, 10:2, 15:3, 20:4, 25:5}
            for p in range(24):
                SS = torch.mm(S, SS)
                S_mask = SS > 1e-3
                SS_max, _ = SS.max(1)
                SS_norm = SS / SS_max.unsqueeze(1)
                idcs = torch.nonzero(S_mask, as_tuple=False)
                if p+2 in selection:
                    paths[selection[p+2]] = idcs.T
                    path_weights[selection[p+2]] = SS_norm[S_mask].unsqueeze(1)

            self.lane_path.append(paths)
            self.lane_path_weight.append(path_weights)

            obs_veh_index, fut_veh_index = [], []
            obs_lane_index, fut_lane_index = [], []
            actor_paths = []
            for v in range(veh_start_ctrs.size(0)):
                first_ns = local_search(min_start_idcs[v].numpy(), data.lane_suc_edge_index[0], "start")
                last_ns = local_search(min_end_idcs[v].numpy(), data.lane_suc_edge_index[0], "end")

                obs_veh_index.extend([v for _ in range(len(first_ns))])
                obs_lane_index.extend(first_ns)

                fut_veh_index.extend([v for _ in range(len(last_ns))])
                fut_lane_index.extend(last_ns)
                path_lengths = nx.multi_source_dijkstra_path_length(DG, set(first_ns))
                ks = [21]
                for l in last_ns:
                    if l in path_lengths:
                        if path_lengths[l] == 0:
                            ks.append(0)
                        elif path_lengths[l] > 30:
                            ks.append(1)
                        else:
                            ks.append(path_lengths[l]-1)
                actor_paths.append(np.min(ks))

            self.lane_start.append(torch.tensor([obs_veh_index,obs_lane_index], dtype=torch.long))
            self.lane_target.append(torch.tensor([fut_veh_index,fut_lane_index], dtype=torch.long))
            self.veh_path.append(torch.tensor(actor_paths, dtype=torch.long).unsqueeze(1))



    def get(self, idx):
        f = self.processed_file_names[idx]
        old_data = torch.load(f)
        kwargs = {k: v for k, v in old_data.__dict__.items() if not k.startswith('_')}
        # Reconstruct with new ScenarioData class binding
        data = ScenarioData(**kwargs)
        return data
    
    # some visualization functions()
    def update_figure(self, K, svid):
        d = self.current_data
        veh_ids = d.veh_id.numpy().astype(int)

        veh_ctrs = d.veh_xseq[:,-1,:2]
        lane_ctrs = d.lane_ctrs#d.graphs[0]['ctrs'].cpu()
        dist = veh_ctrs.view(-1,1,2) - lane_ctrs.view(1,-1,2)
        dist = torch.sqrt((dist**2).sum(2))
        _, min_idcs = dist.min(-1)
        veh_grid_idx = min_idcs[svid].item()

        weights = np.zeros_like(veh_ids)
        weights[0] = 1.0
        senders, receivers = d.veh_edge_index.numpy()

        x_node_attr = d.veh_xseq.numpy()
        y_node_attr = d.veh_yseq.numpy()
        has_preds = d.veh_has_preds.numpy()
        mains, surroundings = [], []
        for idx, veh_id in enumerate(veh_ids):
            if weights[idx] == 1.0:
                mains.append(veh_id)
                ego_posG = self.true_ego_vehs[self.time_steps[-1]][veh_id].initial_state.posG
            else:
                surroundings.append(veh_id)
        scene, ego_vehs, nei_vehs = self.expert_cr_scenario.commonroad_scenario_at_time_step(self.time_steps[-1], mains, surroundings)
        plot_limits = np.array([ego_posG.x-self.config["dw"], ego_posG.x+self.config["dw"], ego_posG.y-self.config["dh"], ego_posG.y+self.config["dh"]])
        draw_object(scene, ax=self.ax, draw_params={'scenario':{'dynamic_obstacle':{'show_label': True, 'draw_shape': True, 'shape': {'facecolor':'#aaaaaa','edgecolor': '#aaaaaa'}}}},
                     plot_limits = plot_limits)
        draw_object(ego_vehs, ax=self.ax, draw_params=draw_params_ego)
        draw_object(nei_vehs, ax=self.ax, draw_params=draw_params_neighbor)

        main_obs_lines, surr_obs_lines,  main_fut_lines, surr_fut_lines = [], [], [], []

        tar = d.veh_full_path == 1.0
        path_idx = d.veh_path_edge_index[1][tar.squeeze()]

        for idx, veh_id in enumerate(veh_ids):
            delta_x, delta_y = x_node_attr[idx,:,0], x_node_attr[idx,:,1]
            has_obs = x_node_attr[idx,:,4]
            mask = has_obs > 0.0
            s, c = np.sin(ego_posG.th), np.cos(ego_posG.th)
            center_point_xs = c*delta_x[mask] - s*delta_y[mask] + ego_posG.x
            center_point_ys = s*delta_x[mask] + c*delta_y[mask] + ego_posG.y

            fut_delta_x, fut_delta_y = y_node_attr[idx,:,0], y_node_attr[idx,:,1]
            has_pred = has_preds[idx]
            fut_center_point_xs = c*fut_delta_x[has_pred] - s*fut_delta_y[has_pred] + ego_posG.x
            fut_center_point_ys = s*fut_delta_x[has_pred] + c*fut_delta_y[has_pred] + ego_posG.y
            fut_center_point_xs = np.concatenate([center_point_xs[-1:], fut_center_point_xs], 0)
            fut_center_point_ys = np.concatenate([center_point_ys[-1:], fut_center_point_ys], 0)

            if weights[idx] == 1.0:
                color = "#c30000"
                main_obs_lines.append(list(zip(center_point_xs, center_point_ys)))
                main_fut_lines.append(list(zip(fut_center_point_xs, fut_center_point_ys)))
            else:
                color = "#2469ff"
                surr_obs_lines.append(list(zip(center_point_xs, center_point_ys)))
                surr_fut_lines.append(list(zip(fut_center_point_xs, fut_center_point_ys)))
            plotted_centroid = plt.Circle((center_point_xs[-1], center_point_ys[-1]), 1.6, zorder= 20, color = "k")
            self.ax.add_patch(plotted_centroid)
            plotted_centroid = plt.Circle((center_point_xs[-1], center_point_ys[-1]), 1.0, zorder= 21, color = "w")
            self.ax.add_patch(plotted_centroid)
            plotted_centroid = plt.Circle((center_point_xs[-1], center_point_ys[-1]), 0.6, zorder= 21, color = color)
            self.ax.add_patch(plotted_centroid)
            '''
            for t in self.time_steps:
                if veh_id in self.true_ego_vehs[t]:
                    veh_posG = self.true_ego_vehs[t][veh_id].initial_state.posG
                    self.ax.add_patch(plt.Circle((veh_posG.x, veh_posG.y), 0.3, zorder= 21))


            for t in range(self.time_steps[-1]+self.config["delta_step"], self.time_steps[-1]+self.config["delta_step"]+self.config["prediction_steps"], self.config["delta_step"]):
                if veh_id in self.true_ego_vehs[t]:
                    veh_posG = self.true_ego_vehs[t][veh_id].initial_state.posG
                    self.ax.add_patch(plt.Circle((veh_posG.x, veh_posG.y), 0.3, color = "g", zorder= 21))
            '''
            # the frenet Coordinates
            vertices = []
            for n in d.path_lane_edge_index[1][d.path_lane_edge_index[0]==path_idx[idx]]:
                #lanelet = expert_cr_scenario.lanelet_network.find_lanelet_by_id(lid)
                vertices.append(lane_ctrs[n].numpy())
            vertices = np.array(vertices)
            target_curve = make_curve(vertices)
            start_pos = VecSE2(delta_x[-1], delta_y[-1], 0.0)
            start_proj = start_pos.proj_on_curve(target_curve, clamped=False)
            start_s = lerp_curve_with_ind(target_curve, start_proj.ind)
            start_ind = start_proj.ind

            yy = []
            for t in range(y_node_attr.shape[1]):
                if has_pred[t]:
                    #print("true pos", VecSE2(y_node_attr[idx,t,0], y_node_attr[idx,t,1], 0.0).proj_on_curve(target_curve, clamped=False).ind.t)
                    start_ind, start_pos = move_along_curve(start_ind, target_curve, d.veh_yfre[idx, t, 0].item(), d.veh_yfre[idx, t, 1].item())
                    yy.append([start_pos.x, start_pos.y, d.veh_yfre[idx, t, 0].item(), d.veh_yfre[idx, t, 1].item()])
                    x = c*start_pos.x - s*start_pos.y + ego_posG.x
                    y = s*start_pos.x + c*start_pos.y + ego_posG.y

                    self.ax.add_patch(plt.Circle((x, y), 0.3, zorder= 31))
            #print("yyf", np.array(yy))
            #print("yyc", y_node_attr[idx,:,:2])

        main_obs_traj = mc.LineCollection(main_obs_lines, colors="#ff8888", linewidths=2, zorder=27)
        surr_obs_traj = mc.LineCollection(surr_obs_lines, colors="#b5d0ff", linewidths=2, zorder=25)
        self.ax.add_collection(main_obs_traj)
        self.ax.add_collection(surr_obs_traj)
        main_fut_traj = mc.LineCollection(main_fut_lines, colors="r", linewidths=2, zorder=27)
        surr_fut_traj = mc.LineCollection(surr_fut_lines, colors="b", linewidths=2, zorder=26)
        self.ax.add_collection(main_fut_traj)
        self.ax.add_collection(surr_fut_traj)
        #predictions
        '''
        s, c = np.sin(ego_posG.th), np.cos(ego_posG.th)
        if "reg" in self.ret:
            # top-k predictions
            K_pred_traj_fake_rel = self.ret['reg']
            score = torch.nn.functional.softmax(self.ret['score'], dim=1)
            predict_line = []
            linewidths = []
            colors = []
            for k in range(self.config['num_mods']):
                pred_rel = K_pred_traj_fake_rel[k].permute(1, 0, 2)
                pred_traj = relative_to_abs(pred_rel, self.current_data.obs_traj[-1])
                for n in range(score.size(0)):
                    pred_traj_n = pred_traj[:,n][self.current_data.has_preds[:,n]].cpu().numpy()

                    pred_fut_center_point_xs = c*pred_traj_n[:,0] - s*pred_traj_n[:,1] + ego_posG.x
                    pred_fut_center_point_ys = s*pred_traj_n[:,0] + c*pred_traj_n[:,1] + ego_posG.y

                    predict_line.append(list(zip(pred_fut_center_point_xs, pred_fut_center_point_ys)))
                    linewidths.append(max(10*score[n,k].cpu().item(), 0.1))
                    if n==0:
                        colors.append("#800000")
                    else:
                        colors.append("#001e9d")
            self.ax.add_collection(mc.LineCollection(predict_line, colors=colors, linewidths=linewidths, zorder=27, linestyle='dashed'))

        else:
            # single predictions
            pred_rel = self.ret['pred']
            pred_traj = relative_to_abs(pred_rel, self.current_data.obs_traj[-1])
            predict_line = []
            colors = []
            for n in range(pred_rel.size(1)):
                pred_traj_n = pred_traj[:,n][self.current_data.has_preds[:,n]].cpu().numpy()
                pred_fut_center_point_xs = c*pred_traj_n[:,0] - s*pred_traj_n[:,1] + ego_posG.x
                pred_fut_center_point_ys = s*pred_traj_n[:,0] + c*pred_traj_n[:,1] + ego_posG.y

                predict_line.append(list(zip(pred_fut_center_point_xs, pred_fut_center_point_ys)))
                if n==0:
                    colors.append("#800000")
                else:
                    colors.append("#001e9d")


            self.ax.add_collection(mc.LineCollection(predict_line, colors=colors, linewidths=2, zorder=27, linestyle='dashed'))
        '''
        self.ax.set_aspect('equal')

        lines = []
        for edge_ind in range(len(senders)):
            sender_ind = senders[edge_ind]
            receiver_ind = receivers[edge_ind]
            sender_id = veh_ids[sender_ind]
            receiver_id = veh_ids[receiver_ind]
            if sender_id == receiver_id:
                continue
            sender_posG = self.true_ego_vehs[self.time_steps[-1]][sender_id].initial_state.posG
            receiver_posG = self.true_ego_vehs[self.time_steps[-1]][receiver_id].initial_state.posG
            line = [(sender_posG.x, sender_posG.y), (receiver_posG.x, receiver_posG.y)]
            lines.append(line)
        #self.ax.add_collection(mc.LineCollection(lines, linewidths=2.5, zorder= 19, color = "k"))

        grid_collection, color_collection = [], []
        cm = plt.get_cmap('afmhot')

        #lane_path = d.lane_path[K][1][d.lane_path[K][0]==svid]
        lane_path = d.path_lane_edge_index[1][d.path_lane_edge_index[0]==K]
        #lane_path = d.graphs[0]['path'][K][1][d.graphs[0]['path'][K][0]==svid]
        veh_target = d.veh_target
        for i, grid in enumerate(lane_ctrs):
            delta_x, delta_y = grid[0], grid[1]
            s, c = np.sin(ego_posG.th), np.cos(ego_posG.th)
            grid_x = c*delta_x - s*delta_y + ego_posG.x
            grid_y = s*delta_x + c*delta_y + ego_posG.y
            if veh_target[svid] == i:
                color_collection.append(cm(0.0))
            elif i in lane_path:
                color_collection.append(cm(0.5))
            else:
                color_collection.append(cm(1.0))
            grid_collection.append(plt.Circle((grid_x,grid_y), 0.7))

        grid_collection = mc.PatchCollection(grid_collection,
                                              facecolors=color_collection,
                                              edgecolors='#68478D', zorder=15)
        self.ax.add_collection(grid_collection)


        suc_collection = []
        #for src, rec in zip(d.graphs[0]['suc'][0][0].cpu().numpy(), d.graphs[0]['suc'][0][1].cpu().numpy()):
        for src, rec in zip(d.lane_suc_edge_index[0][0].numpy(), d.lane_suc_edge_index[0][1].numpy()):
            s_delta_x, s_delta_y = lane_ctrs[src, 0], lane_ctrs[src, 1]
            s, c = np.sin(ego_posG.th), np.cos(ego_posG.th)
            s_grid_x = c*s_delta_x - s*s_delta_y + ego_posG.x
            s_grid_y = s*s_delta_x + c*s_delta_y + ego_posG.y

            r_delta_x, r_delta_y = lane_ctrs[rec, 0], lane_ctrs[rec, 1]
            s, c = np.sin(ego_posG.th), np.cos(ego_posG.th)
            r_grid_x = c*r_delta_x - s*r_delta_y + ego_posG.x
            r_grid_y = s*r_delta_x + c*r_delta_y + ego_posG.y
            line = [(s_grid_x, s_grid_y), (r_grid_x, r_grid_y)]
            suc_collection.append(line)

        suc_collection = mc.LineCollection(suc_collection, linewidths=0.8, colors = '#68478D',zorder= 15)
        self.ax.add_collection(suc_collection)

        left_collection = []
        #for src, rec in zip(d.graphs[0]['left'][0].cpu().numpy(), d.graphs[0]['left'][1].cpu().numpy()):
        for src, rec in zip(d.lane_left_edge_index[0].numpy(), d.lane_left_edge_index[1].numpy()):
            s_delta_x, s_delta_y = lane_ctrs[src, 0], lane_ctrs[src, 1]
            s, c = np.sin(ego_posG.th), np.cos(ego_posG.th)
            s_grid_x = c*s_delta_x - s*s_delta_y + ego_posG.x
            s_grid_y = s*s_delta_x + c*s_delta_y + ego_posG.y

            r_delta_x, r_delta_y = lane_ctrs[rec, 0], lane_ctrs[rec, 1]
            s, c = np.sin(ego_posG.th), np.cos(ego_posG.th)
            r_grid_x = c*r_delta_x - s*r_delta_y + ego_posG.x
            r_grid_y = s*r_delta_x + c*r_delta_y + ego_posG.y
            line = [(s_grid_x, s_grid_y), (r_grid_x, r_grid_y)]
            left_collection.append(line)

        left_collection = mc.LineCollection(left_collection, linewidths=0.8, colors = '#11358D',zorder= 15)
        self.ax.add_collection(left_collection)

        right_collection = []
        #for src, rec in zip(d.graphs[0]['right'][0].cpu().numpy(), d.graphs[0]['right'][1].cpu().numpy()):
        for src, rec in zip(d.lane_right_edge_index[0].numpy(), d.lane_right_edge_index[1].numpy()):
            s_delta_x, s_delta_y = lane_ctrs[src, 0], lane_ctrs[src, 1]
            s, c = np.sin(ego_posG.th), np.cos(ego_posG.th)
            s_grid_x = c*s_delta_x - s*s_delta_y + ego_posG.x
            s_grid_y = s*s_delta_x + c*s_delta_y + ego_posG.y

            r_delta_x, r_delta_y = lane_ctrs[rec, 0], lane_ctrs[rec, 1]
            s, c = np.sin(ego_posG.th), np.cos(ego_posG.th)
            r_grid_x = c*r_delta_x - s*r_delta_y + ego_posG.x
            r_grid_y = s*r_delta_x + c*r_delta_y + ego_posG.y
            line = [(s_grid_x, s_grid_y), (r_grid_x, r_grid_y)]
            right_collection.append(line)

        right_collection = mc.LineCollection(right_collection, linewidths=0.8, colors = 'r',zorder= 15)
        self.ax.add_collection(right_collection)

    def remove_collections(self):
        while len(self.ax.collections) > 0:
            self.ax.collections[0].remove()
        while len(self.ax.texts) > 0:
            self.ax.texts[0].remove()
        while len(self.ax.patches) > 0:
            self.ax.patches[0].remove()

    def update_button_next(self, _):
        if self.step + self.config["delta_step"] < self.te:
            gc.collect()
            self.step += self.config["delta_step"]
            self.changed_button = True
            self.time_steps = list(range(self.step, self.step+self.config["horizon_steps"], self.config["delta_step"]))
            #print("before g", self.step, self.te)
            data = self.generate(self.expert_cr_scenario, self.egoid, self.time_steps)
            d = DataLoader([data], batch_size=1, shuffle=False, drop_last=False, collate_fn=traj_collate)
            #print("after d")
            self.current_data = next(iter(d))
            if self.model is not None:
                with torch.no_grad():
                    self.ret = self.model(self.current_data)
            #print("after m")
            self.trigger_update()

    def update_button_previous(self, _):
        if self.step - self.config["delta_step"] > self.ts:
            self.step -= self.config["delta_step"]
            self.changed_button = True
            self.time_steps = list(range(self.step, self.step+self.config["horizon_steps"], self.config["delta_step"]))
            data = self.generate(self.expert_cr_scenario, self.egoid, self.time_steps)
            d = DataLoader([data], batch_size=1, shuffle=False, drop_last=False, collate_fn=traj_collate)
            self.current_data = next(iter(d))
            if self.model is not None:
                with torch.no_grad():
                    self.ret = self.model(self.current_data)
            self.trigger_update()


    def trigger_update(self):
        self.remove_collections()
        self.update_figure(self.K, self.svid)
        self.fig.canvas.draw_idle()

    def start_play(self, _):
        self.timer.start()

    def stop_play(self, _):
        self.timer.stop()

    def render(self, idx, model=None, h=10, w=10, K=1, svid=0):
        self.h, self.w = h, w
        self.K , self.svid = K, svid
        trajectories, vehicleinfos, lanelet_networks = self.trajectories, self.vehicleinfos, self.lanelet_networks
        #self.current_data = self.get(idx)
        traj_idx = int(self.processed_file_names[idx].split('_')[1])
        scene = vehicleinfos[traj_idx][0]
        self.lanelet_network = lanelet_networks[scene]
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(self.h,self.w))
        #d = DataLoader([self.get(idx)], batch_size=1, shuffle=False, drop_last=False, collate_fn=traj_collate)
        self.current_data = self.get(idx)
        '''
        if model is not None:
            model.eval()
            self.model = model
            with torch.no_grad():
                self.ret = model(self.current_data)
        '''
        self.time_steps = self.current_data.veh_t.numpy().astype(int)
        self.egoid = self.current_data.veh_id.numpy().astype(int)[0]
        ts, te = vehicleinfos[traj_idx][1][self.egoid]["frames"]
        self.ts, self.te = ts, te
        self.step = self.time_steps[0]

        true_ego_vehs = defaultdict(lambda:dict())
        for t in range(ts, te+self.config["prediction_steps"]+1):
            for vehid in trajectories[traj_idx][t].keys():
                true_ego_vehs[t][vehid] = DynamicObstacle(obstacle_id=vehid,
                            obstacle_type=vehicleinfos[traj_idx][1][vehid]["type"],
                            initial_state=trajectories[traj_idx][t][vehid],
                            obstacle_shape=vehicleinfos[traj_idx][1][vehid]["shape"])
        self.expert_cr_scenario = ScenarioWrapper(self.config["dt"], self.lanelet_network,
                                                  vehicleinfos[traj_idx][0], true_ego_vehs, None)
        self.expert_cr_scenario.set_sensor_range(self.config["max_veh_radius"],
                                                 self.config["max_veh_disp_front"],
                                                 self.config["max_veh_disp_rear"],
                                                 self.config["max_veh_disp_side"])
        self.true_ego_vehs = true_ego_vehs

        self.ax.set_xticks([])
        self.ax.set_xticklabels([])
        self.ax.set_yticks([])
        self.ax.set_yticklabels([])

        self.update_figure(self.K, self.svid)
        self.ax_button_previous = self.fig.add_axes([0.37, 0.035, 0.06, 0.04])
        self.ax_button_next = self.fig.add_axes([0.44, 0.035, 0.06, 0.04])
        self.ax_button_play = self.fig.add_axes([0.58, 0.035, 0.06, 0.04])
        self.ax_button_stop = self.fig.add_axes([0.65, 0.035, 0.06, 0.04])

        self.button_previous = Button(self.ax_button_previous, 'Previous')
        self.button_next = Button(self.ax_button_next, 'Next')
        self.button_play = Button(self.ax_button_play, 'Play')
        self.button_stop = Button(self.ax_button_stop, 'Stop')

        self.button_next.on_clicked(self.update_button_next)
        self.button_previous.on_clicked(self.update_button_previous)
        self.button_play.on_clicked(self.start_play)
        self.button_stop.on_clicked(self.stop_play)
