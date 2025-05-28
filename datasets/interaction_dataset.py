import numpy as np
import torch
from torch.utils.data import Dataset
# from scipy import sparse
import os
import os.path as osp
# import copy
# import csv
# import pickle
# import re
# from pandas import read_csv
# from tqdm import tqdm
# import math
# import matplotlib.pyplot as plt
from datasets.scenario_dataset import ScenarioData
from typing import Callable, List, Optional, Tuple, Union
# from fjmp_utils import *
# from fjmp_metrics import *
# import lanelet2
# from lanelet2.projection import UtmProjector
# from av2.geometry.interpolate import compute_midpoint_line
class InteractionDataset(Dataset):
    def __init__(self, config, processed_files=None):
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
            raise ValueError("processed_files must be provided")
        else:
            self._processed_file_names = processed_files
            self._num_samples = len(processed_files)
        self.avg_pedcyc_length = 0.7
        self.avg_pedcyc_width = 0.7

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self.raw_files

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    def __len__(self):
        return self._num_samples

    def generate_ref_token(self):
        pass

    def __getitem__(self, idx):
        # with open(self.raw_paths[idx], 'rb') as handle:
        #     data = pickle.load(handle)
        f = self.processed_file_names[idx]
        old_data = torch.load(f)
        kwargs = {k: v for k, v in old_data.__dict__.items() if not k.startswith('_')}
        # Reconstruct with new ScenarioData class binding
        data = ScenarioData(**kwargs)
        data = self.preprocess(data, idx)
        return data

    def preprocess(self, data: ScenarioData, idx):
        new_data = dict()
        new_data['idx'] = idx
        new_data['feats'] = torch.cat([data.veh_xseq[:, :, 2:4], data.veh_yseq[:,:,2:4]], dim=1)
        new_data['feat_locs'] = torch.cat([data.veh_xseq[:, :, 0:2], data.veh_yseq[:, :, 0:2]], dim=1)
        new_data['feat_vels'] = new_data['feats']/ 0.2
        new_data['feat_agenttypes'] = torch.ones((new_data['feats'].shape[0], new_data['feats'].shape[1], 1), dtype=torch.int64)
        new_data['feat_psirads'] = torch.atan2(new_data['feats'][:, :, 1], new_data['feats'][:, :, 0]).unsqueeze(-1)
        new_data['feat_shapes'] = data.veh_shape.unsqueeze(1).expand(-1, new_data['feats'].shape[1], -1)
        new_data['ctrs'] = data.veh_xseq[:, -1, 0:2]
        new_data['has_obss']  = torch.cat([data.veh_xseq[:,:,4].bool(), data.veh_has_preds], dim=1)
        new_data['has_preds'] = data.veh_has_preds
        is_valid_agent = new_data['has_obss'][:, 9] == 1 
        ig_labels_sparse = self.get_interaction_labels_sparse(idx, new_data['ctrs'], new_data['feat_locs'], new_data['feat_vels'], new_data['feat_psirads'], new_data['feat_shapes'], new_data['has_obss'], is_valid_agent, new_data['feat_agenttypes'])
        ig_labels_sparse = np.asarray(ig_labels_sparse)
        new_data['ig_labels_sparse'] = torch.tensor(ig_labels_sparse, dtype=torch.float32)

        new_data['gt_preds'] = data.veh_yseq[:, :, 0:2]
        new_data['gt_psirads'] = new_data['feat_psirads'][:, 10:]
        new_data['gt_vels'] = new_data['feat_vels'][:, 10:]

        graph = {"ctrs":data.lane_ctrs,
                 "feats":data.lane_vecs,
                 "pris":data.lane_pris,
                 "ids":data.lane_id,
                 "num_nodes":data.lane_ctrs.size(0),
                 "pre":{k:v for k, v in data.lane_pre_edge_index.items()},
                 "suc":{k:v for k, v in data.lane_suc_edge_index.items()},
                 "left":data.lane_left_edge_index,
                 "right":data.lane_right_edge_index,
                 }
        new_data['graph'] = graph
        return new_data

    def get_interaction_labels_sparse(self, idx, ctrs, feat_locs, feat_vels, feat_psirads, shapes, has_obss, is_valid_agent, agenttypes):

        # only consider the future
        # we can use data in se(2) transformed coordinates (interaction labelling invariant to se(2)-transformations)
        feat_locs = np.array(feat_locs.cpu().numpy())
        feat_vels = np.array(feat_vels.cpu().numpy())
        feat_psirads = np.array(feat_psirads.cpu().numpy())
        shapes = np.array(shapes.cpu().numpy())
        has_obss = np.array(has_obss.cpu().numpy())
        agenttypes = np.array(agenttypes.cpu().numpy())
        is_valid_agent = np.array(is_valid_agent.cpu().numpy())
        
        feat_locs = feat_locs[:, 10:]
        feat_vels = feat_vels[:, 10:]
        feat_psirads = feat_psirads[:, 10:]
        has_obss = has_obss[:, 10:]
        
        N = feat_locs.shape[0]
        labels = np.zeros((N, N))
        orig_trajs = feat_locs 

        circle_lists = []
        for i in range(N):
            agenttype_i = agenttypes[i][9]
            if agenttype_i == 1:
                shape_i = shapes[i][9]
                length = shape_i[0]
                width = shape_i[1]
            else:
                length = self.avg_pedcyc_length
                width = self.avg_pedcyc_width

            traj_i = orig_trajs[i][has_obss[i] == 1]
            psirad_i = feat_psirads[i][has_obss[i] == 1]
            # shape is [30, c, 2], where c is the number of circles prescribed to vehicle i (depends on the size/shape of vehicle i)
            circle_lists.append(return_circle_list(traj_i[:, 0], traj_i[:, 1], length, width, psirad_i[:, 0]))
        
        for a in range(1, N):
            for b in range(a):
                agenttype_a = agenttypes[a][9]
                if agenttype_a == 1:
                    shape_a = shapes[a][9]
                    width_a = shape_a[1]
                else:
                    width_a = self.avg_pedcyc_width

                agenttype_b = agenttypes[b][9]
                if agenttype_b == 1:
                    shape_b = shapes[b][9]
                    width_b = shape_b[1]
                else:
                    width_b = self.avg_pedcyc_width
                
                # for each (unordered) pairs of vehicles, we check if they are interacting
                # by checking if there is a collision at any pair of future timesteps. 
                circle_list_a = circle_lists[a]
                circle_list_b = circle_lists[b]

                # threshold determined according to widths of vehicles
                thresh = return_collision_threshold(width_a, width_b)

                dist = np.expand_dims(np.expand_dims(circle_list_a, axis=1), axis=2) - np.expand_dims(np.expand_dims(circle_list_b, axis=0), axis=3)
                dist = np.linalg.norm(dist, axis=-1, ord=2)
                # dist = circle_list_a.unsqueeze(1).unsqueeze(2) - circle_list_b.unsqueeze(0).unsqueeze(3)
                # dist = torch.norm(dist, dim=-1, p=2)
                # [T_a, T_b, num_circles_a, num_circles_b], where T_a is the number of ground-truth future positions present in a's trajectory, and b defined similarly.
                is_coll = dist < thresh
                is_coll_cumul = is_coll.sum(2).sum(2)
                # binary mask of shape [T_a, T_b]
                is_coll_mask = is_coll_cumul > 0
                # is_coll_mask = np.array(is_coll_mask.cpu().numpy())
                if is_coll_mask.sum() < 1:
                    continue

                # fill in for indices (0) that do not have a ground-truth position
                for en, ind in enumerate(has_obss[a]):
                    if ind == 0:
                        is_coll_mask = np.insert(is_coll_mask, en, 0, axis=0)

                for en, ind in enumerate(has_obss[b]):
                    if ind == 0:
                        is_coll_mask = np.insert(is_coll_mask, en, 0, axis=1)  

                assert is_coll_mask.shape == (15, 15)

                # [P, 2], first index is a, second is b; P is number of colliding pairs
                coll_ids = np.argwhere(is_coll_mask == 1)
                # only preserve the colliding pairs that are within 2.5 seconds (= 25 timesteps) of eachother
                valid_coll_mask = np.abs(coll_ids[:, 0] - coll_ids[:, 1]) <= 25

                if valid_coll_mask.sum() < 1:
                    continue

                coll_ids = coll_ids[valid_coll_mask]
                
                # first order small_timestep, larger_timestep, index_of_larger_timestep
                coll_ids_sorted = np.sort(coll_ids, axis=-1)
                coll_ids_argsorted = np.argsort(coll_ids, axis=-1)

                conflict_time_influencer = coll_ids_sorted[:, 0].min()
                influencer_mask = coll_ids_sorted[:, 0] == conflict_time_influencer
                candidate_reactors = coll_ids_sorted[coll_ids_sorted[:, 0] == conflict_time_influencer][:, 1]
                conflict_time_reactor = candidate_reactors.min()
                conflict_time_reactor_id = np.argmin(candidate_reactors)

                a_is_influencer = coll_ids_argsorted[influencer_mask][conflict_time_reactor_id][0] == 0
                if a_is_influencer:
                    min_a = conflict_time_influencer 
                    min_b = conflict_time_reactor 
                else:
                    min_a = conflict_time_reactor 
                    min_b = conflict_time_influencer
                
                # a is the influencer
                if min_a < min_b:
                    labels[a, b] = 1
                # b is the influencer
                elif min_b < min_a:
                    labels[b, a] = 1
                else:                    
                    # if both reach the conflict point at the same timestep, the influencer is the vehicle with the higher velocity @ the conflict point.
                    if np.linalg.norm(feat_vels[a][min_a], ord=2) > np.linalg.norm(feat_vels[b][min_b], ord=2):
                        labels[a, b] = 1
                    elif np.linalg.norm(feat_vels[a][min_a], ord=2) < np.linalg.norm(feat_vels[b][min_b], ord=2):
                        labels[b, a] = 1
                    else:
                        labels[a, b] = 0
                        labels[b, a] = 0
        
        n_agents = labels.shape[0]

        assert n_agents == np.sum(is_valid_agent)

        # labels for interaction visualization
        valid_mask = is_valid_agent

        # add indices for the invalid agents (no gt position at timestep 9)
        for ind in range(valid_mask.shape[0]):
            if valid_mask[ind] == 0:
                labels = np.insert(labels, ind, 0, axis=1)

        for ind in range(valid_mask.shape[0]):
            if valid_mask[ind] == 0:
                labels = np.insert(labels, ind, 0, axis=0)

        # There is a label on each (undirected) edge in the fully connected interaction graph
        ig_labels = np.zeros(int(n_agents * (n_agents - 1) / 2))
        count = 0
        for i in range(len(is_valid_agent)):
            if is_valid_agent[i] == 0:
                assert labels[i].sum() == 0
                continue
            
            for j in range(len(is_valid_agent)):
                if is_valid_agent[j] == 0:
                    assert labels[:,j].sum() == 0
                    continue
                
                # we want only the indices where i < j
                if i >= j:
                    continue 

                # i influences j
                if labels[i, j] == 1:
                    ig_labels[count] = 1
                # j influences i
                elif labels[j, i] == 1:
                    ig_labels[count] = 2
                
                count += 1   

        assert ig_labels.shape[0] == count

        return ig_labels


def return_circle_list(x, y, l, w, yaw):
    r = w/np.sqrt(2)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    if l < 4.0:
        c1 = [x-(l-w)/2*cos_yaw, y-(l-w)/2*sin_yaw]
        c2 = [x+(l-w)/2*cos_yaw, y+(l-w)/2*sin_yaw]
        c = [c1, c2]
    elif l >= 4.0 and l < 8.0:
        c0 = [x, y]
        c1 = [x-(l-w)/2*cos_yaw, y-(l-w)/2*sin_yaw]
        c2 = [x+(l-w)/2*cos_yaw, y+(l-w)/2*sin_yaw]
        c = [c0, c1, c2]
    else:
        c0 = [x, y]
        c1 = [x-(l-w)/2*cos_yaw, y-(l-w)/2*sin_yaw]
        c2 = [x+(l-w)/2*cos_yaw, y+(l-w)/2*sin_yaw]
        c3 = [x-(l-w)/2*cos_yaw/2, y-(l-w)/2*sin_yaw/2]
        c4 = [x+(l-w)/2*cos_yaw/2, y+(l-w)/2*sin_yaw/2]
        c = [c0, c1, c2, c3, c4]
    for i in range(len(c)):
        c[i] = np.stack(c[i], axis=-1)
    c = np.stack(c, axis=-2)
    return c

# def return_circle_list(x, y, l, w, yaw):
#     r = w / torch.sqrt(torch.tensor(2.0, device=w.device, dtype=w.dtype))
#     cos_yaw = torch.cos(yaw)
#     sin_yaw = torch.sin(yaw)

#     if l < 4.0:
#         c1 = torch.stack([x - (l - w)/2 * cos_yaw, y - (l - w)/2 * sin_yaw], dim=-1)
#         c2 = torch.stack([x + (l - w)/2 * cos_yaw, y + (l - w)/2 * sin_yaw], dim=-1)
#         c = torch.stack([c1, c2], dim=0)

#     elif l < 8.0:
#         c0 = torch.stack([x, y], dim=-1)
#         c1 = torch.stack([x - (l - w)/2 * cos_yaw, y - (l - w)/2 * sin_yaw], dim=-1)
#         c2 = torch.stack([x + (l - w)/2 * cos_yaw, y + (l - w)/2 * sin_yaw], dim=-1)
#         c = torch.stack([c0, c1, c2], dim=0)

#     else:
#         c0 = torch.stack([x, y], dim=-1)
#         c1 = torch.stack([x - (l - w)/2 * cos_yaw, y - (l - w)/2 * sin_yaw], dim=-1)
#         c2 = torch.stack([x + (l - w)/2 * cos_yaw, y + (l - w)/2 * sin_yaw], dim=-1)
#         c3 = torch.stack([x - (l - w)/2 * cos_yaw / 2, y - (l - w)/2 * sin_yaw / 2], dim=-1)
#         c4 = torch.stack([x + (l - w)/2 * cos_yaw / 2, y + (l - w)/2 * sin_yaw / 2], dim=-1)
#         c = torch.stack([c0, c1, c2, c3, c4], dim=0)

#     return c  # shape [num_circles, 2]

def return_collision_threshold(w1, w2):
    return (w1 + w2) / np.sqrt(3.8)