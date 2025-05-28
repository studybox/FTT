import pickle
import os
import os.path as osp
from collections import defaultdict
import numpy as np
import torch

from scipy.sparse import csr_matrix
import scipy
from scipy import sparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle, Arc, Polygon
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from commonroad.scenario.lanelet import LineMarking, LaneletType
from commonroad.scenario.trajectorycomplement import FrenetState, Frenet
from torch_geometric.data import Dataset, Data, Batch
from commonroad.scenario.scenariocomplement import MultiFeatureExtractor, CoreFeatureExtractor1, NeighborFeatureExtractor, ScenarioWrapper, LaneletToLaneletFeatureExtractor1, LaneletFeatureExtractor1, VehicleToLaneletFeatureExtractor1
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType, DynamicObstacle
from commonroad.common.file_reader_complement import LaneletCurveNetworkReader
from commonroad.geometry.shape import Rectangle

from commonroad.scenario.trajectorycomplement import FrenetState, Frenet, move_along_curve
from commonroad.scenario.laneletcomplement import *

from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad.scenario.laneletcomplement import make_lanelet_curve_network
from commonroad.common.file_reader import CommonRoadFileReader
from torch_geometric.utils import degree, add_self_loops
from torch_scatter import scatter_mean, scatter_max, scatter_add
#from config import basic_shape_parameters_ego, basic_shape_parameters_nei, basic_shape_parameters_obs, draw_params_neighbor, draw_params_ego, draw_params_obstacle
#from motion_prediction.losses import l2_loss, displacement_error, final_displacement_error

# from losses import l2_loss, displacement_error, final_displacement_error

MAX_SPEED = 45  # maximum speed [m/s]
MAX_ACCEL = 3.0  # maximum acceleration [m/ss]
MAX_DECEL = 8.0 # maximum dec[m/ss]
MAX_CURVATURE = 1.0  # maximum curvature [1/m]

from datasets.seen_data_files import all_val_files_2

def mix_dataset(Config):
    train_files, val_files_1, val_files_2 = [], [], []
    for dataset in Config["datasets"]:
        data_root_dir = Config[dataset]["data_dir"]
        data_files = [osp.join(data_root_dir, "processed", d) for d in  os.listdir(osp.join(data_root_dir, "processed")) if 'data' in d]
        if dataset in ["EP", "mcity"]:
            val_files_1.extend(data_files)
        else:
            # tf, vf = train_test_split(data_files, test_size=0.15, random_state=42)
            for file_name in data_files:
                if file_name in all_val_files_2:
                    val_files_2.append(file_name)
                else:
                    train_files.append(file_name)
            # train_files.extend(tf)
            # val_files_2.extend(vf)
    return train_files, val_files_1, val_files_2

def load_ngsim_scenarios(lanelet_network_filepaths, trajectory_filepaths):

    lanelet_networks = {}
    # load lanelet_networks
    for fp in lanelet_network_filepaths:
        if "i80" in fp:
            lanelet_networks["i80"] = LaneletCurveNetworkReader(fp).lanelet_network
        elif "u101" in fp:
            u101_lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:39.89 for ln in u101_lanelet_network.lanelets}
            lanelet_networks["u101"] = make_lanelet_curve_network(u101_lanelet_network, speed_limits)
        elif "highD2" in fp:
            lanelet_networks["highD2"] = LaneletCurveNetworkReader(fp).lanelet_network
        elif "highD3" in fp:
            lanelet_networks["highD3"] = LaneletCurveNetworkReader(fp).lanelet_network
        elif "rounD-plus" in fp:
            rounD_lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:13.89 for ln in rounD_lanelet_network.lanelets}
            if "00" in fp:
                lanelet_networks["rounD-plus00"] = make_lanelet_curve_network(rounD_lanelet_network, speed_limits)
            else:
                lanelet_networks["rounD-plus01"] = make_lanelet_curve_network(rounD_lanelet_network, speed_limits)
        elif "rounD" in fp:
            rounD_lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:13.89 for ln in rounD_lanelet_network.lanelets}
            lanelet_networks["rounD"] = make_lanelet_curve_network(rounD_lanelet_network, speed_limits)
        elif "CHN" in fp:
            lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:17.89 for ln in lanelet_network.lanelets}
            lanelet_networks["CHN"] = make_lanelet_curve_network(lanelet_network, speed_limits)
        elif "DEU" in fp:
            lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:17.89 for ln in lanelet_network.lanelets}
            lanelet_networks["DEU"] = make_lanelet_curve_network(lanelet_network, speed_limits)
        elif "SR" in fp:
            lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:17.89 for ln in lanelet_network.lanelets}
            lanelet_networks["SR"] = make_lanelet_curve_network(lanelet_network, speed_limits)
        elif "EP" in fp:
            lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:17.89 for ln in lanelet_network.lanelets}
            lanelet_networks["EP"] = make_lanelet_curve_network(lanelet_network, speed_limits)
        elif "FT" in fp:
            lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:17.89 for ln in lanelet_network.lanelets}
            lanelet_networks["FT"] = make_lanelet_curve_network(lanelet_network, speed_limits)
        elif "mcity" in fp:
            mcity_lanelet_network = CommonRoadFileReader(fp).open_lanelet_network()
            speed_limits = {ln.lanelet_id:13.89 for ln in mcity_lanelet_network.lanelets}
            lanelet_networks["mcity"] = make_lanelet_curve_network(mcity_lanelet_network, speed_limits)
        else:
            raise ValueError("Can not identify lanelet_network in {}".format(fp))

    trajectories = []
    vehicleinfo = []
    # load trajectories
    for fp in trajectory_filepaths:
        trajdata = pickle.load(open(fp, "rb"))

        obstacle_infos = {}
        obstacle_states = defaultdict(lambda:dict())
        for d in trajdata['def']:
            carid, length, width, tp, f_lo, f_hi = d
            obstacle_infos[int(carid)] = {"shape":Rectangle(length,width),
                                        "type":tp, "frames":(int(f_lo),int(f_hi))}
        for d in trajdata['state']:
            step, carid, x, y, ori, v, i, t, lid, s, d, phi = d
            state = FrenetState(position=np.array([x,y]), orientation=ori, velocity=v, time_step = int(step))
            posF = Frenet(None, None, (i, t, lid, s, d, phi))
            state.posF = posF
            obstacle_states[int(step)][int(carid)] = state

        trajectories.append(obstacle_states)
        if "i80" in fp:
            lanelane_network_id = "i80"
        elif "u101" in fp:
            lanelane_network_id = "u101"
        elif "highD2" in fp:
            lanelane_network_id = "highD2"
        elif "highD3" in fp:
            lanelane_network_id = "highD3"
        elif "rounD-plus" in fp:
            if "00" in fp:
                lanelane_network_id = "rounD-plus00"
            else:
                lanelane_network_id = "rounD-plus01"
        elif "rounD" in fp:
            lanelane_network_id = "rounD"
        elif "mcity" in fp:
            lanelane_network_id = "mcity"
        elif "CHN" in fp:
            lanelane_network_id = "CHN"
        elif "DEU" in fp:
            lanelane_network_id = "DEU"
        elif "SR" in fp:
            lanelane_network_id = "SR"
        elif "EP" in fp:
            lanelane_network_id = "EP"
        elif "FT" in fp:
            lanelane_network_id = "FT"
        else:
            raise ValueError("Can not identify trajectory in {}".format(fp))

        vehicleinfo.append((lanelane_network_id, obstacle_infos))

    return trajectories, vehicleinfo, lanelet_networks



def local_search(nodes, edges, mode):
    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n)
    for j in range(edges.size(-1)):
        src, des = edges[0][j].item(), edges[1][j].item()
        if g.has_node(src) and g.has_node(des):
            g.add_edge(src, des)
    #print(g.edges)
    cands = []
    for n in nodes:
        if len(cands) == 0:
            cands.append(n)
            continue
        else:
            if mode == "start":
                path_lengths = nx.multi_source_dijkstra_path_length(g, cands)
                if n in path_lengths:
                    continue
                else:
                    cands.append(n)

            else:
                path_lengths = nx.single_source_dijkstra_path_length(g, n)
                for s in cands:
                    if s in path_lengths:
                        break
                else:
                    cands.append(n)
    return cands

def dilated_nbrs(sender, receiver, num_nodes, num_scales):
    data = np.ones(len(sender), np.bool)
    csr = sparse.csr_matrix((data, (sender, receiver)), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, num_scales):
        mat = mat * mat

        nbr = dict()
        coo = mat.tocoo()
        new_sender = coo.row.astype(np.int64)
        new_receiver = coo.col.astype(np.int64)
        nbrs.append([new_sender,new_receiver])
    return nbrs

def make_grids(scenario, ego_id, startframe, grid_length, max_disp_front=55, max_disp_rear=30, max_radius=55, cross_dist=6.0, num_scales=6):
    ego_posG = scenario.obstacles[startframe][ego_id].initial_state.get_posG()
    ego_lanelet_id = scenario.obstacles[startframe][ego_id].initial_state.posF.ind[1]
    ego_lanelet = scenario.lanelet_network.find_lanelet_by_id(ego_lanelet_id)

    selected_lanelets = []
    for lanelet in scenario.lanelet_network.lanelets:
        if lanelet.lanelet_id == ego_lanelet_id:
            selected_lanelets.append(lanelet)
            continue
        if len(lanelet.successor)==0 and len(lanelet.predecessor)==0:
            # this is case for isolated lanelets
            continue
        s_pos_x, s_pos_y = lanelet.center_curve[0].pos.x, lanelet.center_curve[0].pos.y
        if (s_pos_x - ego_posG.x)**2+(s_pos_y - ego_posG.y)**2 <= max_radius**2:
            selected_lanelets.append(lanelet)
            continue
        e_pos_x, e_pos_y = lanelet.center_curve[-1].pos.x, lanelet.center_curve[-1].pos.y
        if (e_pos_x - ego_posG.x)**2+(e_pos_y - ego_posG.y)**2 <= max_radius**2:
            selected_lanelets.append(lanelet)
            continue
        for curvept in lanelet.center_curve[1:-1]:
            if (curvept.pos.x - ego_posG.x)**2+(curvept.pos.y - ego_posG.y)**2 <= max_radius**2:
                selected_lanelets.append(lanelet)
                break

    ctrs = []
    vecs = []
    pris = []
    lrdists = []

    suc_edges, suc_edges = {}, {}
    pre_edges, pre_edges = {}, {}
    lane_ids = [lanelet.lanelet_id for lanelet in selected_lanelets]
    grids = []
    for lanelet in selected_lanelets:
        start_curvePt = lanelet.center_curve[0]
        start_curveInd = CurveIndex(0, 0.0)
        nodes = []

        lanelet_leftmost = lanelet
        while lanelet_leftmost.adj_left is not None and lanelet_leftmost.adj_left_same_direction:
            lanelet_leftmost = scenario.lanelet_network.find_lanelet_by_id(lanelet_leftmost.adj_left)
        lanelet_rightmost = lanelet
        while lanelet_rightmost.adj_right is not None and lanelet_rightmost.adj_right_same_direction:
            lanelet_rightmost = scenario.lanelet_network.find_lanelet_by_id(lanelet_rightmost.adj_right)


        if start_curvePt.s + grid_length*1.5 > lanelet.center_curve[-1].s:
            # at least make two nodes from a single lanelet
            ds = (lanelet.center_curve[-1].s - start_curvePt.s) * 0.5
            center_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, ds*0.5)
            center_curvePt = lanelet.get_curvePt_by_curveid(center_curveInd)
            end_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, ds)
            end_curvePt = lanelet.get_curvePt_by_curveid(end_curveInd)
            nodes.append([(start_curvePt,start_curveInd),
                      (center_curvePt,center_curveInd),
                      (end_curvePt, end_curveInd)])

            start_curvePt = end_curvePt
            start_curveInd = end_curveInd
            center_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, ds*0.5)
            center_curvePt = lanelet.get_curvePt_by_curveid(center_curveInd)
            end_curveInd = CurveIndex(len(lanelet.center_curve)-1, 1.0)
            end_curvePt = lanelet.center_curve[-1]
            nodes.append([(start_curvePt,start_curveInd),
                      (center_curvePt,center_curveInd),
                      (end_curvePt, end_curveInd)])

            #continue
        else:
            while start_curvePt.s + grid_length <= lanelet.center_curve[-1].s:
                center_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, grid_length*0.5)
                center_curvePt = lanelet.get_curvePt_by_curveid(center_curveInd)
                end_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, grid_length)
                end_curvePt = lanelet.get_curvePt_by_curveid(end_curveInd)
                nodes.append([(start_curvePt,start_curveInd),
                              (center_curvePt,center_curveInd),
                              (end_curvePt, end_curveInd)])

                start_curvePt = end_curvePt
                start_curveInd = end_curveInd
                if lanelet.center_curve[-1].s - start_curvePt.s < 0.5*grid_length:
                    start_curvePt, start_curveInd = nodes[-1][0]
                    ds = lanelet.center_curve[-1].s - start_curvePt.s
                    center_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, ds*0.5)
                    center_curvePt = lanelet.get_curvePt_by_curveid(center_curveInd)
                    end_curveInd = CurveIndex(len(lanelet.center_curve)-1, 1.0)
                    end_curvePt = lanelet.center_curve[-1]
                    nodes[-1] = [(start_curvePt,start_curveInd),
                              (center_curvePt,center_curveInd),
                              (end_curvePt, end_curveInd)]
                    break
                elif lanelet.center_curve[-1].s - start_curvePt.s >= 0.5*grid_length and \
                      lanelet.center_curve[-1].s - start_curvePt.s < grid_length:
                    ds = lanelet.center_curve[-1].s - start_curvePt.s
                    center_curveInd, _ = get_curve_index(start_curveInd, lanelet.center_curve, ds*0.5)
                    center_curvePt = lanelet.get_curvePt_by_curveid(center_curveInd)
                    end_curveInd = CurveIndex(len(lanelet.center_curve)-1, 1.0)
                    end_curvePt = lanelet.center_curve[-1]
                    nodes.append([(start_curvePt,start_curveInd),
                              (center_curvePt,center_curveInd),
                              (end_curvePt, end_curveInd)])
                    break

        for n in nodes:
            grids.append(Grid(len(grids)))
            grids[-1].add_pos(n[1][0].pos)
            grids[-1].add_ind((n[1][1], lanelet.lanelet_id))

        ctr = []
        vec = []
        origin_ctr = []
        for n in nodes:
            sta_proj = n[0][0].pos.inertial2body(ego_posG)
            cen_proj = n[1][0].pos.inertial2body(ego_posG)
            las_proj = n[2][0].pos.inertial2body(ego_posG)
            ctr.append([cen_proj.x, cen_proj.y])
            vec.append([las_proj.x-sta_proj.x, las_proj.y-sta_proj.y])
            origin_ctr.append([n[1][0].pos.x, n[1][0].pos.y])

        ctrs.append(np.array(ctr))

        vecs.append(np.array(vec))

        lrdists.append(np.concatenate([lanelet_leftmost.distance_line2line(np.array(origin_ctr), line="left"),
                                 lanelet_rightmost.distance_line2line(np.array(origin_ctr), line="right")]
                                 ,-1))
        #ctrs.append(np.array([[n[1][0].pos.x, n[1][0].pos.y] for n in nodes]))
        #vecs.append(np.array([[n[2][0].pos.x-n[0][0].pos.x, n[2][0].pos.y-n[0][0].pos.y] for n in nodes]))
        if LaneletType.ACCESS_RAMP in lanelet.lanelet_type:
            pris.append(np.array([[1,0] for _ in range(len(nodes))]))
        elif LaneletType.EXIT_RAMP in lanelet.lanelet_type:
            pris.append(np.array([[0,1] for _ in range(len(nodes))]))
        else:
            pris.append(np.array([[1,1] for _ in range(len(nodes))]))


    node_idcs = []
    count = 0
    node2lane = {}
    lane2node = {lanelet.lanelet_id:[] for lanelet in selected_lanelets}
    for lanelet, ctr in zip(selected_lanelets, ctrs):
        node_idcs.append(range(count, count + len(ctr)))
        lane2node[lanelet.lanelet_id] = range(count, count + len(ctr))
        for idx in node_idcs[-1]:
            node2lane[idx] = lanelet.lanelet_id
        count += len(ctr)
    num_nodes = count

    pre_sender, pre_receiver, suc_sender, suc_receiver = [], [], [], []
    for i, lane in enumerate(selected_lanelets):
        idcs = node_idcs[i]
        pre_sender += idcs[1:]
        pre_receiver += idcs[:-1]
        if len(lane.predecessor) > 0:
            for nbr_id in lane.predecessor:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    pre_sender.append(idcs[0])
                    pre_receiver.append(node_idcs[j][-1])


        suc_sender += idcs[:-1]
        suc_receiver += idcs[1:]

        if len(lane.successor) > 0:
            for nbr_id in lane.successor:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    suc_sender.append(idcs[-1])
                    suc_receiver.append(node_idcs[j][0])

    suc_edges[0] = torch.tensor([suc_sender,suc_receiver], dtype=torch.long)
    pre_edges[0] = torch.tensor([pre_sender,pre_receiver], dtype=torch.long)

    i = 1
    for edges in dilated_nbrs(suc_sender, suc_receiver, num_nodes, num_scales):
        suc_edges[i] = torch.tensor(edges, dtype=torch.long)
        i += 1
    i = 1
    for edges in dilated_nbrs(pre_sender, pre_receiver, num_nodes, num_scales):
        pre_edges[i] = torch.tensor(edges, dtype=torch.long)
        i += 1

    ctrs = torch.tensor(np.concatenate(ctrs, 0), dtype=torch.float)
    vecs = torch.tensor(np.concatenate(vecs, 0), dtype=torch.float)
    pris = torch.tensor(np.concatenate(pris, 0), dtype=torch.float)

    lrdists = torch.tensor(np.concatenate(lrdists, 0), dtype=torch.float)

    lane_idcs = []
    for i, idcs in enumerate(node_idcs):
        lane_idcs.append(i * np.ones(len(idcs), np.int64))
    lane_idcs = np.concatenate(lane_idcs, 0)

    pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
    for i, lane in enumerate(selected_lanelets):
        nbr_ids = lane.predecessor
        for nbr_id in nbr_ids:
            if nbr_id in lane_ids:
                j = lane_ids.index(nbr_id)
                pre_pairs.append([i, j])

        nbr_ids = lane.successor
        for nbr_id in nbr_ids:
            if nbr_id in lane_ids:
                j = lane_ids.index(nbr_id)
                suc_pairs.append([i, j])

        nbr_id = lane.adj_left
        if nbr_id is not None and lane.adj_left_same_direction:
            if nbr_id in lane_ids:
                j = lane_ids.index(nbr_id)
                left_pairs.append([i, j])

        nbr_id = lane.adj_right
        if nbr_id is not None and lane.adj_right_same_direction:
            if nbr_id in lane_ids:
                j = lane_ids.index(nbr_id)
                right_pairs.append([i, j])

    pre_pairs = torch.tensor(pre_pairs, dtype=torch.long)
    suc_pairs = torch.tensor(suc_pairs, dtype=torch.long)
    left_pairs = torch.tensor(left_pairs, dtype=torch.long)
    right_pairs = torch.tensor(right_pairs, dtype=torch.long)

    num_lanes = len(selected_lanelets)
    dist = ctrs.unsqueeze(1) - ctrs.unsqueeze(0)
    dist = torch.sqrt((dist ** 2).sum(2))
    hi = torch.arange(num_nodes).long().to(dist.device).view(-1, 1).repeat(1, num_nodes).view(-1)
    wi = torch.arange(num_nodes).long().to(dist.device).view(1, -1).repeat(num_nodes, 1).view(-1)
    row_idcs = torch.arange(num_nodes).long().to(dist.device)

    pre = pre_pairs.new().float().resize_(num_lanes, num_lanes).zero_()
    pre[pre_pairs[:, 0], pre_pairs[:, 1]] = 1
    suc = suc_pairs.new().float().resize_(num_lanes, num_lanes).zero_()
    suc[suc_pairs[:, 0], suc_pairs[:, 1]] = 1

    pairs = left_pairs
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        left_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        left_dist[hi[mask], wi[mask]] = 1e6

        min_dist, min_idcs = left_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = vecs[ui]
        f2 = vecs[vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        left_edges = torch.stack([ui, vi])
    else:
        left_edges = torch.tensor([np.zeros(0, np.int16), np.zeros(0, np.int16)], dtype=torch.long)

    pairs = right_pairs
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        right_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        right_dist[hi[mask], wi[mask]] = 1e6

        min_dist, min_idcs = right_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = vecs[ui]
        f2 = vecs[vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        right_edges = torch.stack([ui, vi])
    else:
        right_edges = torch.tensor([np.zeros(0, np.int16), np.zeros(0, np.int16)], dtype=torch.long)

    return grids, ctrs,  vecs, pris, lrdists, suc_edges, pre_edges, left_edges, right_edges, node2lane, lane2node

def generate_frenet_path(curve, s0, vs0, d0, vd0, sT, vsT, dT, vdT, T, DT):
    path = calc_frenet_path(s0, vs0, d0, vd0, sT, vsT, dT, vdT, T, DT)
    #path = calc_global_path(path, curve)
    return path

def calc_frenet_path(s0, vs0, d0, vd0, sT, vsT, dT, vdT, T, DT):
    fp = FrenetPath()
    lat = QuinticPolynomial(d0, vd0, 0.0, dT, vdT, 0.0, T)
    lon = QuinticPolynomial(s0, vs0, 0.0, sT, vsT, 0.0, T)

    fp.t = [t for t in np.arange(0.0, T, DT)]
    fp.d = [lat.calc_point(t) for t in fp.t]
    fp.s = [lon.calc_point(t) for t in fp.t]

class FrenetPath:
    def __init__(self):
        self.t = []
        self.d = []
        self.s = []


class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # calc coefficient of quintic polynomial
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt
