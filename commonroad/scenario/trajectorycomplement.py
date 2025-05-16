import numpy as np
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.scenario.lanelet import LineMarking, LaneletType
from commonroad.scenario.laneletcomplement import VecSE2, lerp_curve, lerp_angle, CurveIndex, get_curve_index, Grid
import math
from scipy.sparse import csr_matrix
import scipy


'''
 `lanelet_id`: road index
 `s`: distance along lane
 `d`: lane offset, positive is to left. zero point is the centerline of the lane.
 `phi`: lane relative heading
'''
class Frenet:
    def __init__(self, roadproj = None, lanelet_network = None, tuple_form = None):
        if tuple_form is not None:
            i,t,lid,s,d,phi = tuple_form
            self.ind = [CurveIndex(i, t), lid]
            self.s = s
            self.d = d
            self.phi = phi
        else:
            curve_ind = roadproj.curveproj.ind
            lanelet_id = roadproj.get_lanelet_id()
            self.ind = (curve_ind, lanelet_id)
            curve = lanelet_network.find_lanelet_by_id(lanelet_id).center_curve
            self.s = lerp_curve(curve[curve_ind.i], curve[curve_ind.i+1], curve_ind.t).s
            self.d = roadproj.curveproj.d
            #self.phi = roadproj.curveproj.phi
            self.phi = self._mod2pi(roadproj.curveproj.phi)

    def _mod2pi(self, x):
        if x > math.pi:
            return self._mod2pi(x-2*math.pi)
        if x <-math.pi:
            return self._mod2pi(x+2*math.pi)
        return x

    def get_posG_from_posF(self, lanelet_network):
        curve_ind, lanelet_id = self.ind

        curve = lanelet_network.find_lanelet_by_id(lanelet_id).center_curve

        footpoint = lerp_curve(curve[curve_ind.i], curve[curve_ind.i+1], curve_ind.t)

        th = footpoint.pos.th + 0.5*np.pi
        posG = VecSE2(footpoint.pos.x + self.d*np.cos(th), footpoint.pos.y + self.d*np.sin(th), footpoint.pos.th + self.phi)

        return posG

    def move_along(self, lanelet_network, delta_s, delta_d, new_phi, target_lanelet=None):
        curve_ind, lanelet_id = self.ind
        lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
        curve = lanelet.center_curve
        curvePt = lerp_curve(curve[curve_ind.i], curve[curve_ind.i+1], curve_ind.t)
        result_posF_adjust = result_posF = None
        #print("gaga ?", curvePt.s + delta_s)
        if curvePt.s + delta_s < -1e-4:#0.0:
            dist_s = [0.0]
            cands = [lanelet_id]
            while len(cands) > 0:
                current_cand = cands.pop(0)
                current_dist_s = dist_s.pop(0)
                current_lane = lanelet_network.find_lanelet_by_id(current_cand)
                if current_cand == target_lanelet:
                    new_delta_s =  current_dist_s + curvePt.s + delta_s
                    current_curve = current_lane.center_curve
                    final_ind, new_s = get_curve_index(CurveIndex(0,0.0), current_curve, new_delta_s)
                    final_lid = current_cand
                    infos = (final_ind.i, final_ind.t, final_lid, new_s, self.d + delta_d, new_phi)
                    result_posF = Frenet(None, None, infos)
                    break
                else:
                    if curvePt.s + delta_s + current_dist_s < 0.0:
                        for pred in current_lane.predecessor:
                            predecessor_lane = lanelet_network.find_lanelet_by_id(pred)
                            cands.append(pred)
                            dist_s.append(current_dist_s + predecessor_lane.center_curve[-1].s)
                    else:
                        new_delta_s = current_dist_s + curvePt.s + delta_s
                        current_curve = current_lane.center_curve
                        final_ind, new_s = get_curve_index(CurveIndex(0,0.0), current_curve, new_delta_s)
                        final_lid = current_cand
                        infos = (final_ind.i, final_ind.t, final_lid, new_s, self.d + delta_d, new_phi)
                        new_result_posF = Frenet(None, None, infos)
                        if target_lanelet is not None:
                            new_result_posG = new_result_posF.get_posG_from_posF(lanelet_network)
                            new_result_posF_adjust = Frenet(new_result_posG.projF(lanelet_network, [target_lanelet]), lanelet_network)
                            if result_posF_adjust is None:
                                result_posF = new_result_posF
                                result_posF_adjust = new_result_posF_adjust
                            elif target_lanelet is not None and np.abs(new_result_posF_adjust.d) < np.abs(result_posF_adjust.d):
                                result_posF = new_result_posF
                                result_posF_adjust = new_result_posF_adjust
                        else:
                            result_posF = new_result_posF

        elif curvePt.s + delta_s > curve[-1].s:
            dist_s = [0.0]
            cands = [lanelet_id]
            while len(cands) > 0:
                current_cand = cands.pop(0)
                current_dist_s = dist_s.pop(0)
                current_lane = lanelet_network.find_lanelet_by_id(current_cand)
                if current_cand == target_lanelet:
                    new_delta_s = curvePt.s + delta_s - current_dist_s
                    current_curve = current_lane.center_curve
                    final_ind, new_s = get_curve_index(CurveIndex(0,0.0), current_curve, new_delta_s)
                    final_lid = current_cand
                    infos = (final_ind.i, final_ind.t, final_lid, new_s, self.d + delta_d, new_phi)
                    result_posF = Frenet(None, None, infos)
                    break
                else:
                    if curvePt.s + delta_s  > current_dist_s + current_lane.center_curve[-1].s:
                        for succ in current_lane.successor:
                            successor_lane = lanelet_network.find_lanelet_by_id(succ)
                            cands.append(succ)
                            dist_s.append(current_dist_s + current_lane.center_curve[-1].s)
                    else:
                        new_delta_s = curvePt.s + delta_s - current_dist_s
                        current_curve = current_lane.center_curve
                        final_ind, new_s = get_curve_index(CurveIndex(0,0.0), current_curve, new_delta_s)
                        final_lid = current_cand
                        infos = (final_ind.i, final_ind.t, final_lid, new_s, self.d + delta_d, new_phi)
                        new_result_posF = Frenet(None, None, infos)
                        if target_lanelet is not None:
                            new_result_posG = new_result_posF.get_posG_from_posF(lanelet_network)
                            new_result_posF_adjust = Frenet(new_result_posG.projF(lanelet_network, [target_lanelet]), lanelet_network)
                            if result_posF_adjust is None:
                                result_posF = new_result_posF
                                result_posF_adjust = new_result_posF_adjust
                            elif target_lanelet is not None and np.abs(new_result_posF_adjust.d) < np.abs(result_posF_adjust.d):
                                result_posF = new_result_posF
                                result_posF_adjust = new_result_posF_adjust
                        else:
                            result_posF = new_result_posF

            #current_s = curvePt.s + delta_s
            #while len(lanelet.successor) > 0:
            #    new_curve = lanelet_network.find_lanelet_by_id(lanelet.successor[0]).center_curve
            #    s_all = new_curve[-1].s
            #    if current_s - lanelet.center_curve[-1].s > s_all:
                    # next
            #        current_s -= lanelet.center_curve[-1].s
            #        lanelet = lanelet_network.find_lanelet_by_id(lanelet.successor[0])
            #    else:
            #        new_delta_s = current_s - lanelet.center_curve[-1].s
            #        new_ind = 0
            #        final_ind, new_s = get_curve_index(CurveIndex(new_ind, 0.0), new_curve, new_delta_s)
            #        final_lid = lanelet.successor[0]
            #        break
            #if len(lanelet.successor) == 0:
                # no successor
            #    final_ind = CurveIndex(len(lanelet.center_curve)-2, 1.0)
            #    final_lid = lanelet.lanelet_id
            #    new_s = lanelet.center_curve[-1].s
        else:
            #print("gaga")
            final_ind, new_s = get_curve_index(curve_ind, curve, delta_s)
            final_lid = lanelet_id
            infos = (final_ind.i, final_ind.t, final_lid, new_s, self.d + delta_d, new_phi)
            result_posF = Frenet(None, None, infos)
            result_lane = lanelet_network.find_lanelet_by_id(result_posF.ind[1])

        #infos = (final_ind.i, final_ind.t, final_lid, new_s, self.d + delta_d, new_phi)
        #result_posF = Frenet(None, None, infos)
        result_lane = lanelet_network.find_lanelet_by_id(result_posF.ind[1])
        if target_lanelet == lanelet.adj_left or target_lanelet == lanelet.adj_right and target_lanelet is not None:
            result_posG = result_posF.get_posG_from_posF(lanelet_network)
            result_posF = Frenet(result_posG.projF(lanelet_network, [target_lanelet, result_lane.lanelet_id]), lanelet_network)
        elif result_lane.adj_left is not None:
            result_posG = result_posF.get_posG_from_posF(lanelet_network)
            result_posF = Frenet(result_posG.projF(lanelet_network, [result_lane.adj_left, result_lane.lanelet_id]), lanelet_network)
        elif result_lane.adj_right is not None:
            result_posG = result_posF.get_posG_from_posF(lanelet_network)
            result_posF = Frenet(result_posG.projF(lanelet_network, [result_lane.adj_right, result_lane.lanelet_id]), lanelet_network)

        #result_posG = result_posF.get_posG_from_posF(lanelet_network)
        #result_posF = Frenet(result_posG.projF(lanelet_network, [lanelet.lanelet_id, lanelet.adj_left, lanelet.adj_right]), lanelet_network)
        #if result_posF.d >= 3.0 *0.5 and result_lane.adj_left is not None:
        #    result_posG = result_posF.get_posG_from_posF(lanelet_network)
        #    result_posF = Frenet(result_posG.projF(lanelet_network, [result_posF.ind[1], result_lane.adj_left]), lanelet_network)
        #elif result_posF.d <= -3.0 *0.5 and result_lane.adj_right is not None:
        #    result_posG = result_posF.get_posG_from_posF(lanelet_network)
        #    result_posF = Frenet(result_posG.projF(lanelet_network, [result_posF.ind[1], result_lane.adj_right]), lanelet_network)

        return result_posF

def posG(s:State):
    return VecSE2(s.position[0], s.position[1], s.orientation)

def move_along_curve(curve_ind, curve, delta_s, next_d, include_curvature=False):
    next_ind, next_s = get_curve_index(curve_ind, curve, delta_s)
    footpoint = lerp_curve(curve[next_ind.i], curve[next_ind.i+1], next_ind.t)
    th = footpoint.pos.th + 0.5*np.pi
    next_posG = VecSE2(footpoint.pos.x + next_d*np.cos(th), footpoint.pos.y + next_d*np.sin(th), footpoint.pos.th + 0.0)
    if include_curvature:
        return next_ind, next_posG, footpoint.k
    return next_ind, next_posG

class FrenetState(State):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.posG = VecSE2(self.position[0], self.position[1], self.orientation)
        self.posF = None
    def get_posG(self):
        if self.posG is None:
            self.posG = VecSE2(self.position[0], self.position[1], self.orientation)
        return self.posG
    def set_posF(self, lanelet_network, clues=None):
        posG = self.get_posG()
        self.posF = Frenet(posG.projF(lanelet_network, clues), lanelet_network)
    def get_posG_from_posF(self, lanelet_network, posF=None):
        if posF is None:
            if self.posF is None:
                print("posF not set")
                self.set_posF(lanelet_network)
            posF = self.posF

        curve_ind, lanelet_id = posF.ind

        curve = lanelet_network.find_lanelet_by_id(lanelet_id).center_curve

        footpoint = lerp_curve(curve[curve_ind.i], curve[curve_ind.i+1], curve_ind.t)

        th = footpoint.pos.th + 0.5*np.pi
        posG = VecSE2(footpoint.pos.x + posF.d*np.cos(th), footpoint.pos.y + posF.d*np.sin(th), footpoint.pos.th + posF.phi)

        return posG
    def get_footpoint(self):
        posG = self.get_posG()
        return VecSE2(posG.x+self.posF.d*np.cos(self.posG.th-self.posF.phi-np.pi*0.5), posG.y+self.posF.d*np.sin(self.posG.th-self.posF.phi-np.pi*0.5), posG.th)

    def moveXY(self, lanelet_network, delta_x, delta_y, vx, vy, delta_step, target_lanelets):
        s, c = np.sin(self.posG.th), np.cos(self.posG.th)
        new_x = c*delta_x - s*delta_y + self.posG.x
        new_y = s*delta_x + c*delta_y + self.posG.y
        new_th = np.arctan2(vy, vx) + self.posG.th
        new_state = FrenetState(position=[new_x, new_y], orientation=new_th, velocity=np.sqrt(vx**2+vy**2), time_step=self.time_step+delta_step)
        new_state.set_posF(lanelet_network, target_lanelets)
        return new_state


def make_grids2(scenario, ego_id, startframe, grid_length, max_disp_front=55, max_disp_rear=30, max_radius=55):
    ego_posG = scenario.obstacles[startframe][ego_id].initial_state.get_posG()
    ego_posF = scenario.obstacles[startframe][ego_id].initial_state.posF
    ego_lanelet_id = scenario.obstacles[startframe][ego_id].initial_state.posF.ind[1]
    ego_lanelet = scenario.lanelet_network.find_lanelet_by_id(ego_lanelet_id)
    initial_nodes = [ego_posF.ind]
    if ego_lanelet.adj_left is not None:
        left_ego_posF = Frenet(ego_posG.projF(scenario.lanelet_network, [ego_lanelet.adj_left]), scenario.lanelet_network)
        left_lanelet = scenario.lanelet_network.find_lanelet_by_id(ego_lanelet.adj_left)
        initial_nodes.append(left_ego_posF.ind)
        if left_lanelet.adj_left is not None:
            ll_ego_posF = Frenet(ego_posG.projF(scenario.lanelet_network, [left_lanelet.adj_left]), scenario.lanelet_network)
            initial_nodes.append(ll_ego_posF.ind)
    if ego_lanelet.adj_right is not None:
        right_ego_posF = Frenet(ego_posG.projF(scenario.lanelet_network, [ego_lanelet.adj_right]), scenario.lanelet_network)
        right_lanelet = scenario.lanelet_network.find_lanelet_by_id(ego_lanelet.adj_right)
        initial_nodes.append(right_ego_posF.ind)
        if right_lanelet.adj_right is not None:
            rr_ego_posF = Frenet(ego_posG.projF(scenario.lanelet_network, [right_lanelet.adj_right]), scenario.lanelet_network)
            initial_nodes.append(rr_ego_posF.ind)
    nodes, edges = span_tree(ego_posG, initial_nodes, scenario.lanelet_network, max_disp_front, max_disp_rear, max_radius, grid_length)
    adj_p, adj_c = {n:[] for n in nodes}, {n:[] for n in nodes}
    for s,r in edges:
        adj_c[s].append(r)
        adj_p[r].append(s)
    new_nodes = []
    for n in nodes:
        if len(adj_c[n])+len(adj_p[n]) != 0:
            new_nodes.append(n)
    nodes = new_nodes
    new_nodes = {}
    new_nodes_idx = {}
    combined = {}
    combine = {}
    count = 0
    for i in range(len(nodes)):
        if nodes[i] in combined:
            continue
        iPt,_,_ = get_curvePt_from_index(scenario.lanelet_network, nodes[i])
        x, y, th = iPt.pos.x, iPt.pos.y, iPt.pos.th
        for j in range(len(nodes)):
            if j > i:
                jPt,_,_ = get_curvePt_from_index(scenario.lanelet_network, nodes[j])
                if (iPt.pos.x-jPt.pos.x)**2+(iPt.pos.y-jPt.pos.y)**2 <= 4:
                    combined[nodes[j]] = nodes[i]
                    combine[nodes[i]] = nodes[j]
                    x, y = 0.5*(iPt.pos.x+jPt.pos.x), 0.5*(iPt.pos.y+jPt.pos.y)
                    #th = 0.5*(iPt.pos.th+jPt.pos.th)
                    th = lerp_angle(iPt.pos.th, jPt.pos.th, 0.5)
                    break
        new_nodes[nodes[i]] = VecSE2(x, y, th)
        new_nodes_idx[nodes[i]] = count
        count += 1

    new_edges = []
    row_col_dict = {}
    row = []
    col = []
    adj_c = {n:[] for n in new_nodes}
    adj_p = {n:[] for n in new_nodes}
    adj_l = {n:[] for n in new_nodes}
    adj_r = {n:[] for n in new_nodes}
    adj_f = {n:[] for n in new_nodes}

    for s, r in edges:
        if s in combined:
            s = combined[s]

        if r in combined:
            r = combined[r]
        if (s, r) in row_col_dict:
            continue
        else:
            new_edges.append((s, r))
            row.append(new_nodes_idx[s])
            col.append(new_nodes_idx[r])
            row_col_dict[(s,r)] = len(row)-1
            row.append(new_nodes_idx[r])
            col.append(new_nodes_idx[s])
            row_col_dict[(r,s)] = len(row)-1
            adj_c[s].append(r)
            adj_p[r].append(s)
    val = np.zeros(len(row))
    for ii, n in enumerate(new_nodes):
        front_x = []
        posN = new_nodes[n]
        laneN = scenario.lanelet_network.find_lanelet_by_id(n[1])
        for v in adj_c[n]:
            posV = new_nodes[v].inertial2body(posN)
            front_x.append(posV.x)
        sorted_idx = np.argsort(front_x)
        for idx, v in enumerate(adj_c[n]):
            if sorted_idx[idx] == 0:
                val[row_col_dict[(n,v)]] = 4
            else:
                val[row_col_dict[(n,v)]] = 2

        rear_x = []
        for u in adj_p[n]:
            posU = new_nodes[u].inertial2body(posN)
            rear_x.append(posU.x)
        sorted_idx = np.argsort(rear_x)
        for idx, u in enumerate(adj_p[n]):
            if sorted_idx[idx] == 0:
                val[row_col_dict[(n, u)]] = 3
            else:
                val[row_col_dict[(n, u)]] = 1
        for jj, m in enumerate(new_nodes):
            if jj <= ii:
                continue
            posM = new_nodes[m]
            laneM = scenario.lanelet_network.find_lanelet_by_id(m[1])
            if laneN.adj_left == m[1]:
                if (posN.x-posM.x)**2+(posN.y-posM.y)**2 <= 30:
                    adj_l[n].append(m)
                    adj_r[m].append(n)
            if laneN.adj_right == m[1]:
                if (posN.x-posM.x)**2+(posN.y-posM.y)**2 <= 30:
                    adj_r[n].append(m)
                    adj_l[m].append(n)
            intersect = [lanelet_id for lanelet_id in laneN.successor if lanelet_id in laneM.successor]
            if len(intersect) != 0:
                if (posN.x-posM.x)**2+(posN.y-posM.y)**2 <= 30:
                    adj_f[m].append(n)
                    adj_f[n].append(m)
    W = csr_matrix((val,(row,col)), shape=(len(new_nodes),len(new_nodes)))
    if len(new_nodes) == 0:
        print("No nodes", ego_id, startframe, nodes)
    graphs, parents = coarsen(W)
    clusters = parents[2][parents[1][parents[0]]]

    grids = [Grid(i) for i in range(max(clusters)+1)]
    for n in new_nodes:
        nidx = new_nodes_idx[n]
        cluster_id = clusters[nidx]
        grids[cluster_id].add_pos(new_nodes[n])
        grids[cluster_id].add_ind(n)
        if n in combine:
            grids[cluster_id].add_ind(combine[n])
        for v in adj_c[n]:
            vidx = new_nodes_idx[v]
            vcluster_id = clusters[vidx]
            if vcluster_id != cluster_id:
                grids[cluster_id].add_front(vcluster_id)
        for u in adj_p[n]:
            uidx = new_nodes_idx[u]
            ucluster_id = clusters[uidx]
            if ucluster_id != cluster_id:
                grids[cluster_id].add_back(ucluster_id)
        for l in adj_l[n]:
            lidx = new_nodes_idx[l]
            lcluster_id = clusters[lidx]
            if lcluster_id != cluster_id:
                grids[cluster_id].add_left(lcluster_id)
        for r in adj_r[n]:
            ridx = new_nodes_idx[r]
            rcluster_id = clusters[ridx]
            if rcluster_id != cluster_id:
                grids[cluster_id].add_right(rcluster_id)
        for f in adj_f[n]:
            fidx = new_nodes_idx[f]
            fcluster_id = clusters[fidx]
            if fcluster_id != cluster_id:
                grids[cluster_id].add_conflict(fcluster_id)
    return new_nodes, new_nodes_idx, (adj_c,adj_p,adj_l,adj_r,adj_f), clusters, grids

def coarsen(W, levels=3):
    N, N = W.shape
    parents, graphs = [], []
    degree = W.toarray().sum(axis=1)
    rid = np.argsort(-degree)
    graphs.append(W)
    supernode_size = np.ones(N)
    for _ in range(levels):
        dd = degree
        idx_row, idx_col, val = scipy.sparse.find(W)
        perm = np.argsort(idx_row)
        rr = idx_row[perm]
        cc = idx_col[perm]
        vv = val[perm]
        cluster_id = metis_one_level(rr, cc, vv, rid, dd, N)
        parents.append(cluster_id)
        child_rr = cluster_id[rr]
        child_cc = cluster_id[cc]
        child_vv = vv
        try:
            child_N = cluster_id.max() + 1
        except:
            print(graphs)
            print(parents)
            raise
        W = csr_matrix((child_vv,(child_rr,child_cc)), shape=(child_N,child_N))
        W.eliminate_zeros()
        graphs.append(W)
        if len(cluster_id) == 1:
            return graphs, parents
        degree = W.toarray().sum(axis=1)
        try:
            supernode_size = csr_matrix((supernode_size, (cluster_id,np.zeros(N))),
                                        shape=(child_N,1)).toarray().squeeze()
        except:
            print(supernode_size)
            print(child_N)
            print(cluster_id)
            print(N)
            print(parents)
            print(graphs)
            raise
        N = child_N
        rid = np.argsort(supernode_size)
    return graphs, parents

def metis_one_level(rr,cc,vv,rid,dd, N):
    E = rr.shape[0]
    marked = np.zeros(N, np.bool)
    rowstart = -1*np.ones(N, np.int32)
    rowlength = np.zeros(N, np.int32)
    cluster_id = np.zeros(N, np.int32)

    row_dict = {n:[] for n in range(N)}
    for ii in range(E):
        row_dict[rr[ii]].append(ii)

    clustercount = 0
    for ii in range(N):
        tid = rid[ii]
        if not marked[tid]:
            wmax = 0.0
            marked[tid] = True
            bestW = -1
            bestneighbor = -1
            for jj in row_dict[tid]:
                nid = cc[jj]
                if marked[nid]:
                    tval = 0.0
                else:
                    tval = vv[jj] * (1.0/(dd[tid]+0.01) + 1.0/(dd[nid]+0.01))
                if tval > wmax:
                    wmax = tval
                    bestneighbor = nid
                    bestW = vv[jj]
            cluster_id[tid] = clustercount

            if bestneighbor > -1:
                cluster_id[bestneighbor] = clustercount
                marked[bestneighbor] = True
                if bestW == 4 or bestW == 2:
                    for jj in row_dict[tid]:
                        nid = cc[jj]
                        if (vv[jj]==4 or vv[jj]==2) and nid!=bestneighbor:
                            vv[jj] = 0
                            for kk in row_dict[nid]:
                                if cc[kk] == tid:
                                    vv[kk] = 0
                elif bestW == 3 or bestW == 1:
                    for jj in row_dict[tid]:
                        nid = cc[jj]
                        if (vv[jj]==3 or vv[jj]==1) and nid!=bestneighbor:
                            vv[jj] = 0
                            for kk in row_dict[nid]:
                                if cc[kk] == tid:
                                    vv[kk] = 0
            clustercount += 1
    return cluster_id

def span_tree(ego_posG, nodes, lanelet_network, max_disp_front, max_disp_rear, max_radius, grid_length):
    visited_nodes = set()
    out_nodes = []
    out_edges = []
    #base_l = 4
    base_l = grid_length#12
    for start in nodes:
        if visited(start, visited_nodes,1,lanelet_network):
            continue
        visited_nodes.add(start)
        out_nodes.append(start)
        children = adj_nodes(start, visited_nodes, lanelet_network, base_l)
        stack = [(start, 0., children)]
        while stack:
            parent, s, children = stack[-1]
            try:
                child, d, new_nodes = next(children)
                if len(new_nodes) > 0:
                    nodes.extend(new_nodes)

                is_in_range = in_range(ego_posG,
                            parent,
                            child,
                            s+base_l*d, max_disp_front,
                            max_disp_rear,
                            max_radius,
                            lanelet_network)
                if is_in_range and not visited(child, visited_nodes,d,lanelet_network):
                    if d == 1:
                        out_edges.append((parent, child))
                    else:
                        out_edges.append((child, parent))
                    visited_nodes.add(child)
                    out_nodes.append(child)
                    stack.append((child, s+base_l*d, adj_nodes(child, visited_nodes, lanelet_network,base_l)))
            except StopIteration:
                stack.pop()
    return out_nodes, out_edges

def get_curvePt_from_index(G, ind):
    l = G.find_lanelet_by_id(ind[1])
    cur = l.center_curve
    return lerp_curve(cur[ind[0].i],cur[ind[0].i+1],ind[0].t), l, cur

def in_range(pos_C, u, v, s, mf, mr, R, G):
    curvePt, _, _ = get_curvePt_from_index(G, v)
    pos_V = curvePt.pos
    dist2 = (pos_V.x-pos_C.x)**2+(pos_V.y-pos_C.y)**2
    if dist2 > R**2:
        return False
    #if u[1] != v[1] and (s > mf or s < -mr):
    if s > mf or s < -mr:
        return False
    return True

def adj_nodes(n, vn, G, base_l):
    curvePt, lane, curve = get_curvePt_from_index(G, n)
    children = []
    directions = []
    new_nodes = []
    # forward
    fn = []
    if curvePt.s + base_l > curve[-1].s:
        new_s = curvePt.s+base_l-curve[-1].s
        for succ in lane.successor:
            successor_lane = G.find_lanelet_by_id(succ)
            new_index, _ = get_curve_index(CurveIndex(0, 0.0), successor_lane.center_curve, new_s)
            assert new_index.t <= 1.0 + 1e-3
            children.append((new_index, succ))
            directions.append(1)
            new_curvePt = lerp_curve(successor_lane.center_curve[new_index.i],
                                         successor_lane.center_curve[new_index.i+1], new_index.t)
            if lane.adj_left is None and successor_lane.adj_left is not None:
                left_posF = Frenet(new_curvePt.pos.projF(G, [successor_lane.adj_left]), G)
                fn.append(left_posF.ind)
                left_lanelet = G.find_lanelet_by_id(successor_lane.adj_left)
                if left_lanelet.adj_left is not None:
                    left_left_posF = Frenet(new_curvePt.pos.projF(G, [left_lanelet.adj_left]), G)
                    fn.append(left_left_posF.ind)
            if lane.adj_right is None and successor_lane.adj_right is not None:
                right_posF = Frenet(new_curvePt.pos.projF(G, [successor_lane.adj_right]), G)
                fn.append(right_posF.ind)
                right_lanelet = G.find_lanelet_by_id(successor_lane.adj_right)
                if right_lanelet.adj_right is not None:
                    right_right_posF = Frenet(new_curvePt.pos.projF(G, [right_lanelet.adj_right]), G)
                    fn.append(right_right_posF.ind)
            new_nodes.append(fn)
    else:
        new_index, _ = get_curve_index(n[0], curve, base_l)
        children.append((new_index, n[1]))
        directions.append(1)
        new_nodes.append(fn)
    # backward
    bn = []
    if curvePt.s - base_l < 0.0:
        for pred in lane.predecessor:
            predecessor_lane = G.find_lanelet_by_id(pred)
            new_s = curvePt.s+predecessor_lane.center_curve[-1].s-base_l
            new_index, _ = get_curve_index(CurveIndex(0, 0.0), predecessor_lane.center_curve, new_s)
            assert new_index.t >= -1e-3, "new_index {}, lane {}, cur_s {}".format(new_index, lane.lanelet_id, curvePt.s)
            children.append((new_index, pred))
            directions.append(-1)
            new_curvePt = lerp_curve(predecessor_lane.center_curve[new_index.i],
                                         predecessor_lane.center_curve[new_index.i+1], new_index.t)
            if lane.adj_left is None and predecessor_lane.adj_left is not None:
                left_posF = Frenet(new_curvePt.pos.projF(G, [predecessor_lane.adj_left]), G)
                bn.append(left_posF.ind)
                left_lanelet = G.find_lanelet_by_id(predecessor_lane.adj_left)
                if left_lanelet.adj_left is not None:
                    left_left_posF = Frenet(new_curvePt.pos.projF(G, [left_lanelet.adj_left]), G)
                    bn.append(left_left_posF.ind)
            if lane.adj_right is None and predecessor_lane.adj_right is not None:
                right_posF = Frenet(new_curvePt.pos.projF(G, [predecessor_lane.adj_right]), G)
                bn.append(right_posF.ind)
                right_lanelet = G.find_lanelet_by_id(predecessor_lane.adj_right)
                if right_lanelet.adj_right is not None:
                    right_right_posF = Frenet(new_curvePt.pos.projF(G, [right_lanelet.adj_right]), G)
                    bn.append(right_right_posF.ind)
            new_nodes.append(bn)
    else:
        new_index, _ = get_curve_index(n[0], curve, -base_l)
        children.append((new_index, n[1]))
        directions.append(-1)
        new_nodes.append(bn)
    return iter(zip(children, directions, new_nodes))

def visited(n, vn, d, G):
    same_id = []
    for v in vn:
        if v[0].i == n[0].i and np.abs(v[0].t-n[0].t)<0.1 and v[1] == n[1]:
            return True
        if v[1] == n[1]:
            same_id.append(v)
    if len(same_id)==0:
        return False
    if d == 1:
        for v in same_id:
            if v[0].i > n[0].i or (v[0].i == n[0].i and v[0].t > n[0].t):
                return True
        return False
    else:
        assert d==-1
        for v in same_id:
            if v[0].i < n[0].i or (v[0].i == n[0].i and v[0].t < n[0].t):
                return True
        return False
