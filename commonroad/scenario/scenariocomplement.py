import numpy as np
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectorycomplement import FrenetState, Frenet, get_curvePt_from_index
from commonroad.scenario.laneletcomplement import *
from commonroad.scenario.lanelet import LineMarking, LaneletType
from copy import deepcopy
# Items

LANECURVATURE = 1
LINE_MARKING_LEFT_DIST = 2
LINE_MARKING_RIGHT_DIST = 3
TIMEGAP = 4
TTC = 5
NEIGHBOR_FORE_ALONG_LANE = 6
NEIGHBOR_FORE_ALONG_LEFT_LANE = 7
NEIGHBOR_FORE_ALONG_RIGHT_LANE = 8
NEIGHBOR_REAR_ALONG_LANE = 9
NEIGHBOR_REAR_ALONG_LEFT_LANE = 10
NEIGHBOR_REAR_ALONG_RIGHT_LANE = 11
NEIGHBOR_FORE_REAR_LEFT_MID_RIGHT = 15
NEIGHBOR_LEFT_LANE = 16
NEIGHBOR_RIGHT_LANE = 17
IS_COLLIDING = 12
ROAD_DIST_LEFT = 13
ROAD_DIST_RIGHT = 14


class ScenarioWrapper:
    def __init__(self, dt, lanelet_network, lanelet_network_id, obstacles, collision_objects, id=0):
        self.dt = dt
        self.scenario_id = id
        self.lanelet_network = lanelet_network
        self.lanelet_network_id = lanelet_network_id
        self.obstacles = obstacles
        self.collision_objects = collision_objects

    def draw(self, renderer, draw_params = None,
                 call_stack = tuple()):
        renderer.draw_scenario(self, draw_params, call_stack)

    def commonroad_scenario_at_time_step_all(self, time_step):
        cr_scenario = Scenario(self.dt)
        cr_scenario.lanelet_network = self.lanelet_network
        obstacles = []
        for veh_id, obs in self.obstacles[time_step].items():
            cobs = deepcopy(obs)
            cobs.initial_state.time_step = 0
            obstacles.append(cobs)
        cr_scenario.add_objects(obstacles)
        return cr_scenario

    def commonroad_scenario_at_time_step(self, time_step, ego_vehids, neighbor_vehids):
        cr_scenario = Scenario(self.dt)
        cr_scenario.lanelet_network = self.lanelet_network
        obstacles = []
        ego_vehs = []
        nei_vehs = []
        for veh_id, obs in self.obstacles[time_step].items():
            cobs = deepcopy(obs)
            cobs.initial_state.time_step = 0
            if veh_id in ego_vehids:
                ego_vehs.append(cobs)
            elif veh_id in neighbor_vehids:
                nei_vehs.append(cobs)
            else:
                obstacles.append(cobs)
        cr_scenario.add_objects(obstacles)
        return cr_scenario, ego_vehs, nei_vehs

    def set_sensor_range(self, radius, front, rear, side):
        self.max_radius = radius
        self.max_front = front
        self.max_rear = rear
        self.max_side = side

    def add_grids(self, grids):
        self.grids = grids

    def set_route(self, route):
        self.route = route

    def locate_on_grid(self, time_step, vehicle_id, initial_cands=None):
        ego_vehicle = self.obstacles[time_step][vehicle_id]
        posF = ego_vehicle.initial_state.posF
        cands = []
        projections = {posF.ind[1]:(posF.s, posF.d)}
        best_dist2 = float('inf')
        best_projection = None
        for index, grid in enumerate(self.grids):
            # get the s and d
            if initial_cands is not None:
                if index not in initial_cands:
                    continue
            if grid.from_lanelet_id not in projections:
                projF = Frenet(ego_vehicle.initial_state.posG.projF(self.lanelet_network, [grid.from_lanelet_id]), self.lanelet_network)
                projections[grid.from_lanelet_id] = (projF.s, projF.d)
            if grid.center_lanelet_id not in projections:
                projF = Frenet(ego_vehicle.initial_state.posG.projF(self.lanelet_network, [grid.center_lanelet_id]), self.lanelet_network)
                projections[grid.center_lanelet_id] = (projF.s, projF.d)
            if grid.to_lanelet_id not in projections:
                projF = Frenet(ego_vehicle.initial_state.posG.projF(self.lanelet_network, [grid.to_lanelet_id]), self.lanelet_network)
                projections[grid.to_lanelet_id] = (projF.s, projF.d)

            if grid.is_in_grid(posF.ind[0], posF.ind[1]):
                cands.append(index)
        #print(time_step, vehicle_id, posF.ind[1], projections)
        #if time_step == 4923 and vehicle_id==127:
        #    print(posF.ind[1], posF.ind[0].i, posF.ind[0].t)
        if len(cands) > 4:
            print("there are {} grid for veh {} at {} {} {}".format(len(cands), vehicle_id, posF.ind[1], posF.ind[0].i, posF.ind[0].t))
            #for grid in self.grids:
            #    print(grid)
            #assert (len(cands) == 0 or len(cands) == 1 or len(cands) == 2)
        if len(cands) == 0:
            for index, grid in enumerate(self.grids):
                if initial_cands is not None:
                    if index not in initial_cands:
                        continue
                from_lanelet = self.lanelet_network.find_lanelet_by_id(grid.from_lanelet_id)
                from_curve = from_lanelet.center_curve
                from_curve_pt = lerp_curve(from_curve[grid.from_index.i], from_curve[grid.from_index.i+1], grid.from_index.t)
                new_s = projections[grid.from_lanelet_id][0]-from_curve_pt.s

                center_lanelet = self.lanelet_network.find_lanelet_by_id(grid.center_lanelet_id)
                center_curve = center_lanelet.center_curve
                center_curve_pt = lerp_curve(center_curve[grid.center_index.i], center_curve[grid.center_index.i+1], grid.center_index.t)
                c_dist2 = (center_curve_pt.s- projections[grid.center_lanelet_id][0])**2 + projections[grid.center_lanelet_id][1]**2
                if c_dist2 < best_dist2:
                    best_projection = (new_s, index)
                    best_dist2 = c_dist2
                to_lanelet = self.lanelet_network.find_lanelet_by_id(grid.to_lanelet_id)
                to_curve = to_lanelet.center_curve
                to_curve_pt = lerp_curve(to_curve[grid.to_index.i], to_curve[grid.to_index.i+1], grid.to_index.t)
                t_dist2 = (to_curve_pt.s- projections[grid.to_lanelet_id][0])**2 + projections[grid.to_lanelet_id][1]**2
                if t_dist2 < best_dist2:
                    best_projection = (new_s, index)
                    best_dist2 = t_dist2
                #print("grid {} c_dist2 {} t_dist2 {} cur_best {}".format(grid, c_dist2, t_dist2, self.grids[best_projection[1]]))
            return best_projection
        if len(cands) == 1:
            grid = self.grids[cands[0]]
            from_lanelet = self.lanelet_network.find_lanelet_by_id(grid.from_lanelet_id)
            from_curve = from_lanelet.center_curve
            from_curve_pt = lerp_curve(from_curve[grid.from_index.i], from_curve[grid.from_index.i+1], grid.from_index.t)
            if grid.from_lanelet_id == posF.ind[1]:
                new_s = posF.s - from_curve_pt.s
            else:
                new_s = posF.s + from_curve[-1].s - from_curve_pt.s
            assert new_s >= 0.0, "pos_s {} for veh {} at time {} and curve_pt s {} is negative laneletids from {} center {} to {} veh {} index i {} {} {} index t {} {} {}".format(posF.s,
                                                                                                           vehicle_id,
                                                                                                           time_step,
                                                                                                           from_curve_pt.s,
                                                                                                           grid.from_lanelet_id,
                                                                                                           grid.center_lanelet_id,
                                                                                                           grid.to_lanelet_id,
                                                                                                           posF.ind[1],
                                                                                                           grid.from_index.i,
                                                                                                           grid.center_index.i,
                                                                                                           grid.to_index.i,
                                                                                                           grid.from_index.t,
                                                                                                           grid.center_index.t,
                                                                                                           grid.to_index.t
                                                                                                           )
            return new_s, cands[0]
        else:
            best_from = self.grids[cands[0]].from_lanelet_id
            best_to = None
            best_to_dist = float("inf")
            # four cands
            for index in cands:
                grid = self.grids[index]
                t = 1
                while vehicle_id in self.obstacles[time_step-t]:
                    if self.obstacles[time_step-t][vehicle_id].initial_state.posF.ind[1] == grid.from_lanelet_id:
                        best_from = grid.from_lanelet_id
                        break
                    t += 1
                to_lanelet = self.lanelet_network.find_lanelet_by_id(grid.to_lanelet_id)
                to_curve = to_lanelet.center_curve
                to_curve_pt = lerp_curve(to_curve[grid.to_index.i], to_curve[grid.to_index.i+1], grid.to_index.t)
                distance2 = (to_curve_pt.pos.x-ego_vehicle.initial_state.posG.x)**2 + (to_curve_pt.pos.y-ego_vehicle.initial_state.posG.y)**2
                if distance2 < best_to_dist:
                    best_to = grid.to_lanelet_id
                    best_to_dist = distance2
            #print("best_from", best_from)
            #print("best_to", best_to)
            #print("cands", [self.grids[c] for c in cands])
            for index in cands:
                grid = self.grids[index]
                if grid.from_lanelet_id == best_from and grid.to_lanelet_id == best_to:
                    from_lanelet = self.lanelet_network.find_lanelet_by_id(grid.from_lanelet_id)
                    from_curve = from_lanelet.center_curve
                    from_curve_pt = lerp_curve(from_curve[grid.from_index.i], from_curve[grid.from_index.i+1], grid.from_index.t)
                    if grid.from_lanelet_id == posF.ind[1]:
                        new_s = posF.s - from_curve_pt.s
                    else:
                        new_s = posF.s + from_curve[-1].s - from_curve_pt.s
                    return new_s, index
            for index in cands:
                grid = self.grids[index]
                if grid.to_lanelet_id == best_to:
                    from_lanelet = self.lanelet_network.find_lanelet_by_id(grid.from_lanelet_id)
                    from_curve = from_lanelet.center_curve
                    from_curve_pt = lerp_curve(from_curve[grid.from_index.i], from_curve[grid.from_index.i+1], grid.from_index.t)
                    if grid.from_lanelet_id == posF.ind[1]:
                        new_s = posF.s - from_curve_pt.s
                    else:
                        new_s = posF.s + from_curve[-1].s - from_curve_pt.s
                    return new_s, index
        '''
        elif len(cands)==2:
            if self.grids[cands[0]].from_lanelet_id != self.grids[cands[1]].from_lanelet_id:
                if vehicle_id in self.obstacles[time_step-1]:
                    if self.obstacles[time_step-1][vehicle_id].initial_state.posF.ind[1] == self.grids[cands[1]].from_lanelet_id:
                        grid_index = cands[1]
                    else:
                        grid_index = cands[0]
                else:
                    grid_index = cands[0]
                grid = self.grids[grid_index]
                from_lanelet = self.lanelet_network.find_lanelet_by_id(grid.from_lanelet_id)
                from_curve = from_lanelet.center_curve
                from_curve_pt = lerp_curve(from_curve[grid.from_index.i], from_curve[grid.from_index.i+1], grid.from_index.t)
                if grid.from_lanelet_id == posF.ind[1]:
                    new_s = posF.s - from_curve_pt.s
                else:
                    new_s = posF.s + from_curve[-1].s - from_curve_pt.s
                assert new_s >= 0.0
                return new_s, grid_index
            else:
                grid1 = self.grids[cands[0]]
                to_lanelet1 = self.lanelet_network.find_lanelet_by_id(grid1.to_lanelet_id)
                to_curve1 = to_lanelet1.center_curve
                to_curve_pt1 = lerp_curve(to_curve1[grid1.to_index.i], to_curve1[grid1.to_index.i+1], grid1.to_index.t)
                distance1 = (to_curve_pt1.pos.x-ego_vehicle.initial_state.posG.x)**2 + (to_curve_pt1.pos.y-ego_vehicle.initial_state.posG.y)**2

                grid2 = self.grids[cands[1]]
                to_lanelet2 = self.lanelet_network.find_lanelet_by_id(grid2.to_lanelet_id)
                to_curve2 = to_lanelet2.center_curve
                to_curve_pt2 = lerp_curve(to_curve2[grid2.to_index.i], to_curve2[grid2.to_index.i+1], grid2.to_index.t)
                distance2 = (to_curve_pt2.pos.x-ego_vehicle.initial_state.posG.x)**2 + (to_curve_pt2.pos.y-ego_vehicle.initial_state.posG.y)**2
                if distance1 <= distance2:
                    from_lanelet1 = self.lanelet_network.find_lanelet_by_id(grid1.from_lanelet_id)
                    from_curve1 = from_lanelet1.center_curve
                    from_curve_pt1 = lerp_curve(from_curve1[grid1.from_index.i], from_curve1[grid1.from_index.i+1], grid1.from_index.t)
                    if grid1.from_lanelet_id == posF.ind[1]:
                        new_s = posF.s - from_curve_pt1.s
                    else:
                        new_s = posF.s + from_curve1[-1].s - from_curve_pt1.s

                    assert new_s >= 0.0
                    return new_s, cands[0]
                else:
                    from_lanelet2 = self.lanelet_network.find_lanelet_by_id(grid2.from_lanelet_id)
                    from_curve2 = from_lanelet2.center_curve
                    from_curve_pt2 = lerp_curve(from_curve2[grid2.from_index.i], from_curve2[grid2.from_index.i+1], grid2.from_index.t)
                    if grid2.from_lanelet_id == posF.ind[1]:
                        new_s = posF.s - from_curve_pt2.s
                    else:
                        new_s = posF.s + from_curve2[-1].s - from_curve_pt2.s
                    assert new_s >= 0.0
                    return new_s, cands[1]
        '''
    def locate_on_grid1(self, time_step, vehicle_id, initial_cands=None):
        ego_vehicle = self.obstacles[time_step][vehicle_id]
        ego_posG = ego_vehicle.initial_state.posG
        best_dist2 = float('inf')
        best_projection = None
        if initial_cands is None:
            initial_cands = np.arange(len(self.grids))
        for index in initial_cands:
            grid = self.grids[index]
            best_point_dist2 = float('inf')
            for pos in grid.pos_list:
                dist2 = (pos.x-ego_posG.x)**2 + (pos.y-ego_posG.y)**2
                if dist2 < best_point_dist2:
                    best_point_dist2 = dist2
            if best_point_dist2 < best_dist2:
                best_dist2 = best_point_dist2
                best_projection = index
        return best_projection
    def _get_cr_obstacles(time_step):
        obstacles = []
        for veh_id, obs in self.obstacles[time_step].items():
            obstacles.append(obs)
        return obstacles
    def get_attribute(self, item, veh_id, time_step):
        if item == LANECURVATURE:
            return self.get_lane_curvature(veh_id, time_step)
        if item == LINE_MARKING_LEFT_DIST:
            return self.get_lane_marking_left_distance(veh_id, time_step)
        if item == LINE_MARKING_RIGHT_DIST:
            return self.get_lane_marking_right_distance(veh_id, time_step)
        if item == TIMEGAP:
            return self.get_time_gap(veh_id, time_step)
        if item == TTC:
            return self.get_inverse_time_to_collision(veh_id, time_step)
        if item == NEIGHBOR_FORE_ALONG_LANE:
            return self.get_front_neighbor_along_lane(veh_id, time_step, max_radius=self.max_radius, front=self.max_front, side=self.max_side,rear=self.max_rear)
        if item == NEIGHBOR_FORE_ALONG_LEFT_LANE:
            return self.get_front_neighbor_along_left_lane(veh_id, time_step, max_radius=self.max_radius, front=self.max_front, side=self.max_side,rear=self.max_rear)
        if item == NEIGHBOR_FORE_ALONG_RIGHT_LANE:
            return self.get_front_neighbor_along_right_lane(veh_id, time_step, max_radius=self.max_radius, front=self.max_front, side=self.max_side,rear=self.max_rear)
        if item == NEIGHBOR_LEFT_LANE:
            return self.get_left_neighbor(veh_id, time_step, max_radius=self.max_radius, front=self.max_front, side=self.max_side,rear=self.max_rear)
        if item == NEIGHBOR_RIGHT_LANE:
            return self.get_right_neighbor(veh_id, time_step, max_radius=self.max_radius, front=self.max_front, side=self.max_side,rear=self.max_rear)
        if item == NEIGHBOR_REAR_ALONG_LANE:
            return self.get_rear_neighbor_along_lane(veh_id, time_step, max_radius=self.max_radius, front=self.max_front, side=self.max_side,rear=self.max_rear)
        if item == NEIGHBOR_REAR_ALONG_LEFT_LANE:
            return self.get_rear_neighbor_along_left_lane(veh_id, time_step, max_radius=self.max_radius, front=self.max_front, side=self.max_side,rear=self.max_rear)
        if item == NEIGHBOR_REAR_ALONG_RIGHT_LANE:
            return self.get_rear_neighbor_along_right_lane(veh_id, time_step, max_radius=self.max_radius, front=self.max_front, side=self.max_side,rear=self.max_rear)
        if item == NEIGHBOR_FORE_REAR_LEFT_MID_RIGHT:
            return self.get_fore_rear_left_mid_right(veh_id, time_step, max_radius=self.max_radius, front=self.max_front, side=self.max_side,rear=self.max_rear)

        if item == IS_COLLIDING:
            return self.is_colliding(veh_id, time_step)
        if item == ROAD_DIST_LEFT:
            return self.get_left_road_distance(veh_id, time_step)
        if item == ROAD_DIST_RIGHT:
            return self.get_right_road_distance(veh_id, time_step)

    def get_vector_between_cars(self, vehicle_id, veh_id, startframe, along_lanelet=False):
        if vehicle_id not in self.obstacles[startframe] or veh_id not in self.obstacles[startframe]:
            return float("inf"), float("inf")
        ego = self.obstacles[startframe][vehicle_id]
        ego_lane_id = ego.initial_state.posF.ind[1]
        ego_lane = self.lanelet_network.find_lanelet_by_id(ego_lane_id)
        veh = self.obstacles[startframe][veh_id]
        if along_lanelet:
            delta_d = delta_s = delta_th = delta_vel_s = delta_vel_d = None
            veh_lane_id = veh.initial_state.posF.ind[1]
            veh_lane = self.lanelet_network.find_lanelet_by_id(ego_lane_id)
            if ego_lane_id == veh_lane_id:
                delta_d = veh.initial_state.posF.d - ego.initial_state.posF.d
                delta_s = veh.initial_state.posF.s - ego.initial_state.posF.s
                delta_th = veh.initial_state.posF.phi - ego.initial_state.posF.phi
                delta_vel_s = veh.initial_state.velocity*np.cos(veh.initial_state.posF.phi) - ego.initial_state.velocity*np.cos(ego.initial_state.posF.phi)
                delta_vel_d = veh.initial_state.velocity*np.sin(veh.initial_state.posF.phi) - ego.initial_state.velocity*np.sin(ego.initial_state.posF.phi)
            else:
                posG = veh.initial_state.get_posG()
                posF_adjust = Frenet(posG.projF(self.lanelet_network, [ego_lane_id]), self.lanelet_network)
                max_ind = len(ego_lane.center_curve)-2
                if posF_adjust.ind[0].i == 0 and posF_adjust.ind[0].t == 0.0 and len(ego_lane.predecessor) > 0:
                    # rematch
                    dist_s = [0.0]
                    cands = [ego_lane_id]
                    while len(cands) > 0:
                        current_cand = cands.pop(0)
                        current_dist_s = dist_s.pop(0)
                        if current_cand == veh_lane_id:
                            posF_adjust = veh.initial_state.posF
                            delta_d = posF_adjust.d - ego.initial_state.posF.d
                            delta_s = posF_adjust.s -(ego.initial_state.posF.s + current_dist_s)
                            delta_th = posF_adjust.phi - ego.initial_state.posF.phi
                            delta_vel_s = veh.initial_state.velocity*np.cos(posF_adjust.phi) - ego.initial_state.velocity*np.cos(ego.initial_state.posF.phi)
                            delta_vel_d = veh.initial_state.velocity*np.sin(posF_adjust.phi) - ego.initial_state.velocity*np.sin(ego.initial_state.posF.phi)
                            break
                        else:
                            current_lane = self.lanelet_network.find_lanelet_by_id(current_cand)
                            posF_adjust = Frenet(posG.projF(self.lanelet_network, [current_cand]), self.lanelet_network)
                            if posF_adjust.ind[0].i == 0 and posF_adjust.ind[0].t == 0.0:
                                for pred in current_lane.predecessor:
                                    predecessor_lane = self.lanelet_network.find_lanelet_by_id(pred)
                                    cands.append(pred)
                                    dist_s.append(current_dist_s+predecessor_lane.center_curve[-1].s)
                            else:
                                if delta_d is None:
                                    delta_d = posF_adjust.d - ego.initial_state.posF.d
                                    delta_s = posF_adjust.s - (ego.initial_state.posF.s + current_dist_s)
                                    delta_th = posF_adjust.phi - ego.initial_state.posF.phi
                                    delta_vel_s = veh.initial_state.velocity*np.cos(posF_adjust.phi) - ego.initial_state.velocity*np.cos(ego.initial_state.posF.phi)
                                    delta_vel_d = veh.initial_state.velocity*np.sin(posF_adjust.phi) - ego.initial_state.velocity*np.sin(ego.initial_state.posF.phi)
                                elif np.abs(posF_adjust.d - ego.initial_state.posF.d) < np.abs(delta_d):
                                    delta_d = posF_adjust.d - ego.initial_state.posF.d
                                    delta_s = posF_adjust.s - (ego.initial_state.posF.s + current_dist_s)
                                    delta_th = posF_adjust.phi - ego.initial_state.posF.phi
                                    delta_vel_s = veh.initial_state.velocity*np.cos(posF_adjust.phi) - ego.initial_state.velocity*np.cos(ego.initial_state.posF.phi)
                                    delta_vel_d = veh.initial_state.velocity*np.sin(posF_adjust.phi) - ego.initial_state.velocity*np.sin(ego.initial_state.posF.phi)


                elif posF_adjust.ind[0].i >= max_ind and posF_adjust.ind[0].t == 1.0 and len(ego_lane.successor) > 0:
                    #rematch

                    dist_s = [0.0]
                    cands = [ego_lane_id]
                    while len(cands) > 0:
                        current_cand = cands.pop(0)
                        current_dist_s = dist_s.pop(0)
                        if current_cand == veh_lane_id:
                            posF_adjust = veh.initial_state.posF
                            delta_d = posF_adjust.d - ego.initial_state.posF.d
                            delta_s = posF_adjust.s + current_dist_s - ego.initial_state.posF.s
                            delta_th = posF_adjust.phi - ego.initial_state.posF.phi
                            delta_vel_s = veh.initial_state.velocity*np.cos(posF_adjust.phi) - ego.initial_state.velocity*np.cos(ego.initial_state.posF.phi)
                            delta_vel_d = veh.initial_state.velocity*np.sin(posF_adjust.phi) - ego.initial_state.velocity*np.sin(ego.initial_state.posF.phi)
                            break
                        else:
                            current_lane = self.lanelet_network.find_lanelet_by_id(current_cand)
                            posF_adjust = Frenet(posG.projF(self.lanelet_network, [current_cand]), self.lanelet_network)
                            if posF_adjust.ind[0].i >= len(current_lane.center_curve)-2 and posF_adjust.ind[0].t == 1.0:
                                for succ in current_lane.successor:
                                    successor_lane = self.lanelet_network.find_lanelet_by_id(succ)
                                    cands.append(succ)
                                    dist_s.append(current_dist_s+current_lane.center_curve[-1].s)
                            else:
                                if delta_d is None:
                                    delta_d = posF_adjust.d - ego.initial_state.posF.d
                                    delta_s = posF_adjust.s + current_dist_s - ego.initial_state.posF.s
                                    delta_th = posF_adjust.phi - ego.initial_state.posF.phi
                                    delta_vel_s = veh.initial_state.velocity*np.cos(posF_adjust.phi) - ego.initial_state.velocity*np.cos(ego.initial_state.posF.phi)
                                    delta_vel_d = veh.initial_state.velocity*np.sin(posF_adjust.phi) - ego.initial_state.velocity*np.sin(ego.initial_state.posF.phi)
                                elif np.abs(posF_adjust.d - ego.initial_state.posF.d) < np.abs(delta_d):
                                    delta_d = posF_adjust.d - ego.initial_state.posF.d
                                    delta_s = posF_adjust.s + current_dist_s - ego.initial_state.posF.s
                                    delta_th = posF_adjust.phi - ego.initial_state.posF.phi
                                    delta_vel_s = veh.initial_state.velocity*np.cos(posF_adjust.phi) - ego.initial_state.velocity*np.cos(ego.initial_state.posF.phi)
                                    delta_vel_d = veh.initial_state.velocity*np.sin(posF_adjust.phi) - ego.initial_state.velocity*np.sin(ego.initial_state.posF.phi)

                else:
                    delta_d = posF_adjust.d - ego.initial_state.posF.d
                    delta_s = posF_adjust.s - ego.initial_state.posF.s
                    delta_th = posF_adjust.phi - ego.initial_state.posF.phi
                    delta_vel_s = veh.initial_state.velocity*np.cos(posF_adjust.phi) - ego.initial_state.velocity*np.cos(ego.initial_state.posF.phi)
                    delta_vel_d = veh.initial_state.velocity*np.sin(posF_adjust.phi) - ego.initial_state.velocity*np.sin(ego.initial_state.posF.phi)
            return delta_s, delta_d, delta_th, delta_vel_s, delta_vel_d
        else:
            #TODO instant not along lanelet
            curve_ind, lanelet_id = ego.initial_state.posF.ind
            curve = ego_lane.center_curve
            instant_curvePt = lerp_curve(curve[curve_ind.i], curve[curve_ind.i+1], curve_ind.t)
            veh_posG = veh.initial_state.get_posG()
            ego_posG = ego.initial_state.get_posG()
            F = veh_posG.inertial2body(ego_posG)
            delta_s, delta_d, delta_th = F.x, F.y, F.th
            delta_vel_s = veh.initial_state.velocity*np.cos(delta_th) - ego.initial_state.velocity
            delta_vel_d = veh.initial_state.velocity*np.sin(delta_th)
            #delta_s, delta_d = veh_posG.projCurvePt(instant_curvePt)
            return delta_s, delta_d, delta_th, delta_vel_s, delta_vel_d


    def get_lane_curvature(self, veh_id, time_step):
        vehicle = self.obstacles[time_step][veh_id]
        curve_ind, lanelet_id = vehicle.initial_state.posF.ind
        curvept = self.lanelet_network.find_lanelet_by_id(lanelet_id).get_curvePt_by_curveid(curve_ind)
        val = curvept.k
        return val
    def get_lane_marking_left_distance(self, veh_id, time_step):
        vehicle = self.obstacles[time_step][veh_id]
        curve_ind, lanelet_id = vehicle.initial_state.posF.ind
        offset = vehicle.initial_state.posF.d
        width = 3.2
        val = width*0.5 - offset
        return val
    def get_lane_marking_right_distance(self, veh_id, time_step):
        vehicle = self.obstacles[time_step][veh_id]
        curve_ind, lanelet_id = vehicle.initial_state.posF.ind
        offset = vehicle.initial_state.posF.d
        width = 3.2
        val = width*0.5 + offset
        return val
    def get_time_gap(self, veh_id, time_step):
        front_neighbor, front_neighbor_delta_s,_ = self.get_front_neighbor_along_lane(veh_id, time_step, max_range=250.0, max_veh=1)

        vel = self.obstacles[time_step][veh_id].initial_state.velocity

        if vel <= 0.0 or front_neighbor[0] is None:
            return 20.0, 0.0
        else:
            #len_ego = self.obstacles[time_step][veh_id].obstacle_shape.length
            #len_oth = self.obstacles[time_step][front_neighbor].obstacle_shape.length
            delta_s = front_neighbor_delta_s[0] #- len_oth #TODO check this
            if delta_s > 0.0:
                return delta_s / vel, 1.0
            else:
                return 0.0, 1.0
    def get_inverse_time_to_collision(self, veh_id, time_step):
        front_neighbor, front_neighbor_delta_s, _ = self.get_front_neighbor_along_lane(veh_id, time_step, max_range=250.0, max_veh=1)
        if front_neighbor[0] is None:
            return 0.0, 0.0
        else:
            veh_fore = self.obstacles[time_step][veh_id]
            veh_rear = self.obstacles[time_step][front_neighbor[0]]

            #len_fore = veh_fore.obstacle_shape.length
            #len_rear = veh_rear.obstacle_shape.length
            delta_s = front_neighbor_delta_s[0]# - len_fore

            delta_v = veh_fore.initial_state.velocity - veh_rear.initial_state.velocity

            if delta_s < 0.0:
                return 20.0, 1.0
            elif delta_v > 0.0:
                return 0.0, 0.0
            else:
                f = -delta_v/delta_s
                return f, 1.0
    def is_colliding(self, ego_id, time_step):
        #ego_posG = self.obstacles[time_step][ego_id].initial_state.get_posG()
        ego = self.obstacles[time_step][ego_id]
        ego_r = np.sqrt(ego.obstacle_shape.length**2 + (0.5*ego.obstacle_shape.width)**2)

        for veh_id, veh in self.obstacles[time_step].items():
            if veh_id != ego_id:
                #veh_posG = veh.initial_state.get_posG()
                veh_r = np.sqrt(veh.obstacle_shape.length**2 + (0.5*veh.obstacle_shape.width)**2)
                delta_pos = ego.initial_state.position-veh.initial_state.position
                dist2 = delta_pos[0]**2 + delta_pos[1]**2
                if dist2 <= ego_r**2 + veh_r**2 + 2*ego_r*veh_r:
                    if self.collision_objects[time_step][ego_id].collide(self.collision_objects[time_step][veh_id]):
                        return True
        return False

    def get_left_road_distance(self, veh_id, time_step):
        # the distance to left road boundary
        veh = self.obstacles[time_step][veh_id]
        offset = veh.initial_state.posF.d
        footpoint = veh.initial_state.get_footpoint()
        lane = self.lanelet_network.find_lanelet_by_id(veh.initial_state.posF.ind[1])
        while lane.adj_left is not None:
            lane = self.lanelet_network.find_lanelet_by_id(lane.adj_left)
        roadproj = veh.initial_state.get_posG().projF(self.lanelet_network, [lane.lanelet_id])
        curvept = lane.get_curvePt_by_curveid(roadproj.curveproj.ind)
        return 3.2*0.5 + np.sqrt((curvept.pos.x-footpoint.x)**2+(curvept.pos.y-footpoint.y)**2) - offset

    def get_right_road_distance(self, veh_id, time_step):
        # the distance to right road boundary
        veh = self.obstacles[time_step][veh_id]
        offset = veh.initial_state.posF.d
        footpoint = veh.initial_state.get_footpoint()
        lane = self.lanelet_network.find_lanelet_by_id(veh.initial_state.posF.ind[1])
        while lane.adj_right is not None:
            lane = self.lanelet_network.find_lanelet_by_id(lane.adj_right)
        roadproj = veh.initial_state.get_posG().projF(self.lanelet_network, [lane.lanelet_id])
        curvept = lane.get_curvePt_by_curveid(roadproj.curveproj.ind)
        return 3.2*0.5 + np.sqrt((curvept.pos.x-footpoint.x)**2+(curvept.pos.y-footpoint.y)**2) + offset

    def get_left_neighbor(self, ego_id, startframe, max_radius=40, front=40, side=30,rear=30):
        ego_posG = self.obstacles[startframe][ego_id].initial_state.get_posG()
        ego_lanelet_id = self.obstacles[startframe][ego_id].initial_state.posF.ind[1]
        ego_lanelet = self.lanelet_network.find_lanelet_by_id(ego_lanelet_id)
        best_ids = []
        for veh_id, veh in self.obstacles[startframe].items():
            if veh_id != ego_id:
                veh_posG = self.obstacles[startframe][veh_id].initial_state.get_posG()
                distance2 = (veh_posG.x-ego_posG.x)**2 + (veh_posG.y-ego_posG.y)**2
                if distance2 < max_radius**2:
                    best_ids.append(veh_id)
        final_ids = []
        #delta_step = int(1/self.dt)\
        length= 5.0
        width = 4.0

        left_id = -100
        left_s = float("inf")
        left_d = -100
        left_vel_s = -100
        left_vel_d = -100

        for veh_id in best_ids:
            delta_s, delta_d, _, delta_vel_s, delta_vel_d = self.get_vector_between_cars(ego_id, veh_id, startframe, along_lanelet=False)
            if delta_s <= front and delta_s >= - rear:
                if delta_d > width/2 and delta_d <= side and delta_s < length/2 and delta_s > -length/2:
                     if np.abs(delta_s) < np.abs(left_s):
                         left_id = veh_id
                         left_s = delta_s
                         left_d = delta_d
                         left_vel_s = delta_vel_s
                         left_vel_d = delta_vel_d

        if left_id != -100:
            ret = (left_id, left_s, left_d, left_vel_s, left_vel_d)
        else:
            ret = (-100, -100, -100, -100, -100)
        return ret

    def get_right_neighbor(self, ego_id, startframe, max_radius=40, front=40, side=30,rear=30):
        ego_posG = self.obstacles[startframe][ego_id].initial_state.get_posG()
        ego_lanelet_id = self.obstacles[startframe][ego_id].initial_state.posF.ind[1]
        ego_lanelet = self.lanelet_network.find_lanelet_by_id(ego_lanelet_id)
        best_ids = []
        for veh_id, veh in self.obstacles[startframe].items():
            if veh_id != ego_id:
                veh_posG = self.obstacles[startframe][veh_id].initial_state.get_posG()
                distance2 = (veh_posG.x-ego_posG.x)**2 + (veh_posG.y-ego_posG.y)**2
                if distance2 < max_radius**2:
                    best_ids.append(veh_id)
        final_ids = []
        #delta_step = int(1/self.dt)\
        length= 5.0
        width = 4.0

        right_id = -100
        right_s = float("inf")
        right_d = -100
        right_vel_s = -100
        right_vel_d = -100

        for veh_id in best_ids:
            delta_s, delta_d, _, delta_vel_s, delta_vel_d = self.get_vector_between_cars(ego_id, veh_id, startframe, along_lanelet=False)
            if delta_s <= front and delta_s >= - rear:
                if delta_d < -width/2 and delta_d >= -side and delta_s < length/2 and delta_s > -length/2:
                     if np.abs(delta_s) < np.abs(right_s):
                         right_id = veh_id
                         right_s = delta_s
                         right_d = delta_d
                         right_vel_s = delta_vel_s
                         right_vel_d = delta_vel_d

        if right_id != -100:
            ret = (right_id, right_s, right_d, right_vel_s, right_vel_d)
        else:
            ret = (-100, -100, -100, -100, -100)
        return ret

    def get_rear_neighbor_along_right_lane(self, ego_id, startframe, max_radius=40, front=40, side=30,rear=30):
        ego_posG = self.obstacles[startframe][ego_id].initial_state.get_posG()
        ego_lanelet_id = self.obstacles[startframe][ego_id].initial_state.posF.ind[1]
        ego_lanelet = self.lanelet_network.find_lanelet_by_id(ego_lanelet_id)
        best_ids = []
        for veh_id, veh in self.obstacles[startframe].items():
            if veh_id != ego_id:
                veh_posG = self.obstacles[startframe][veh_id].initial_state.get_posG()
                distance2 = (veh_posG.x-ego_posG.x)**2 + (veh_posG.y-ego_posG.y)**2
                if distance2 < max_radius**2:
                    best_ids.append(veh_id)
        final_ids = []
        #delta_step = int(1/self.dt)\
        length= 5.0
        width = 4.0

        rear_right_id = -100
        rear_right_s = -float("inf")
        rear_right_d = -100
        rear_right_vel_s = -100
        rear_right_vel_d = -100

        for veh_id in best_ids:
            delta_s, delta_d, _, delta_vel_s, delta_vel_d = self.get_vector_between_cars(ego_id, veh_id, startframe, along_lanelet=False)
            if delta_s <= front and delta_s >= - rear:
                if delta_d < -width/2 and delta_d >= -side and delta_s < -length/2:
                     if delta_s > rear_right_s:
                         rear_right_id = veh_id
                         rear_right_s = delta_s
                         rear_right_d = delta_d
                         rear_right_vel_s = delta_vel_s
                         rear_right_vel_d = delta_vel_d

        if rear_right_id != -100:
            ret = (rear_right_id, rear_right_s, rear_right_d, rear_right_vel_s, rear_right_vel_d)
        else:
            ret = (-100, -100, -100, -100, -100)
        return ret


    def get_front_neighbor_along_right_lane(self, ego_id, startframe, max_radius=40, front=40, side=30,rear=30):
        ego_posG = self.obstacles[startframe][ego_id].initial_state.get_posG()
        ego_lanelet_id = self.obstacles[startframe][ego_id].initial_state.posF.ind[1]
        ego_lanelet = self.lanelet_network.find_lanelet_by_id(ego_lanelet_id)
        best_ids = []
        for veh_id, veh in self.obstacles[startframe].items():
            if veh_id != ego_id:
                veh_posG = self.obstacles[startframe][veh_id].initial_state.get_posG()
                distance2 = (veh_posG.x-ego_posG.x)**2 + (veh_posG.y-ego_posG.y)**2
                if distance2 < max_radius**2:
                    best_ids.append(veh_id)
        final_ids = []
        #delta_step = int(1/self.dt)
        length = 5.0
        width = 4.0

        front_right_id = -100
        front_right_s = float("inf")
        front_right_d = -100
        front_right_vel_s = -100
        front_right_vel_d = -100

        for veh_id in best_ids:
            delta_s, delta_d, _, delta_vel_s, delta_vel_d = self.get_vector_between_cars(ego_id, veh_id, startframe, along_lanelet=False)
            if delta_s <= front and delta_s >= - rear:
                if delta_d < -width/2 and delta_d >= -side and delta_s >= length/2:
                     if delta_s < front_right_s:
                         front_right_id = veh_id
                         front_right_s = delta_s
                         front_right_d = delta_d
                         front_right_vel_s = delta_vel_s
                         front_right_vel_d = delta_vel_d

        if front_right_id != -100:
            ret = (front_right_id, front_right_s, front_right_d, front_right_vel_s, front_right_vel_d)
        else:
            ret = (-100, -100, -100, -100, -100)
        return ret



    def get_rear_neighbor_along_left_lane(self, ego_id, startframe, max_radius=40, front=40, side=30,rear=30):
        ego_posG = self.obstacles[startframe][ego_id].initial_state.get_posG()
        ego_lanelet_id = self.obstacles[startframe][ego_id].initial_state.posF.ind[1]
        ego_lanelet = self.lanelet_network.find_lanelet_by_id(ego_lanelet_id)
        best_ids = []
        for veh_id, veh in self.obstacles[startframe].items():
            if veh_id != ego_id:
                veh_posG = self.obstacles[startframe][veh_id].initial_state.get_posG()
                distance2 = (veh_posG.x-ego_posG.x)**2 + (veh_posG.y-ego_posG.y)**2
                if distance2 < max_radius**2:
                    best_ids.append(veh_id)
        final_ids = []
        #delta_step = int(1/self.dt)
        length = 5.0
        width = 4.0

        rear_left_id = -100
        rear_left_s = -float("inf")
        rear_left_d = -100
        rear_left_vel_s = -100
        rear_left_vel_d = -100

        for veh_id in best_ids:
            delta_s, delta_d, _, delta_vel_s, delta_vel_d = self.get_vector_between_cars(ego_id, veh_id, startframe, along_lanelet=False)
            if delta_s <= front and delta_s >= - rear:
                if delta_d > width/2 and delta_d <= side and delta_s < -length/2:
                     if delta_s > rear_left_s:
                         rear_left_id = veh_id
                         rear_left_s = delta_s
                         rear_left_d = delta_d
                         rear_left_vel_s = delta_vel_s
                         rear_left_vel_d = delta_vel_d

        if rear_left_id != -100:
            ret = (rear_left_id, rear_left_s, rear_left_d, rear_left_vel_s, rear_left_vel_d)
        else:
            ret = (-100, -100, -100, -100, -100)
        return ret

    def get_front_neighbor_along_left_lane(self, ego_id, startframe, max_radius=40, front=40, side=30,rear=30):
        ego_posG = self.obstacles[startframe][ego_id].initial_state.get_posG()
        ego_lanelet_id = self.obstacles[startframe][ego_id].initial_state.posF.ind[1]
        ego_lanelet = self.lanelet_network.find_lanelet_by_id(ego_lanelet_id)
        best_ids = []
        for veh_id, veh in self.obstacles[startframe].items():
            if veh_id != ego_id:
                veh_posG = self.obstacles[startframe][veh_id].initial_state.get_posG()
                distance2 = (veh_posG.x-ego_posG.x)**2 + (veh_posG.y-ego_posG.y)**2
                if distance2 < max_radius**2:
                    best_ids.append(veh_id)
        final_ids = []
        #delta_step = int(1/self.dt)
        length = 5.0
        width = 4.0

        front_left_id = -100
        front_left_s = float("inf")
        front_left_d = -100
        front_left_vel_s = -100
        front_left_vel_d = -100

        for veh_id in best_ids:
            delta_s, delta_d, _, delta_vel_s, delta_vel_d = self.get_vector_between_cars(ego_id, veh_id, startframe, along_lanelet=False)
            if delta_s <= front and delta_s >= - rear:
                if delta_d > width/2 and delta_d <= side and delta_s >= length/2:
                     if delta_s < front_left_s:
                         front_left_id = veh_id
                         front_left_s = delta_s
                         front_left_d = delta_d
                         front_left_vel_s = delta_vel_s
                         front_left_vel_d = delta_vel_d
        if front_left_id != -100:
            ret = (front_left_id, front_left_s, front_left_d, front_left_vel_s, front_left_vel_d)
        else:
            ret = (-100, -100, -100, -100, -100)
        return ret

    def get_rear_neighbor_along_lane(self, ego_id, startframe, max_radius=40, front=40, side=30,rear=30):
        ego_posG = self.obstacles[startframe][ego_id].initial_state.get_posG()
        ego_lanelet_id = self.obstacles[startframe][ego_id].initial_state.posF.ind[1]
        ego_lanelet = self.lanelet_network.find_lanelet_by_id(ego_lanelet_id)
        best_ids = []
        for veh_id, veh in self.obstacles[startframe].items():
            if veh_id != ego_id:
                veh_posG = self.obstacles[startframe][veh_id].initial_state.get_posG()
                distance2 = (veh_posG.x-ego_posG.x)**2 + (veh_posG.y-ego_posG.y)**2
                if distance2 < max_radius**2:
                    best_ids.append(veh_id)
        final_ids = []
        #delta_step = int(1/self.dt)
        length = 5.0
        width = 4.0
        rear_id = -100
        rear_s = -float("inf")
        rear_d = -100
        rear_vel_s = -100
        rear_vel_d = -100

        for veh_id in best_ids:
            delta_s, delta_d, _, delta_vel_s, delta_vel_d  = self.get_vector_between_cars(ego_id, veh_id, startframe, along_lanelet=False)
            if delta_s <= front and delta_s >= - rear:
                if delta_d <= width/2 and delta_d >= -width/2 and delta_s < -length/2:
                     if delta_s > rear_s:
                         rear_id = veh_id
                         rear_s = delta_s
                         rear_d = delta_d
                         rear_vel_s = delta_vel_s
                         rear_vel_d = delta_vel_d

        if rear_id != -100:
            ret = (rear_id, rear_s, rear_d, rear_vel_s, rear_vel_d)
        else:
            ret = (-100, -100, -100, -100, -100)
        return ret

    def get_front_neighbor_along_lane(self, ego_id, startframe, max_radius=40, front=40, side=30,rear=30):
        ego_posG = self.obstacles[startframe][ego_id].initial_state.get_posG()
        ego_lanelet_id = self.obstacles[startframe][ego_id].initial_state.posF.ind[1]
        ego_lanelet = self.lanelet_network.find_lanelet_by_id(ego_lanelet_id)
        best_ids = []
        for veh_id, veh in self.obstacles[startframe].items():
            if veh_id != ego_id:
                veh_posG = self.obstacles[startframe][veh_id].initial_state.get_posG()
                distance2 = (veh_posG.x-ego_posG.x)**2 + (veh_posG.y-ego_posG.y)**2
                if distance2 < max_radius**2:
                    best_ids.append(veh_id)
        final_ids = []
        #delta_step = int(1/self.dt)
        length = 5.0
        width = 4.0
        front_id = -100
        front_s = float("inf")
        front_d = -100
        front_vel_s = -100
        front_vel_d = -100
        for veh_id in best_ids:
            delta_s, delta_d, _, delta_vel_s, delta_vel_d = self.get_vector_between_cars(ego_id, veh_id, startframe, along_lanelet=False)
            if delta_s <= front and delta_s >= - rear:
                if delta_d <= width/2 and delta_d >= -width/2 and delta_s >= length/2:
                     if delta_s < front_s:
                         front_id = veh_id
                         front_s = delta_s
                         front_d = delta_d
                         front_vel_s = delta_vel_s
                         front_vel_d = delta_vel_d
        if front_id != -100:
            ret = (front_id, front_s, front_d, front_vel_s, front_vel_d)
        else:
            ret = (-100,  -100, -100, -100, -100)
        return ret

    def is_in_entrances(self, veh, target_lanelet):
        # convert a vehicle at the entrance of target_lanelet to this target_lanelet
        veh_curve_ind, veh_lanelet_id = veh.initial_state.posF.ind
        veh_lanelet = self.lanelet_network.find_lanelet_by_id(veh_lanelet_id)
        if veh_lanelet_id in target_lanelet.predecessor:
            con = target_lanelet.predecessor_connections[veh_lanelet_id]
            t_x, t_y = target_lanelet.center_curve[0].pos.x, target_lanelet.center_curve[0].pos.y
            v_x, v_y = veh_lanelet.center_curve[-1].pos.x, target_lanelet.center_curve[-1].pos.y
            con_dist = np.sqrt((t_x-v_x)**2 + (t_y-v_y)**2)
            if veh_curve_ind.i >= con[1].i and veh_curve_ind.t == 1.0 and con_dist >= 0.0:#if np.abs(veh_curve_ind.i - con[0].i) <= 5:
                # need to project to check
                posG = veh.initial_state.get_posG()
                projF = posG.projF(self.lanelet_network, [target_lanelet.lanelet_id])
                if np.abs(projF.curveproj.d) <= 3.2*0.5:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    def is_in_exits(self, veh, target_lanelet):
        # convert a vehicle at the exit of target_lanelet to this target_lanelet
        veh_curve_ind, veh_lanelet_id = veh.initial_state.posF.ind
        veh_lanelet = self.lanelet_network.find_lanelet_by_id(veh_lanelet_id)
        if veh_lanelet_id in target_lanelet.successor:
            con = target_lanelet.successor_connections[veh_lanelet_id]
            t_x, t_y = target_lanelet.center_curve[-1].pos.x, target_lanelet.center_curve[-1].pos.y
            v_x, v_y = veh_lanelet.center_curve[0].pos.x, target_lanelet.center_curve[0].pos.y
            con_dist = np.sqrt((t_x-v_x)**2 + (t_y-v_y)**2)
            if veh_curve_ind.i == 0 and veh_curve_ind.t==0.0 and con_dist >=0.0:#np.abs(veh_curve_ind.i - con[0].i) <= 5:
                # need to project to check
                posG = veh.initial_state.get_posG()
                projF = posG.projF(self.lanelet_network, [target_lanelet.lanelet_id])
                if np.abs(projF.curveproj.d) <= 3.2*0.5:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    def get_targetpoint_delta(self, ref, veh, posF):
        if ref == "front_center":
            return 0.0
        else:
            return -veh.obstacle_shape.length*np.cos(posF.phi)

    def get_neighbors(self, ego_id, startframe, along_lanelet=False, max_radius=30, front=30, side=20, rear=25, max_vehicles=8):
    #TODO make the reference straight instead of along lanelet
        ego_posG = self.obstacles[startframe][ego_id].initial_state.get_posG()
        ego_lanelet_id = self.obstacles[startframe][ego_id].initial_state.posF.ind[1]
        ego_lanelet = self.lanelet_network.find_lanelet_by_id(ego_lanelet_id)
        best_ids = []
        best_dists = []
        for veh_id, veh in self.obstacles[startframe].items():
            if veh_id != ego_id:
                veh_posG = self.obstacles[startframe][veh_id].initial_state.get_posG()
                distance2 = (veh_posG.x-ego_posG.x)**2 + (veh_posG.y-ego_posG.y)**2
                is_in_grids = False
                for grid in self.grids:
                    if grid.in_grid(veh.initial_state.posF.ind, veh_posG):
                        is_in_grids = True
                        break
                if distance2 < max_radius**2 and is_in_grids:
                    best_ids.append(veh_id)
                    best_dists.append(distance2)
        final_ids = []
        final_dists = []
        #delta_step = int(1/self.dt)
        for veh_id, dist2 in zip(best_ids, best_dists):
            delta_s, delta_d, _, _, _ = self.get_vector_between_cars(ego_id, veh_id, startframe, along_lanelet)
            if delta_s <= front and delta_s >= - rear:
                if delta_d <= side and delta_d >= -side:
                    final_ids.append(veh_id)
                    final_dists.append(dist2)
        if len(final_ids) > max_vehicles:
            order = np.argsort(final_dists)
            final_ids = [final_ids[ii] for ii in order]
            final_ids = final_ids[:max_vehicles]
        return final_ids

    def get_fore_rear_left_mid_right(self, ego_id, startframe, max_radius=40, front=40, side=30,rear=30):
        ego_posG = self.obstacles[startframe][ego_id].initial_state.get_posG()
        ego_lanelet_id = self.obstacles[startframe][ego_id].initial_state.posF.ind[1]
        ego_lanelet = self.lanelet_network.find_lanelet_by_id(ego_lanelet_id)
        best_ids = []
        for veh_id, veh in self.obstacles[startframe].items():
            if veh_id != ego_id:
                veh_posG = self.obstacles[startframe][veh_id].initial_state.get_posG()
                distance2 = (veh_posG.x-ego_posG.x)**2 + (veh_posG.y-ego_posG.y)**2
                if distance2 < max_radius**2:
                    best_ids.append(veh_id)
        final_ids = []
        #delta_step = int(1/self.dt)
        width = 4.0

        front_id = -100
        front_s = float("inf")
        front_d = -100
        front_th = -100

        rear_id = -100
        rear_s = -float("inf")
        rear_d = -100
        rear_th = -100

        best_left_ids = [-100 for _ in range(3)]
        best_left_s = [float("inf") for _ in range(3)] # closest to furthest
        best_left_d = [-100 for _ in range(3)]
        best_left_th = [-100 for _ in range(3)]

        best_right_ids = [-100 for _ in range(3)]
        best_right_s = [float("inf") for _ in range(3)]
        best_right_d = [-100 for _ in range(3)]
        best_right_th = [-100 for _ in range(3)]

        for veh_id in best_ids:
            delta_s, delta_d, delta_th, _, _ = self.get_vector_between_cars(ego_id, veh_id, startframe, along_lanelet=False)
            if delta_s <= front and delta_s >= - rear:
                if delta_d <= width/2 and delta_d >= -width/2 and delta_s >= 0.0:
                     if delta_s < front_s:
                         front_id = veh_id
                         front_s = delta_s
                         front_d = delta_d
                         front_th = delta_th
                elif delta_d > width/2 and delta_d <= side:
                    if np.abs(delta_s) < np.abs(best_left_s[-1]):
                        for idx in range(len(best_left_s)):
                            if np.abs(delta_s) < np.abs(best_left_s[idx]):
                                best_left_s.insert(idx, delta_s)
                                best_left_s = best_left_s[:-1]
                                best_left_ids.insert(idx, veh_id)
                                best_left_ids = best_left_ids[:-1]
                                best_left_d.insert(idx, delta_d)
                                best_left_d = best_left_d[:-1]
                                best_left_th.insert(idx, delta_th)
                                best_left_th = best_left_th[:-1]
                                break

                elif delta_d < -width/2 and delta_d >= -side:
                    if np.abs(delta_s) < np.abs(best_right_s[-1]):
                        for idx in range(len(best_right_s)):
                            if np.abs(delta_s) < np.abs(best_right_s[idx]):
                                best_right_s.insert(idx, delta_s)
                                best_right_s = best_right_s[:-1]
                                best_right_ids.insert(idx, veh_id)
                                best_right_ids = best_right_ids[:-1]
                                best_right_d.insert(idx, delta_d)
                                best_right_d = best_right_d[:-1]
                                best_right_th.insert(idx, delta_th)
                                best_right_th = best_right_th[:-1]
                                break
                elif delta_d <= width/2 and delta_d >= -width/2 and delta_s < 0.0:

                    if np.abs(delta_s) < np.abs(rear_s):
                        rear_id = veh_id
                        rear_s = delta_s
                        rear_d = delta_d
                        rear_th = delta_th
        ret = {}
        if front_id != -100:
            delta_s_along_lane, delta_d_along_lane, _, _, _ = self.get_vector_between_cars(ego_id, front_id, startframe, along_lanelet=True)
            ret["front"] = (front_id, delta_s_along_lane, delta_d_along_lane, front_s, front_d, front_th)
        else:
            ret["front"] = (-100, -100, -100, -100, -100, -100)

        if rear_id != -100:
            delta_s_along_lane, delta_d_along_lane, _, _, _ = self.get_vector_between_cars(ego_id, rear_id, startframe, along_lanelet=True)
            ret["rear"] = (rear_id, delta_s_along_lane, delta_d_along_lane, rear_s, rear_d, rear_th)
        else:
            ret["rear"] = (-100, -100, -100, -100, -100, -100)

        best_left_s_along_lane = []
        best_left_d_along_lane = []
        for left_veh_id in best_left_ids:
            if left_veh_id != -100:
                delta_s_along_lane, delta_d_along_lane, _, _, _ = self.get_vector_between_cars(ego_id, left_veh_id, startframe, along_lanelet=True)
                best_left_s_along_lane.append(delta_s_along_lane)
                best_left_d_along_lane.append(delta_d_along_lane)
            else:
                best_left_s_along_lane.append(-100)
                best_left_d_along_lane.append(-100)
        ret["left"] = (best_left_ids, best_left_s_along_lane, best_left_d_along_lane, best_left_s, best_left_d, best_left_th)

        best_right_s_along_lane = []
        best_right_d_along_lane = []
        for right_veh_id in best_right_ids:
            if right_veh_id != -100:
                delta_s_along_lane, delta_d_along_lane, _, _, _ = self.get_vector_between_cars(ego_id, right_veh_id, startframe, along_lanelet=True)
                best_right_s_along_lane.append(delta_s_along_lane)
                best_right_d_along_lane.append(delta_d_along_lane)
            else:
                best_right_s_along_lane.append(-100)
                best_right_d_along_lane.append(-100)
        ret["right"] = (best_right_ids, best_right_s_along_lane, best_right_d_along_lane, best_right_s, best_right_d, best_right_th)
        return ret

    def get_grids(self, ego_id, startframe, grid_length, max_disp_front=55, max_disp_rear=30, max_radius=55):
        ego_posG = self.obstacles[startframe][ego_id].initial_state.get_posG()
        ego_posF = self.obstacles[startframe][ego_id].initial_state.posF
        ego_curve_index = self.obstacles[startframe][ego_id].initial_state.posF.ind[0]
        ego_lanelet_id = self.obstacles[startframe][ego_id].initial_state.posF.ind[1]
        ego_lanelet = self.lanelet_network.find_lanelet_by_id(ego_lanelet_id)

        # this is the center of everything
        cand_indexs = [(ego_curve_index, ego_lanelet_id)]
        future_indexs = []
        all_lanelets = set()
        all_lanelets.add(ego_lanelet_id)
        #print("gaga", ego_lanelet_id)
        #future_indexs_backward = [(ego_curve_index, ego_lanelet_id, 0.0)]
        if ego_lanelet.adj_left is not None:
            left_ego_posF = Frenet(ego_posG.projF(self.lanelet_network, [ego_lanelet.adj_left]), self.lanelet_network)
            left_lanelet = self.lanelet_network.find_lanelet_by_id(ego_lanelet.adj_left)
            future_indexs.append((left_ego_posF.ind[0], ego_lanelet.adj_left, 0.0, "forward"))
            future_indexs.append((left_ego_posF.ind[0], ego_lanelet.adj_left, 0.0, "backward"))
            all_lanelets.add(ego_lanelet.adj_left)
            #future_indexs_backward.append((ego_curve_index, ego_lanelet.adj_left, 0.0))
            cand_indexs.append((left_ego_posF.ind[0], ego_lanelet.adj_left))
            if left_lanelet.adj_left is not None:
                left_left_ego_posF = Frenet(ego_posG.projF(self.lanelet_network, [left_lanelet.adj_left]), self.lanelet_network)
                future_indexs.append((left_left_ego_posF.ind[0], left_lanelet.adj_left, 0.0, "forward"))
                future_indexs.append((left_left_ego_posF.ind[0], left_lanelet.adj_left, 0.0, "backward"))
                cand_indexs.append((left_left_ego_posF.ind[0], left_lanelet.adj_left))
                all_lanelets.add(left_lanelet.adj_left)

        if ego_lanelet.adj_right is not None:
            right_ego_posF = Frenet(ego_posG.projF(self.lanelet_network, [ego_lanelet.adj_right]), self.lanelet_network)
            right_lanelet = self.lanelet_network.find_lanelet_by_id(ego_lanelet.adj_right)
            future_indexs.append((right_ego_posF.ind[0], ego_lanelet.adj_right, 0.0, "backward"))
            future_indexs.append((right_ego_posF.ind[0], ego_lanelet.adj_right, 0.0, "forward"))

            all_lanelets.add(ego_lanelet.adj_right)
            #future_indexs_backward.append((ego_curve_index, ego_lanelet.adj_right, 0.0))
            cand_indexs.append((right_ego_posF.ind[0], ego_lanelet.adj_right))
            if right_lanelet.adj_right is not None:
                right_right_ego_posF = Frenet(ego_posG.projF(self.lanelet_network, [right_lanelet.adj_right]), self.lanelet_network)
                future_indexs.append((right_right_ego_posF.ind[0], right_lanelet.adj_right, 0.0, "backward"))
                future_indexs.append((right_right_ego_posF.ind[0], right_lanelet.adj_right, 0.0, "forward"))
                cand_indexs.append((right_right_ego_posF.ind[0], right_lanelet.adj_right))
                all_lanelets.add(right_lanelet.adj_right)


        future_indexs.append((ego_curve_index, ego_lanelet_id, 0.0, "backward"))
        future_indexs.append((ego_curve_index, ego_lanelet_id, 0.0, "forward"))

        # forward and backward
        while len(future_indexs) > 0:
            current_index, current_lanelet_id, current_disp, dir = future_indexs.pop(-1)
            current_lanelet = self.lanelet_network.find_lanelet_by_id(current_lanelet_id)
            current_curve = current_lanelet.center_curve
            #print(len(current_curve), current_index.i, current_lanelet_id, dir, current_disp)
            current_curvePt = lerp_curve(current_curve[current_index.i], current_curve[current_index.i+1], current_index.t)
            distance2 = (current_curvePt.pos.x-ego_posG.x)**2 + (current_curvePt.pos.y-ego_posG.y)**2
            if (dir == "forward" and current_disp > max_disp_front) or (dir == "backward" and current_disp > max_disp_rear) or distance2 > max_radius**2:
                #print("out", current_disp + grid_length > max_disp_front, current_disp + grid_length > max_disp_rear, distance2>max_radius**2)
                continue
            elif current_disp != 0.0:
                #print("in", dir, current_lanelet_id, current_index.i, current_disp, np.sqrt(distance2))
                cand_indexs.append((current_index, current_lanelet_id))
            if dir == "forward":
                if current_curvePt.s + grid_length > current_curve[-1].s:
                    ori_new_s = current_curvePt.s+grid_length-current_curve[-1].s
                    next_lanes = []
                    for succ in current_lanelet.successor:
                        successor_lane = self.lanelet_network.find_lanelet_by_id(succ)
                        new_index, _ = get_curve_index(CurveIndex(0, 0.0), successor_lane.center_curve, ori_new_s)
                        if new_index.t > 1.0:
                            for succ_succ in successor_lane.successor:
                                if succ_succ not in all_lanelets:
                                    next_lanes.append((succ_succ, ori_new_s-successor_lane.center_curve[-1].s, succ))
                        elif succ not in all_lanelets:
                            next_lanes.append((succ, ori_new_s, succ))
                    for succ, new_s, ori_succ in next_lanes:
                        successor_lane = self.lanelet_network.find_lanelet_by_id(succ)
                        all_lanelets.add(succ)
                        new_index, _ = get_curve_index(CurveIndex(0, 0.0), successor_lane.center_curve, new_s)
                        future_indexs.append((new_index, succ, current_disp+grid_length, "forward"))
                        if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                            print("1 {}".format(future_indexs[-1]))
                        if current_lanelet.adj_left is None and successor_lane.adj_left is not None and successor_lane.adj_left not in all_lanelets:
                            new_curvePt = lerp_curve(successor_lane.center_curve[new_index.i],
                                                     successor_lane.center_curve[new_index.i+1], new_index.t)
                            left_posF = Frenet(new_curvePt.pos.projF(self.lanelet_network, [successor_lane.adj_left]), self.lanelet_network)
                            future_indexs.append((left_posF.ind[0], successor_lane.adj_left, current_disp + grid_length, "forward"))
                            if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                print("2 {}".format(future_indexs[-1]))
                            future_indexs.append((left_posF.ind[0], successor_lane.adj_left, 0.0, "backward"))
                            all_lanelets.add(successor_lane.adj_left)
                            if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                print("3 {}".format(future_indexs[-1]))
                            left_lanelet = self.lanelet_network.find_lanelet_by_id(successor_lane.adj_left)
                            if left_lanelet.adj_left is not None and left_lanelet.adj_left not in all_lanelets:
                                all_lanelets.add(left_lanelet.adj_left)
                                left_left_posF = Frenet(new_curvePt.pos.projF(self.lanelet_network, [left_lanelet.adj_left]), self.lanelet_network)
                                future_indexs.append((left_left_posF.ind[0], left_lanelet.adj_left, current_disp + grid_length, "forward"))
                                if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                    print("4 {}".format(future_indexs[-1]))
                                future_indexs.append((left_left_posF.ind[0], left_lanelet.adj_left, 0.0, "backward"))
                                if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                    print("5 {}".format(future_indexs[-1]))

                        if current_lanelet.adj_right is None and successor_lane.adj_right is not None and successor_lane.adj_right not in all_lanelets:
                            new_curvePt = lerp_curve(successor_lane.center_curve[new_index.i],
                                                     successor_lane.center_curve[new_index.i+1], new_index.t)
                            right_posF = Frenet(new_curvePt.pos.projF(self.lanelet_network, [successor_lane.adj_right]), self.lanelet_network)
                            future_indexs.append((right_posF.ind[0], successor_lane.adj_right, current_disp + grid_length, "forward"))
                            if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                print("6 {}".format(future_indexs[-1]))
                            future_indexs.append((right_posF.ind[0], successor_lane.adj_right, 0.0, "backward"))
                            all_lanelets.add(successor_lane.adj_right)
                            if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                print("7 {}".format(future_indexs[-1]))
                            right_lanelet = self.lanelet_network.find_lanelet_by_id(successor_lane.adj_right)
                            if right_lanelet.adj_right is not None and right_lanelet.adj_right not in all_lanelets:
                                all_lanelets.add(right_lanelet.adj_right)
                                right_right_posF = Frenet(new_curvePt.pos.projF(self.lanelet_network, [right_lanelet.adj_right]), self.lanelet_network)
                                future_indexs.append((right_right_posF.ind[0], right_lanelet.adj_right, current_disp + grid_length, "forward"))
                                if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                    print("8 {}".format(future_indexs[-1]))
                                future_indexs.append((right_right_posF.ind[0], right_lanelet.adj_right, 0.0, "backward"))
                                if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                    print("9 {}".format(future_indexs[-1]))
                        if len(successor_lane.predecessor) > 1: # there exists a conflict lane
                            for pred in successor_lane.predecessor:
                                if (succ == ori_succ and pred != current_lanelet_id and pred not in all_lanelets) or (succ != ori_succ and pred != ori_succ and pred not in all_lanelets):
                                    conflict_lanelet = self.lanelet_network.find_lanelet_by_id(pred)
                                    conflict_new_s = conflict_lanelet.center_curve[-1].s+new_s-grid_length
                                    if conflict_new_s < 0:
                                        for con_pred in conflict_lanelet.predecessor:
                                            if con_pred not in all_lanelets:
                                                pred_conflict_lanelet = self.lanelet_network.find_lanelet_by_id(con_pred)
                                                pred_conflict_new_s = conflict_new_s + pred_conflict_lanelet.center_curve[-1].s
                                                conflict_new_index,_ = get_curve_index(CurveIndex(0,0.0), pred_conflict_lanelet.center_curve, pred_conflict_new_s)
                                                future_indexs.append((conflict_new_index, con_pred, 5., "backward"))
                                                if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                                    print("11.0 {}".format(future_indexs[-1]))
                                                all_lanelets.add(con_pred)
                                    else:
                                        conflict_new_index,_ = get_curve_index(CurveIndex(0,0.0), conflict_lanelet.center_curve, conflict_new_s)
                                        future_indexs.append((conflict_new_index, pred, 5., "backward"))
                                        all_lanelets.add(pred)
                                        if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                            print("10 {}".format(future_indexs[-1]))
                                            print(pred, ori_succ, succ, new_s, grid_length)
                        if succ != ori_succ:
                            ori_succ_lane = self.lanelet_network.find_lanelet_by_id(ori_succ)
                            if len(ori_succ_lane.predecessor) > 1:
                                for pred in ori_succ_lane.predecessor:
                                    if pred != current_lanelet_id and pred not in all_lanelets:
                                        conflict_lanelet = self.lanelet_network.find_lanelet_by_id(pred)
                                        conflict_new_s = np.maximum(conflict_lanelet.center_curve[-1].s+new_s+ori_succ_lane.center_curve[-1].s-grid_length, 0.0)
                                        conflict_new_index,_ = get_curve_index(CurveIndex(0,0.0), conflict_lanelet.center_curve, conflict_new_s)
                                        future_indexs.append((conflict_new_index, pred, 5., "backward"))
                                        all_lanelets.add(pred)



                else:
                    new_index, _ = get_curve_index(current_index, current_curve, grid_length)
                    future_indexs.append((new_index, current_lanelet_id, current_disp + grid_length, "forward"))
                    if future_indexs[-1][0].t >=1.0:
                        print("11 {}".format(future_indexs[-1]))
            else:
                if current_curvePt.s - grid_length < 0.0:
                    next_lanes = []
                    for pred in current_lanelet.predecessor:
                        predecessor_lane = self.lanelet_network.find_lanelet_by_id(pred)
                        if predecessor_lane.center_curve[-1].s + current_curvePt.s - grid_length < 0:
                            for pred_pred in predecessor_lane.predecessor:
                                if pred_pred not in all_lanelets:
                                    pred_pred_lane = self.lanelet_network.find_lanelet_by_id(pred_pred)
                                    next_lanes.append((pred_pred, pred_pred_lane.center_curve[-1].s+predecessor_lane.center_curve[-1].s+current_curvePt.s-grid_length, pred))
                        elif pred not in all_lanelets:
                            next_lanes.append((pred, predecessor_lane.center_curve[-1].s+current_curvePt.s-grid_length, pred))

                    for pred, new_s, ori_pred in next_lanes:
                        predecessor_lane = self.lanelet_network.find_lanelet_by_id(pred)
                        all_lanelets.add(pred)
                        #new_s = predecessor_lane.center_curve[-1].s + current_curvePt.s - grid_length
                        new_index, _ = get_curve_index(CurveIndex(0,0.0), predecessor_lane.center_curve, new_s)
                        future_indexs.append((new_index, pred, current_disp + grid_length, "backward"))
                        if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                            print("12 {}".format(future_indexs[-1]))
                        if current_lanelet.adj_left is None and predecessor_lane.adj_left is not None and predecessor_lane.adj_left not in all_lanelets:
                            new_curvePt = lerp_curve(predecessor_lane.center_curve[new_index.i],
                                                     predecessor_lane.center_curve[new_index.i+1], new_index.t)
                            left_posF = Frenet(new_curvePt.pos.projF(self.lanelet_network, [predecessor_lane.adj_left]), self.lanelet_network)
                            future_indexs.append((left_posF.ind[0], predecessor_lane.adj_left, current_disp + grid_length, "backward"))
                            if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                print("13 {}".format(future_indexs[-1]))
                            future_indexs.append((left_posF.ind[0], predecessor_lane.adj_left, 0.0, "forward"))
                            all_lanelets.add(predecessor_lane.adj_left)
                            if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                print("14 {}".format(future_indexs[-1]))
                            left_lanelet = self.lanelet_network.find_lanelet_by_id(predecessor_lane.adj_left)
                            if left_lanelet.adj_left is not None and left_lanelet.adj_left not in all_lanelets:
                                all_lanelets.add(left_lanelet.adj_left)
                                left_left_posF = Frenet(new_curvePt.pos.projF(self.lanelet_network, [left_lanelet.adj_left]), self.lanelet_network)
                                future_indexs.append((left_left_posF.ind[0], left_lanelet.adj_left, current_disp + grid_length, "backward"))
                                if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                    print("15 {}".format(future_indexs[-1]))
                                future_indexs.append((left_left_posF.ind[0], left_lanelet.adj_left, 0.0, "forward"))
                                if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                    print("16 {}".format(future_indexs[-1]))
                        if current_lanelet.adj_right is None and predecessor_lane.adj_right is not None and predecessor_lane.adj_right not in all_lanelets:
                            new_curvePt = lerp_curve(predecessor_lane.center_curve[new_index.i],
                                                     predecessor_lane.center_curve[new_index.i+1], new_index.t)
                            right_posF = Frenet(new_curvePt.pos.projF(self.lanelet_network, [predecessor_lane.adj_right]), self.lanelet_network)
                            future_indexs.append((right_posF.ind[0], predecessor_lane.adj_right, current_disp + grid_length, "backward"))
                            if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                print("17 {}".format(future_indexs[-1]))
                            future_indexs.append((right_posF.ind[0], predecessor_lane.adj_right, 0.0, "forward"))
                            all_lanelets.add(predecessor_lane.adj_right)
                            if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                print("18 {}".format(future_indexs[-1]))
                            right_lanelet = self.lanelet_network.find_lanelet_by_id(predecessor_lane.adj_right)
                            if right_lanelet.adj_right is not None and right_lanelet.adj_right not in all_lanelets:
                                all_lanelets.add(right_lanelet.adj_right)
                                right_right_posF = Frenet(new_curvePt.pos.projF(self.lanelet_network, [right_lanelet.adj_right]), self.lanelet_network)
                                future_indexs.append((right_right_posF.ind[0], right_lanelet.adj_right, current_disp + grid_length, "backward"))
                                if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                    print("19 {}".format(future_indexs[-1]))
                                future_indexs.append((right_right_posF.ind[0], right_lanelet.adj_right, 0.0, "forward"))
                                if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                    print("20 {}".format(future_indexs[-1]))
                        if len(predecessor_lane.successor) > 1: # conflict_lane
                            for succ in predecessor_lane.successor:
                                if pred == ori_pred:
                                    if succ != current_lanelet_id and succ not in all_lanelets:
                                        conflict_lanelet = self.lanelet_network.find_lanelet_by_id(succ)
                                        conflict_new_s = current_curvePt.s
                                        #TODO
                                        if conflict_new_s > conflict_lanelet.center_curve[-1].s:
                                            #assert len(conflict_lanelet.successor) == 1, "conflict {}".format(succ)
                                            for con_succ in conflict_lanelet.successor:
                                                if con_succ not in all_lanelets:
                                                    succ_conflict_new_s = conflict_new_s - conflict_lanelet.center_curve[-1].s
                                                    succ_conflict_lanelet = self.lanelet_network.find_lanelet_by_id(con_succ)
                                                    conflict_new_index,_ = get_curve_index(CurveIndex(0,0.0), succ_conflict_lanelet.center_curve, succ_conflict_new_s)
                                                    future_indexs.append((conflict_new_index, con_succ, 5., "forward"))
                                                    if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                                        print("21.0 {}".format(future_indexs[-1]))
                                                    all_lanelets.add(con_succ)

                                        else:
                                            conflict_new_index,_ = get_curve_index(CurveIndex(0,0.0), conflict_lanelet.center_curve, conflict_new_s)
                                            future_indexs.append((conflict_new_index, succ, 5., "forward"))
                                            if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                                print("21 {}".format(future_indexs[-1]))
                                            all_lanelets.add(succ)

                                else:
                                    if succ != ori_pred and succ not in all_lanelets:
                                        conflict_lanelet = self.lanelet_network.find_lanelet_by_id(succ)
                                        ori_pred_lane = self.lanelet_network.find_lanelet_by_id(ori_pred)
                                        conflict_new_s = current_curvePt.s + ori_pred_lane.center_curve[-1].s
                                        #TODO
                                        if conflict_new_s > conflict_lanelet.center_curve[-1].s:
                                            #print(conflict_new_s, conflict_lanelet.center_curve[-1].s)
                                            #assert len(conflict_lanelet.successor) == 1, "conflict {}".format(succ)
                                            for con_succ in conflict_lanelet.successor:
                                                if con_succ not in all_lanelets:
                                                    succ_conflict_new_s = conflict_new_s - conflict_lanelet.center_curve[-1].s
                                                    succ_conflict_lanelet = self.lanelet_network.find_lanelet_by_id(con_succ)
                                                    conflict_new_index,_ = get_curve_index(CurveIndex(0,0.0), succ_conflict_lanelet.center_curve, succ_conflict_new_s)
                                                    future_indexs.append((conflict_new_index, con_succ, 5., "forward"))
                                                    if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                                        print("22.0 {}".format(future_indexs[-1]))
                                                        print(conflict_new_s)
                                                    all_lanelets.add(con_succ)

                                        else:
                                            conflict_new_index,_ = get_curve_index(CurveIndex(0,0.0), conflict_lanelet.center_curve, conflict_new_s)
                                            future_indexs.append((conflict_new_index, succ, 5., "forward"))
                                            if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                                print("22 {}".format(future_indexs[-1]))
                                            all_lanelets.add(succ)
                                    #if current_lanelet.adj_left is None and conflict_lanelet.adj_left is not None:
                                    #    future_indexs.append((conflict_new_index, conflict_lanelet.adj_left, 10., "forward"))
                                    #if current_lanelet.adj_right is None and conflict_lanelet.adj_right is not None:
                                    #    future_indexs.append((conflict_new_index, conflict_lanelet.adj_right, 10., "forward"))
                        if pred != ori_pred:
                            ori_pred_lane = self.lanelet_network.find_lanelet_by_id(ori_pred)
                            if len(ori_pred_lane.successor) > 1:
                                for succ in ori_pred_lane.successor:
                                    if succ != current_lanelet_id and succ not in all_lanelets:
                                        conflict_lanelet = self.lanelet_network.find_lanelet_by_id(succ)
                                        ori_pred_lane = self.lanelet_network.find_lanelet_by_id(ori_pred)
                                        conflict_new_s = np.minimum(current_curvePt.s + ori_pred_lane.center_curve[-1].s, conflict_lanelet.center_curve[-1].s)
                                        conflict_new_index,_ = get_curve_index(CurveIndex(0,0.0), conflict_lanelet.center_curve, conflict_new_s)
                                        future_indexs.append((conflict_new_index, succ, 5., "forward"))
                                        if future_indexs[-1][0].t >1.0 or future_indexs[-1][0].t < 0.0:
                                            print("23 {}".format(future_indexs[-1]))
                                        all_lanelets.add(succ)

                else:
                    new_index, _ = get_curve_index(current_index, current_curve, -grid_length)
                    future_indexs.append((new_index,current_lanelet_id, current_disp + grid_length, "backward"))
                    if future_indexs[-1][0].t > 1.0 or future_indexs[-1][0].t < 0.0:
                        print("24 {}".format(future_indexs[-1]))

        # backward
        '''
        while len(future_indexs_backward) > 0:
            current_index, current_lanelet_id, current_disp = future_indexs_backward.pop(0)
            current_lanelet = self.lanelet_network.find_lanelet_by_id(current_lanelet_id)
            current_curve = current_lanelet.center_curve
            current_curvePt = lerp_curve(current_curve[current_index.i], current_curve[current_index.i+1], current_index.t)
            if current_disp + grid_length > max_disp:
                continue
            elif current_disp != 0.0:
                print("backward", current_lanelet_id, current_index.i)
                cand_indexs.append((current_index, current_lanelet_id))
            if current_curvePt.s - grid_length < 0.0:
                for pred in current_lanelet.predecessor:
                    predecessor_lane = self.lanelet_network.find_lanelet_by_id(pred)
                    new_s = predecessor_lane.center_curve[-1].s + current_curvePt.s - grid_length
                    new_index, _ = get_curve_index(CurveIndex(0,0.0), predecessor_lane.center_curve, new_s)
                    future_indexs_backward.append((new_index, pred, current_disp + grid_length))
            else:
                new_index, _ = get_curve_index(current_index, current_curve, -grid_length)
                future_indexs_backward.append((new_index,current_lanelet_id, current_disp + grid_length))
        '''
        grids = []
        for center_index, center_lanelet_id in cand_indexs:

            center_lanelet = self.lanelet_network.find_lanelet_by_id(center_lanelet_id)
            center_curve = center_lanelet.center_curve
            center_curvePt = lerp_curve(center_curve[center_index.i], center_curve[center_index.i+1], center_index.t)
            #distance2 = (center_curvePt.pos.x-ego_posG.x)**2 + (center_curvePt.pos.y-ego_posG.y)**2
            #if distance2 > max_radius**2:
            #    continue
            to_points = []
            if center_curvePt.s + grid_length * 0.5 > center_curve[-1].s:
                if len(center_lanelet.successor) > 0:
                    for succ in center_lanelet.successor:
                        succ_lanelet = self.lanelet_network.find_lanelet_by_id(succ)
                        new_index,_ = get_curve_index(CurveIndex(0,0.0), succ_lanelet.center_curve, center_curvePt.s+grid_length*0.5-center_curve[-1].s)
                        to_points.append((new_index, succ, grid_length*0.5))
                else:
                    new_index = CurveIndex(len(center_curve)-2, 1.0)
                    to_points.append((new_index,center_lanelet_id, center_curve[-1].s-center_curvePt.s))
            else:
                if center_curvePt.s + grid_length > center_curve[-1].s and len(center_lanelet.successor) == 0:
                    new_index = CurveIndex(len(center_curve)-2,1.0)
                    to_points.append((new_index,center_lanelet_id, center_curve[-1].s-center_curvePt.s))
                else:
                    new_index, _ = get_curve_index(center_index, center_curve, grid_length*0.5)
                    to_points.append((new_index,center_lanelet_id, grid_length * 0.5))

            from_points = []
            if center_curvePt.s - grid_length * 0.5 < 0.0:
                if len(center_lanelet.predecessor) > 0:
                    for pred in center_lanelet.predecessor:
                        pred_lanelet = self.lanelet_network.find_lanelet_by_id(pred)
                        new_s = pred_lanelet.center_curve[-1].s + center_curvePt.s - grid_length*0.5
                        new_index, _ = get_curve_index(CurveIndex(0,0.0), pred_lanelet.center_curve, new_s)
                        assert new_index.t <=1.0 , "pred_lanelet {} length {} new_s {}".format(pred, pred_lanelet.center_curve[-1].s, new_s)
                        from_points.append((new_index, pred, -grid_length * 0.5))
                else:
                    new_index = CurveIndex(0, 0.0)
                    from_points.append((new_index,center_lanelet_id, -center_curvePt.s))
            else:
                if center_curvePt.s - grid_length < 0.0 and len(center_lanelet.predecessor) == 0:
                    new_index = CurveIndex(0, 0.0)
                    from_points.append((new_index,center_lanelet_id, -center_curvePt.s))
                else:
                    new_index, _ = get_curve_index(center_index, center_curve, -grid_length*0.5)
                    from_points.append((new_index,center_lanelet_id, -grid_length * 0.5))

            for from_index,from_lanelet_id,from_disp in from_points:
                for to_index,to_lanelet_id,to_disp in to_points:
                    grids.append(Grid(from_index=from_index,
                                      to_index=to_index,
                                      from_lanelet_id=from_lanelet_id,
                                      to_lanelet_id=to_lanelet_id,
                                      center_index=center_index,
                                      center_lanelet_id=center_lanelet_id,
                                      from_disp=from_disp,
                                      to_disp=to_disp, lanelet_network=self.lanelet_network))
        return grids

class CoreFeatureExtractor0:
    def __init__(self, config):
        self.horizon_steps = config["horizon_steps"]
        self.delta_step = config["delta_step"]
        self.prediction_steps = config["prediction_steps"]
        self.horizon_len = int(self.horizon_steps/self.delta_step)
        self.prediction_len = int(self.prediction_steps/self.delta_step)
        self.num_features = 3+6*self.horizon_len+6*self.prediction_len

        self.feature_info = { "egoid":{"high":3000, "low":0},
                              "obs_xs":{"high":40.,    "low":-40.},
                              "obs_ys":{"high":40.,    "low":-40.},
                              "obs_rel_xs":{"high":40.,    "low":-40.},
                              "obs_rel_ys":{"high":40.,    "low":-40.},
                              "masked":{"high":1, "low":0},
                              "length":{"high":30.,    "low":2.},
                              "width":{"high":3.,     "low":.9},
                              "fut_xs":{"high":40, "low":0},
                              "fut_ys":{"high":40, "low":0},
                              "fut_rel_xs":{"high":40, "low":0},
                              "fut_rel_ys":{"high":40, "low":0},
                              "has_preds":{"high":True, "low":False},
                              "obs_lanelet":{"high":500, "low":0},
                              "fut_lanelet":{"high":500, "low":0},
                              }

        self.feature_index = {"egoid":0,
                              "obs_xs":[1+i for i in range(self.horizon_len)],
                              "obs_ys":[1+self.horizon_len+i for i in range(self.horizon_len)],
                              "obs_lanelet":[1+2*self.horizon_len+i for i in range(self.horizon_len)],
                              "obs_rel_xs":[1+3*self.horizon_len+i for i in range(self.horizon_len)],
                              "obs_rel_ys":[1+4*self.horizon_len+i for i in range(self.horizon_len)],
                              "masked":[1+5*self.horizon_len+i for i in range(self.horizon_len)],

                              "length":1+6*self.horizon_len,
                              "width":2+6*self.horizon_len,
                              "fut_xs":[3+6*self.horizon_len+i for i in range(self.prediction_len)],
                              "fut_ys":[3+6*self.horizon_len+self.prediction_len+i for i in range(self.prediction_len)],
                              "fut_lanelet":[3+6*self.horizon_len+2*self.prediction_len+i for i in range(self.prediction_len)],
                              "fut_rel_xs":[3+6*self.horizon_len+3*self.prediction_len+i for i in range(self.prediction_len)],
                              "fut_rel_ys":[3+6*self.horizon_len+4*self.prediction_len+i for i in range(self.prediction_len)],
                              "has_preds":[3+6*self.horizon_len+5*self.prediction_len+i for i in range(self.prediction_len)],
                              }

    def set_origin(self, cr_scenario,  vehicle_id, startframe):
        ego_vehicle = cr_scenario.obstacles[startframe][vehicle_id]
        self.pos = ego_vehicle.initial_state.posG

    def get_xy(self, cr_scenario, vehicle_id, startframe, step):
        if vehicle_id in cr_scenario.obstacles[startframe] and vehicle_id in cr_scenario.obstacles[startframe-step]:
            current_state = cr_scenario.obstacles[startframe][vehicle_id].initial_state
            current_proj = current_state.posG.inertial2body(self.pos)
            cur_x, cur_y = current_proj.x, current_proj.y
            last_state = cr_scenario.obstacles[startframe-step][vehicle_id].initial_state
            last_proj = last_state.posG.inertial2body(self.pos)
            las_x, las_y = last_proj.x, last_proj.y
            delta_x, delta_y = cur_x - las_x, cur_y - las_y
            lanelet = current_state.posF.ind[1]

            return cur_x, cur_y, delta_x, delta_y, lanelet, 1
        elif vehicle_id in cr_scenario.obstacles[startframe]:
            current_state = cr_scenario.obstacles[startframe][vehicle_id].initial_state
            current_proj = current_state.posG.inertial2body(self.pos)
            cur_x, cur_y = current_proj.x, current_proj.y
            lanelet = current_state.posF.ind[1]

            return cur_x, cur_y, 0, 0, lanelet, 0
        else:
            return 0, 0, 0, 0, 0, 0

    def get_delta(self, cr_scenario, vehicle_id, startframe, step):
        if vehicle_id in cr_scenario.obstacles[startframe] and vehicle_id in cr_scenario.obstacles[startframe+step]:
            current_state = cr_scenario.obstacles[startframe][vehicle_id].initial_state
            current_proj = current_state.posG.inertial2body(self.pos)
            cur_x, cur_y = current_proj.x, current_proj.y
            target_state = cr_scenario.obstacles[startframe+step][vehicle_id].initial_state
            target_proj = target_state.posG.inertial2body(self.pos)
            tar_x, tar_y = target_proj.x, target_proj.y
            delta_x, delta_y = tar_x - cur_x, tar_y - cur_y
            lanelet = current_state.posF.ind[1]
            return tar_x, tar_y, delta_x, delta_y, lanelet, True
        elif vehicle_id in cr_scenario.obstacles[startframe]:
            current_state = cr_scenario.obstacles[startframe][vehicle_id].initial_state
            lanelet = current_state.posF.ind[1]
            return 0, 0, 0, 0, lanelet, False
        else :
            return 0, 0, 0, 0, 0, False

    def get_features(self, cr_scenario, vehicle_id, startframe):
        self.features = np.zeros(self.num_features, dtype=np.float32)
        # History trajectories can be missing
        #for t in range(self.horizon_len+1):
        #    if vehicle_id not in cr_scenario.obstacles[startframe-t*self.delta_step].keys():
        #        self.features = np.ones(self.num_features, dtype=np.float32)*-100
        #        return self.features
        #if vehicle_id not in cr_scenario.obstacles[startframe-self.horizon_len//2*self.delta_step].keys():
        #    self.features = np.ones(self.num_features, dtype=np.float32)*-100
        #    return self.features

        #for t in range(self.prediction_len):
        #if vehicle_id not in cr_scenario.obstacles[startframe+self.delta_step].keys():
        #    self.features = np.ones(self.num_features, dtype=np.float32)*-100
        #    return self.features

        ego_vehicle = cr_scenario.obstacles[startframe][vehicle_id]
        ego_pos = ego_vehicle.initial_state.posG
        self.features[self.feature_index["egoid"]] = vehicle_id
        self.features[self.feature_index["length"]] = ego_vehicle.obstacle_shape.length
        self.features[self.feature_index["width"]] = ego_vehicle.obstacle_shape.width

        for t in range(self.horizon_len):
            self.features[self.feature_index["obs_xs"][t]],\
            self.features[self.feature_index["obs_ys"][t]],\
            self.features[self.feature_index["obs_rel_xs"][t]],\
            self.features[self.feature_index["obs_rel_ys"][t]],\
            self.features[self.feature_index["obs_lanelet"][t]],\
            self.features[self.feature_index["masked"][t]] = self.get_xy(cr_scenario, vehicle_id, startframe-(self.horizon_len-1-t)*self.delta_step, self.delta_step)

            #ego_vehicle = cr_scenario.obstacles[startframe-(self.horizon_len-1-t)*self.delta_step][vehicle_id]
            #lane = cr_scenario.lanelet_network.find_lanelet_by_id(ego_vehicle.initial_state.posF.ind[1])
            #if lane.adj_left is not None:
            #    self.features[self.feature_index["lane_lefts"][t]] = 1
            #else:
            #    self.features[self.feature_index["lane_lefts"][t]] = 0
            #if lane.adj_right is not None:
            #    self.features[self.feature_index["lane_rights"][t]] = 1
            #else:
            #    self.features[self.feature_index["lane_rights"][t]] = 0
        for t in range(self.prediction_len):
            self.features[self.feature_index["fut_xs"][t]],\
            self.features[self.feature_index["fut_ys"][t]],\
            self.features[self.feature_index["fut_rel_xs"][t]],\
            self.features[self.feature_index["fut_rel_ys"][t]],\
            self.features[self.feature_index["fut_lanelet"][t]],\
            self.features[self.feature_index["has_preds"][t]] = self.get_delta(cr_scenario, vehicle_id, startframe+t*self.delta_step, self.delta_step)
        return self.features

class CoreFeatureExtractor:
    def __init__(self, config):
        self.delta_step = config["delta_step"]
        self.prediction_steps = config["prediction_steps"]
        self.prediction_len = int(self.prediction_steps/self.delta_step)
        self.num_features = 7+4*self.prediction_len

        self.feature_info = { "egoid":{"high":3000, "low":0},
                              "obs_x":{"high":40.,    "low":-40.},
                              "obs_y":{"high":40.,    "low":-40.},
                              "obs_rel_x":{"high":40.,    "low":-40.},
                              "obs_rel_y":{"high":40.,    "low":-40.},
                              "length":{"high":30.,    "low":2.},
                              "width":{"high":3.,     "low":.9},
                              "fut_xs":{"high":40, "low":0},
                              "fut_ys":{"high":40, "low":0},
                              "fut_rel_xs":{"high":40, "low":0},
                              "fut_rel_ys":{"high":40, "low":0}
                              }

        self.feature_index = {"egoid":0,
                              "obs_x":1,
                              "obs_y":2,
                              "obs_rel_x":3,
                              "obs_rel_y":4,
                              "length":5,
                              "width":6,
                              "fut_xs":[7+i for i in range(self.prediction_len)],
                              "fut_ys":[7+self.prediction_len+i for i in range(self.prediction_len)],
                              "fut_rel_xs":[7+2*self.prediction_len+i for i in range(self.prediction_len)],
                              "fut_rel_ys":[7+3*self.prediction_len+i for i in range(self.prediction_len)]
                              }

    def set_origin(self, cr_scenario,  vehicle_id, startframe):
        ego_vehicle = cr_scenario.obstacles[startframe][vehicle_id]
        self.pos = ego_vehicle.initial_state.posG

    def get_xy(self, cr_scenario, vehicle_id, startframe, step):
        current_state = cr_scenario.obstacles[startframe][vehicle_id].initial_state
        current_proj = current_state.posG.inertial2body(self.pos)
        cur_x, cur_y = current_proj.x, current_proj.y
        if vehicle_id in cr_scenario.obstacles[startframe-step]:
            last_state = cr_scenario.obstacles[startframe-step][vehicle_id].initial_state
            last_proj = last_state.posG.inertial2body(self.pos)
            las_x, las_y = last_proj.x, last_proj.y
            delta_x, delta_y = cur_x - las_x, cur_y - las_y
        else:
            next_state = cr_scenario.obstacles[startframe+step][vehicle_id].initial_state
            next_proj = next_state.posG.inertial2body(self.pos)
            nex_x, nex_y = next_proj.x, next_proj.y
            delta_x, delta_y = nex_x - cur_x, nex_y - cur_y
        return cur_x, cur_y, delta_x, delta_y

    def get_delta(self, cr_scenario, vehicle_id, startframe, step):
        current_state = cr_scenario.obstacles[startframe][vehicle_id].initial_state
        current_proj = current_state.posG.inertial2body(self.pos)
        cur_x, cur_y = current_proj.x, current_proj.y
        target_state = cr_scenario.obstacles[startframe+step][vehicle_id].initial_state
        target_proj = target_state.posG.inertial2body(self.pos)
        tar_x, tar_y = target_proj.x, target_proj.y
        delta_x, delta_y = tar_x - cur_x, tar_y - cur_y
        return tar_x, tar_y, delta_x, delta_y

    def get_features(self, cr_scenario, vehicle_id, startframe):
        self.features = np.zeros(self.num_features, dtype=np.float32)

        ego_vehicle = cr_scenario.obstacles[startframe][vehicle_id]
        ego_pos = ego_vehicle.initial_state.posG
        self.features[self.feature_index["egoid"]] = vehicle_id
        self.features[self.feature_index["length"]] = ego_vehicle.obstacle_shape.length
        self.features[self.feature_index["width"]] = ego_vehicle.obstacle_shape.width

        self.features[self.feature_index["obs_x"]],\
        self.features[self.feature_index["obs_y"]],\
        self.features[self.feature_index["obs_rel_x"]],\
        self.features[self.feature_index["obs_rel_y"]] = self.get_xy(cr_scenario, vehicle_id, startframe, self.delta_step)

        for t in range(self.prediction_len):
            if vehicle_id not in cr_scenario.obstacles[startframe+(t+1)*self.delta_step].keys():
                for t in range(self.prediction_len):
                    self.features[self.feature_index["fut_xs"][t]],\
                    self.features[self.feature_index["fut_ys"][t]],\
                    self.features[self.feature_index["fut_rel_xs"][t]],\
                    self.features[self.feature_index["fut_rel_ys"][t]] = -100, -100, -100, -100
                return self.features

        for t in range(self.prediction_len):
            self.features[self.feature_index["fut_xs"][t]],\
            self.features[self.feature_index["fut_ys"][t]],\
            self.features[self.feature_index["fut_rel_xs"][t]],\
            self.features[self.feature_index["fut_rel_ys"][t]] = self.get_delta(cr_scenario, vehicle_id, startframe+t*self.delta_step, self.delta_step)
        return self.features



class CoreFeatureExtractor1:
    def __init__(self):
        self.num_features = 14
        self.features = np.zeros(self.num_features, dtype=np.float32)

        self.feature_info = { "egoid":{"high":3000, "low":0},
                              "relative_offset":{"high":2.,    "low":-2.},
                              "velocity_s":{"high":40.,    "low":0.},
                              "velocity_d":{"high":40.,    "low":0.},
                              "length":{"high":30.,    "low":2.},
                              "width":{"high":3.,     "low":.9},
                              "target_delta_s":{"high":40, "low":0},
                              "target_delta_d":{"high":40, "low":0},
                              "target_velocity_s":{"high":40, "low":0},
                              "target_velocity_d":{"high":40, "low":0},
                              "target_delta_x":{"high":40, "low":0},
                              "target_delta_y":{"high":40, "low":0},
                              "target_velocity_x":{"high":40, "low":0},
                              "target_velocity_y":{"high":40, "low":0},
                              }

        self.feature_index = {"egoid":0,
                              "relative_offset":1,
                              "velocity_s":2,
                              "velocity_d":3,
                              "length":4,
                              "width":5,
                              "target_delta_s":6,
                              "target_delta_d":7,
                              "target_velocity_s":8,
                              "target_velocity_d":9,
                              "target_delta_x":10,
                              "target_delta_y":11,
                              "target_velocity_x":12,
                              "target_velocity_y":13,
                              }
    def set_step_length_and_maneuver_horizon(self, step_length, horizon):
        self.step_length = step_length
        self.horizon = horizon

    def get_delta(self, cr_scenario, vehicle_id, startframe, step):
        ego_vehicle = cr_scenario.obstacles[startframe][vehicle_id]
        current_state = ego_vehicle.initial_state
        current_posF = current_state.posF
        current_lane = cr_scenario.lanelet_network.find_lanelet_by_id(current_posF.ind[1])
        origin_current_lanelet_id = current_lane.lanelet_id
        if current_lane.adj_right is not None: # this is for rounD dataset
            current_lane = cr_scenario.lanelet_network.find_lanelet_by_id(current_lane.adj_right)
            current_posF = Frenet(current_state.posG.projF(cr_scenario.lanelet_network, [current_lane.lanelet_id]), cr_scenario.lanelet_network)

        target_state = cr_scenario.obstacles[startframe+step][vehicle_id].initial_state
        target_proj = target_state.posG.inertial2body(current_state.posG)
        delta_x, delta_y, proj_th = target_proj.x, target_proj.y, target_proj.th

        target_posF = target_state.posF
        target_lane = cr_scenario.lanelet_network.find_lanelet_by_id(target_posF.ind[1])
        origin_target_lanelet_id = target_lane.lanelet_id
        if target_lane.adj_right is not None:
            target_lane = cr_scenario.lanelet_network.find_lanelet_by_id(target_lane.adj_right)
            target_posF = Frenet(target_state.posG.projF(cr_scenario.lanelet_network, [target_lane.lanelet_id]), cr_scenario.lanelet_network)

        if target_posF.ind[1] == current_posF.ind[1]:
            delta_s = target_posF.s - current_posF.s
            delta_d = target_posF.d - current_posF.d
            #target_lanelet_id = origin_target_lanelet_id
        else:
            posG = target_state.get_posG()
            posF_adjust = Frenet(posG.projF(cr_scenario.lanelet_network, [current_posF.ind[1]]), cr_scenario.lanelet_network)
            max_ind = len(current_lane.center_curve)-2
            if posF_adjust.ind[0].i == 0 and posF_adjust.ind[0].t == 0.0 and len(current_lane.predecessor) > 0:
                # normally this should not happen
                delta_s = 0.0
                delta_d = posF_adjust.d - current_posF.d
            elif posF_adjust.ind[0].i >= max_ind and posF_adjust.ind[0].t == 1.0 and len(current_lane.successor) > 0:
                #TODO In the roundabout, this should be different
                dist_s = [0.0]
                cands = [current_posF.ind[1]]
                delta_d = None
                while len(cands) > 0:
                    this_cand = cands.pop(0)
                    this_dist_s = dist_s.pop(0)
                    if this_cand == target_posF.ind[1]:
                        delta_s = target_posF.s + this_dist_s - current_posF.s
                        delta_d = target_posF.d - current_posF.d
                        #target_lanelet_id = origin_target_lanelet_id
                        break
                    else:
                        this_lane = cr_scenario.lanelet_network.find_lanelet_by_id(this_cand)
                        posF_adjust = Frenet(posG.projF(cr_scenario.lanelet_network, [this_cand]), cr_scenario.lanelet_network)
                        if posF_adjust.ind[0].i >= len(this_lane.center_curve)-2 and posF_adjust.ind[0].t == 1.0:
                            for succ in this_lane.successor:
                                successor_lane = cr_scenario.lanelet_network.find_lanelet_by_id(succ)
                                cands.append(succ)
                                dist_s.append(this_dist_s+this_lane.center_curve[-1].s)
                        else:
                            if delta_d is None:
                                delta_s = posF_adjust.s + this_dist_s - current_posF.s
                                delta_d = posF_adjust.d - current_posF.d
                                #target_lanelet_id = origin_target_lanelet_id

                            elif np.abs(posF_adjust.d - current_posF.d) < np.abs(delta_d):
                                delta_s = posF_adjust.s + this_dist_s - current_posF.s
                                delta_d = posF_adjust.d - current_posF.d
                                #target_lanelet_id = origin_target_lanelet_id
            else:
                delta_s = posF_adjust.s - current_posF.s
                delta_d = posF_adjust.d - current_posF.d
                #target_lanelet_id = origin_target_lanelet_id

        return delta_s, delta_d, delta_x, delta_y, proj_th
    def get_features(self, cr_scenario, vehicle_id, startframe):
        ego_vehicle = cr_scenario.obstacles[startframe][vehicle_id]
        self.features[self.feature_index["egoid"]] = vehicle_id
        self.features[self.feature_index["relative_offset"]] = ego_vehicle.initial_state.posF.d
        self.features[self.feature_index["velocity_s"]] = np.cos(ego_vehicle.initial_state.posF.phi)*ego_vehicle.initial_state.velocity
        self.features[self.feature_index["velocity_d"]] = np.sin(ego_vehicle.initial_state.posF.phi)*ego_vehicle.initial_state.velocity
        self.features[self.feature_index["length"]] = ego_vehicle.obstacle_shape.length
        self.features[self.feature_index["width"]] = ego_vehicle.obstacle_shape.width
        delta_step = int(self.step_length/cr_scenario.dt) # should be 10 if dt = 0.1
        #horizon_step = int(self.horizon/cr_scenario.dt)
        if vehicle_id in cr_scenario.obstacles[startframe+delta_step].keys():
            self.features[self.feature_index["target_delta_s"]],\
            self.features[self.feature_index["target_delta_d"]],\
            self.features[self.feature_index["target_delta_x"]],\
            self.features[self.feature_index["target_delta_y"]], proj_th = self.get_delta(cr_scenario, vehicle_id, startframe, delta_step)
            target_state = cr_scenario.obstacles[startframe+delta_step][vehicle_id].initial_state
            self.features[self.feature_index["target_velocity_s"]] = np.cos(target_state.posF.phi)*target_state.velocity
            self.features[self.feature_index["target_velocity_d"]] = np.sin(target_state.posF.phi)*target_state.velocity
            self.features[self.feature_index["target_velocity_x"]] = np.cos(proj_th)*target_state.velocity
            self.features[self.feature_index["target_velocity_y"]] = np.sin(proj_th)*target_state.velocity

        elif vehicle_id in cr_scenario.obstacles[startframe+1].keys():
            i = 1
            while True:
                if vehicle_id in cr_scenario.obstacles[startframe+delta_step-i].keys():
                    delta_s,\
                    delta_d, delta_x, delta_y, proj_th = self.get_delta(cr_scenario, vehicle_id, startframe, delta_step-i)
                    self.features[self.feature_index["target_delta_s"]] = delta_s*delta_step/(delta_step-i)
                    self.features[self.feature_index["target_delta_d"]] = delta_d*delta_step/(delta_step-i)
                    self.features[self.feature_index["target_delta_x"]] = delta_x*delta_step/(delta_step-i)
                    self.features[self.feature_index["target_delta_y"]] = delta_y*delta_step/(delta_step-i)

                    target_state = cr_scenario.obstacles[startframe+delta_step-i][vehicle_id].initial_state
                    self.features[self.feature_index["target_velocity_s"]] = np.cos(target_state.posF.phi)*target_state.velocity
                    self.features[self.feature_index["target_velocity_d"]] = np.sin(target_state.posF.phi)*target_state.velocity
                    self.features[self.feature_index["target_velocity_x"]] = np.cos(proj_th)*target_state.velocity
                    self.features[self.feature_index["target_velocity_y"]] = np.sin(proj_th)*target_state.velocity
                    break
                i += 1
            assert(i < delta_step)
        else:
            i = 0
            while True:
                if vehicle_id in cr_scenario.obstacles[startframe-delta_step+i].keys():
                    delta_s,delta_d, delta_x, delta_y, proj_th = self.get_delta(cr_scenario, vehicle_id, startframe-delta_step+i, delta_step-i)
                    self.features[self.feature_index["target_delta_s"]] = delta_s*delta_step/(delta_step-i)
                    self.features[self.feature_index["target_delta_d"]] = delta_d*delta_step/(delta_step-i)
                    self.features[self.feature_index["target_delta_x"]] = delta_x*delta_step/(delta_step-i)
                    self.features[self.feature_index["target_delta_y"]] = delta_y*delta_step/(delta_step-i)

                    target_state = cr_scenario.obstacles[startframe-delta_step+i][vehicle_id].initial_state
                    self.features[self.feature_index["target_velocity_s"]] = np.cos(target_state.posF.phi)*target_state.velocity
                    self.features[self.feature_index["target_velocity_d"]] = np.sin(target_state.posF.phi)*target_state.velocity
                    self.features[self.feature_index["target_velocity_x"]] = np.cos(proj_th)*target_state.velocity
                    self.features[self.feature_index["target_velocity_y"]] = np.sin(proj_th)*target_state.velocity

                    break
                i += 1
            assert(i < delta_step)

        return self.features

class CoreFeatureExtractor2:
    def __init__(self, config):
        self.horizon_steps = config["horizon_steps"]
        self.delta_step = config["delta_step"]
        self.fut_steps = config["sim_delta_step"]
        self.horizon_len = int(self.horizon_steps/self.delta_step)
        self.num_features = 11+3*self.horizon_len

        self.feature_info = { "egoid":{"high":3000, "low":0},
                              "obs_xs":{"high":40.,    "low":-40.},
                              "obs_ys":{"high":40.,    "low":-40.},
                              "masked":{"high":1, "low":0},
                              "length":{"high":30.,    "low":2.},
                              "width":{"high":3.,     "low":.9},
                              "s":{"high":20.,     "low":0.},
                              "ss":{"high":20.,     "low":0.},
                              "d":{"high":5.,     "low":-5.},
                              "dd":{"high":10.,     "low":-10.},
                              "phi":{"high":np.pi/2,     "low":-np.pi/2.},
                              "vel":{"high":20.,     "low":0.},
                              "delta_s":{"high":20.,     "low":0.},
                              "delta_d":{"high":10.,     "low":-10.}
                              }

        self.feature_index = {"egoid":0,
                              "obs_xs":[1+i for i in range(self.horizon_len)],
                              "obs_ys":[1+self.horizon_len+i for i in range(self.horizon_len)],
                              "masked":[1+2*self.horizon_len+i for i in range(self.horizon_len)],

                              "length":1+3*self.horizon_len,
                              "width":2+3*self.horizon_len,
                              "s":3+3*self.horizon_len,
                              "ss":4+3*self.horizon_len,
                              "d":5+3*self.horizon_len,
                              "dd":6+3*self.horizon_len,
                              "phi":7+3*self.horizon_len,
                              "vel":8+3*self.horizon_len,

                              "delta_s":9+3*self.horizon_len,
                              "delta_d":10+3*self.horizon_len}

    def set_origin(self, cr_scenario,  vehicle_id, startframe):
        ego_vehicle = cr_scenario.obstacles[startframe][vehicle_id]
        self.pos = ego_vehicle.initial_state.posG

    def get_xy(self, cr_scenario, vehicle_id, startframe, step):
        if vehicle_id in cr_scenario.obstacles[startframe]:
            current_state = cr_scenario.obstacles[startframe][vehicle_id].initial_state
            current_proj = current_state.posG.inertial2body(self.pos)
            cur_x, cur_y = current_proj.x, current_proj.y

            return cur_x, cur_y,  1
        else:
            return 0, 0, 0

    def get_features(self, cr_scenario, vehicle_id, startframe, ego=False, act=False):
        self.features = np.zeros(self.num_features, dtype=np.float32)
        # History trajectories can be missing
        if vehicle_id not in cr_scenario.obstacles[startframe].keys():
            raise

        ego_vehicle = cr_scenario.obstacles[startframe][vehicle_id]
        ego_pos = ego_vehicle.initial_state.posG
        self.features[self.feature_index["egoid"]] = vehicle_id
        self.features[self.feature_index["length"]] = ego_vehicle.obstacle_shape.length
        self.features[self.feature_index["width"]] = ego_vehicle.obstacle_shape.width

        for t in range(self.horizon_len):
            self.features[self.feature_index["obs_xs"][t]],\
            self.features[self.feature_index["obs_ys"][t]],\
            self.features[self.feature_index["masked"][t]] = self.get_xy(cr_scenario, vehicle_id, startframe-(self.horizon_len-1-t)*self.delta_step, self.delta_step)

        if not ego:
            return self.features
        principle_curve = cr_scenario.route.curves[cr_scenario.route.principle_reference.index(1)]
        principle_ind = cr_scenario.route.ind
        low_ind = max(principle_ind-5, 0)
        high_ind = min(principle_ind+20, len(principle_curve))
        closest_ind = ego_pos.index_closest_to_point(principle_curve[low_ind:high_ind])+low_ind
        while closest_ind >= high_ind-1:
            high_ind += 1
            closest_ind = ego_pos.index_closest_to_point(principle_curve[low_ind:high_ind])+low_ind

        s_proj = ego_pos.proj_on_curve(principle_curve[low_ind:high_ind], clamped=False)
        s_s = lerp_curve_with_ind(principle_curve[low_ind:high_ind], s_proj.ind)
        phi = s_proj.phi
        vel = ego_vehicle.initial_state.velocity
        if hasattr(ego_vehicle.initial_state, "velocity_y"):
            vel = np.sqrt(vel**2 + ego_vehicle.initial_state.velocity_y**2)
        dd = np.sin(phi) * ego_vehicle.initial_state.velocity
        self.features[self.feature_index["s"]] = s_s
        self.features[self.feature_index["ss"]] = np.cos(phi) * vel
        self.features[self.feature_index["d"]] = s_proj.d
        self.features[self.feature_index["dd"]] = np.sin(phi) * vel
        self.features[self.feature_index["phi"]] = phi
        self.features[self.feature_index["vel"]] = vel

        if not act:
            return self.features

        if vehicle_id not in cr_scenario.obstacles[startframe+self.fut_steps].keys():
            raise
        future_vehicle = cr_scenario.obstacles[startframe+self.fut_steps][vehicle_id]
        fut_pos = future_vehicle.initial_state.posG
        closest_ind = fut_pos.index_closest_to_point(principle_curve[low_ind:high_ind])+low_ind
        while closest_ind >= high_ind-1:
            high_ind += 1
            closest_ind = fut_pos.index_closest_to_point(principle_curve[low_ind:high_ind])+low_ind

        cr_scenario.route.ind = closest_ind
        e_proj = fut_pos.proj_on_curve(principle_curve[low_ind:high_ind], clamped=False)
        s_e = lerp_curve_with_ind(principle_curve[low_ind:high_ind], e_proj.ind)
        self.features[self.feature_index["delta_s"]] = s_e - s_s
        self.features[self.feature_index["delta_d"]] = e_proj.d

        return self.features

    def get_features_by_name(self, features, names):
        ret = []
        for n in names:
            if n in self.feature_index.keys():
                try:
                    ret.extend(features[self.feature_index[n]])
                except:
                    ret.append(features[self.feature_index[n]])
        return ret
class TemporalFeatureExtractor:
    def __init__(self):
        self.num_features = 5
        self.features = np.zeros(self.num_features, dtype=np.float32)
        self.feature_info = {"accel" : {"high":9.,    "low":-9.},
                             "timegap" :  {"high":30.,  "low":0.},
                             "timegap_is_avail" :  {"high":1.,   "low":0.},
                             "inv_time_to_collision" : {"high":30.,  "low":0.},
                             "inv_time_to_collision_is_avail" : {"high":1.,   "low":0.}}
        self.feature_index = {"accel" : 0,
                             "timegap" : 1,
                             "timegap_is_avail" : 2,
                             "inv_time_to_collision" : 3,
                             "inv_time_to_collision_is_avail" : 4}

    def get_features(self, cr_scenario, vehicle_id, startframe):
        ego_vehicle = cr_scenario.obstacles[startframe][vehicle_id]
        ego_vehicle2 = cr_scenario.obstacles[startframe-1][vehicle_id]
        self.features[0] = (ego_vehicle.initial_state.velocity - ego_vehicle2.initial_state.velocity)/0.1
        self.features[1], self.features[2] = cr_scenario.get_attribute(TIMEGAP, vehicle_id, startframe)
        self.features[3], self.features[4] = cr_scenario.get_attribute(TTC, vehicle_id, startframe)
        return self.features

class NeighborFeatureExtractor0:
    def __init__(self, config):
        self.num_features = 7 * 8
        self.features = np.zeros(self.num_features, dtype=np.float32)
        self.feature_info = {}
        self.feature_names = feature_names = ["_veh_id", "_dist_ins", "_offset_ins", "_vel_x", "_vel_y", "_length", "_width"]
        feature_ranges = [{"high":2000,"low":1}, {"high":250.0, "low":-100.0}, {"high":4.0, "low":-4.0}, {"high":20, "low":0}, {"high":20, "low":0}, {"high":5.0, "low":0.0}, {"high":4.0, "low":0.0}]
        self.neighbor_names = neighbor_names = ["front", "left_front", "left", "right_front", "right",
                          "left_rear", "right_rear", "rear"]
        self.feature_index = {}
        index = 0
        for nn in neighbor_names:
            for i in range(len(feature_names)):
                self.feature_info[nn+feature_names[i]] = feature_ranges[i]
                self.feature_index[nn+feature_names[i]] = index
                index += 1

    def get_features(self, cr_scenario, vehicle_id, startframe):
        self.features = np.zeros(self.num_features, dtype=np.float32)
        self.features[self.feature_index["front_veh_id"]],\
        self.features[self.feature_index["front_dist_ins"]],\
        self.features[self.feature_index["front_offset_ins"]],\
        self.features[self.feature_index["front_vel_x"]],\
        self.features[self.feature_index["front_vel_y"]] = cr_scenario.get_attribute(NEIGHBOR_FORE_ALONG_LANE, vehicle_id, startframe)
        if self.features[self.feature_index["front_veh_id"]] != -100:
            self.features[self.feature_index["front_length"]] = cr_scenario.obstacles[startframe][self.features[self.feature_index["front_veh_id"]]].obstacle_shape.length
            self.features[self.feature_index["front_width"]] = cr_scenario.obstacles[startframe][self.features[self.feature_index["front_veh_id"]]].obstacle_shape.width
        else:
            self.features[self.feature_index["front_length"]] = 0.0
            self.features[self.feature_index["front_width"]] = 0.0

        self.features[self.feature_index["left_front_veh_id"]],\
        self.features[self.feature_index["left_front_dist_ins"]],\
        self.features[self.feature_index["left_front_offset_ins"]],\
        self.features[self.feature_index["left_front_vel_x"]],\
        self.features[self.feature_index["left_front_vel_y"]] = cr_scenario.get_attribute(NEIGHBOR_FORE_ALONG_LEFT_LANE, vehicle_id, startframe)
        if self.features[self.feature_index["left_front_veh_id"]] != -100:
            self.features[self.feature_index["left_front_length"]] = cr_scenario.obstacles[startframe][self.features[self.feature_index["left_front_veh_id"]]].obstacle_shape.length
            self.features[self.feature_index["left_front_width"]] = cr_scenario.obstacles[startframe][self.features[self.feature_index["left_front_veh_id"]]].obstacle_shape.width
        else:
            self.features[self.feature_index["left_front_length"]] = 0.0
            self.features[self.feature_index["left_front_width"]] = 0.0

        self.features[self.feature_index["right_front_veh_id"]],\
        self.features[self.feature_index["right_front_dist_ins"]],\
        self.features[self.feature_index["right_front_offset_ins"]],\
        self.features[self.feature_index["right_front_vel_x"]],\
        self.features[self.feature_index["right_front_vel_y"]] = cr_scenario.get_attribute(NEIGHBOR_FORE_ALONG_RIGHT_LANE, vehicle_id, startframe)
        if self.features[self.feature_index["right_front_veh_id"]] != -100:
            self.features[self.feature_index["right_front_length"]] = cr_scenario.obstacles[startframe][self.features[self.feature_index["right_front_veh_id"]]].obstacle_shape.length
            self.features[self.feature_index["right_front_width"]] = cr_scenario.obstacles[startframe][self.features[self.feature_index["right_front_veh_id"]]].obstacle_shape.width
        else:
            self.features[self.feature_index["right_front_length"]] = 0.0
            self.features[self.feature_index["right_front_width"]] = 0.0

        self.features[self.feature_index["left_veh_id"]],\
        self.features[self.feature_index["left_dist_ins"]],\
        self.features[self.feature_index["left_offset_ins"]],\
        self.features[self.feature_index["left_vel_x"]],\
        self.features[self.feature_index["left_vel_y"]]  = cr_scenario.get_attribute(NEIGHBOR_LEFT_LANE, vehicle_id, startframe)
        if self.features[self.feature_index["left_veh_id"]] != -100:
            self.features[self.feature_index["left_length"]] = cr_scenario.obstacles[startframe][self.features[self.feature_index["left_veh_id"]]].obstacle_shape.length
            self.features[self.feature_index["left_width"]] = cr_scenario.obstacles[startframe][self.features[self.feature_index["left_veh_id"]]].obstacle_shape.width
        else:
            self.features[self.feature_index["left_length"]] = 0.0
            self.features[self.feature_index["left_width"]] = 0.0

        self.features[self.feature_index["right_veh_id"]],\
        self.features[self.feature_index["right_dist_ins"]],\
        self.features[self.feature_index["right_offset_ins"]],\
        self.features[self.feature_index["right_vel_x"]],\
        self.features[self.feature_index["right_vel_y"]] = cr_scenario.get_attribute(NEIGHBOR_RIGHT_LANE, vehicle_id, startframe)
        if self.features[self.feature_index["right_veh_id"]] != -100:
            self.features[self.feature_index["right_length"]] = cr_scenario.obstacles[startframe][self.features[self.feature_index["right_veh_id"]]].obstacle_shape.length
            self.features[self.feature_index["right_width"]] = cr_scenario.obstacles[startframe][self.features[self.feature_index["right_veh_id"]]].obstacle_shape.width
        else:
            self.features[self.feature_index["right_length"]] = 0.0
            self.features[self.feature_index["right_width"]] = 0.0

        self.features[self.feature_index["rear_veh_id"]],\
        self.features[self.feature_index["rear_dist_ins"]],\
        self.features[self.feature_index["rear_offset_ins"]],\
        self.features[self.feature_index["rear_vel_x"]],\
        self.features[self.feature_index["rear_vel_y"]] = cr_scenario.get_attribute(NEIGHBOR_REAR_ALONG_LANE, vehicle_id, startframe)
        if self.features[self.feature_index["rear_veh_id"]] != -100:
            self.features[self.feature_index["rear_length"]] = cr_scenario.obstacles[startframe][self.features[self.feature_index["rear_veh_id"]]].obstacle_shape.length
            self.features[self.feature_index["rear_width"]] = cr_scenario.obstacles[startframe][self.features[self.feature_index["rear_veh_id"]]].obstacle_shape.width
        else:
            self.features[self.feature_index["rear_length"]] = 0.0
            self.features[self.feature_index["rear_width"]] = 0.0

        self.features[self.feature_index["left_rear_veh_id"]],\
        self.features[self.feature_index["left_rear_dist_ins"]],\
        self.features[self.feature_index["left_rear_offset_ins"]],\
        self.features[self.feature_index["left_rear_vel_x"]],\
        self.features[self.feature_index["left_rear_vel_y"]]   = cr_scenario.get_attribute(NEIGHBOR_REAR_ALONG_LEFT_LANE, vehicle_id, startframe)
        if self.features[self.feature_index["left_rear_veh_id"]] != -100:
            self.features[self.feature_index["left_rear_length"]] = cr_scenario.obstacles[startframe][self.features[self.feature_index["left_rear_veh_id"]]].obstacle_shape.length
            self.features[self.feature_index["left_rear_width"]] = cr_scenario.obstacles[startframe][self.features[self.feature_index["left_rear_veh_id"]]].obstacle_shape.width
        else:
            self.features[self.feature_index["left_rear_length"]] = 0.0
            self.features[self.feature_index["left_rear_width"]] = 0.0

        self.features[self.feature_index["right_rear_veh_id"]],\
        self.features[self.feature_index["right_rear_dist_ins"]],\
        self.features[self.feature_index["right_rear_offset_ins"]],\
        self.features[self.feature_index["right_rear_vel_x"]],\
        self.features[self.feature_index["right_rear_vel_y"]] = cr_scenario.get_attribute(NEIGHBOR_REAR_ALONG_RIGHT_LANE, vehicle_id, startframe)
        if self.features[self.feature_index["right_rear_veh_id"]] != -100:
            self.features[self.feature_index["right_rear_length"]] = cr_scenario.obstacles[startframe][self.features[self.feature_index["right_rear_veh_id"]]].obstacle_shape.length
            self.features[self.feature_index["right_rear_width"]] = cr_scenario.obstacles[startframe][self.features[self.feature_index["right_rear_veh_id"]]].obstacle_shape.width
        else:
            self.features[self.feature_index["right_rear_length"]] = 0.0
            self.features[self.feature_index["right_rear_width"]] = 0.0

        return self.features

class NeighborFeatureExtractor:
    def __init__(self):
        self.num_features = 3 * 8
        self.features = np.zeros(self.num_features, dtype=np.float32)
        self.feature_info = {}
        self.feature_names = feature_names = ["_veh_id",  "_rel_x", "_rel_y"]
        feature_ranges = [{"high":2000,"low":1},
                          {"high":250.0, "low":-100.0},
                          {"high":250.0, "low":-100.0}]
        self.neighbor_names = neighbor_names = ["front1", "left1", "left2", "left3", "right1", "right2", "right3", "rear1"]
        self.feature_index = {}
        index = 0
        for nn in neighbor_names:
            for i in range(len(feature_names)):
                self.feature_info[nn+feature_names[i]] = feature_ranges[i]
                self.feature_index[nn+feature_names[i]] = index
                index += 1

    def set_origin(self, cr_scenario,  vehicle_id, startframe):
        ego_vehicle = cr_scenario.obstacles[startframe][vehicle_id]
        self.pos = ego_vehicle.initial_state.posG

    def get_rel_xy(self, cr_scenario, veh1, veh2, startframe):
        veh1_state = cr_scenario.obstacles[startframe][veh1].initial_state
        veh1_proj = veh1_state.posG.inertial2body(self.pos)
        veh1_x, veh1_y = veh1_proj.x, veh1_proj.y

        veh2_state = cr_scenario.obstacles[startframe][veh2].initial_state
        veh2_proj = veh2_state.posG.inertial2body(self.pos)
        veh2_x, veh2_y = veh2_proj.x, veh2_proj.y

        delta_x, delta_y = veh2_x - veh1_x, veh2_y - veh1_y
        return delta_x, delta_y

    def get_features(self, cr_scenario, vehicle_id, startframe):
        self.features = np.zeros(self.num_features, dtype=np.float32)
        ret = cr_scenario.get_attribute(NEIGHBOR_FORE_REAR_LEFT_MID_RIGHT, vehicle_id, startframe)
        # neighbors features: rel_pos, vel, length, width, rel_heading
        fore_M, fore_Mdist, fore_Moffset, fore_Mdist2, fore_Moffset2, fore_Mori2 = ret["front"]
        veh_id = fore_M
        if veh_id != -100:
            self.features[self.feature_index["front1_veh_id"]] = veh_id
            self.features[self.feature_index["front1_rel_x"]], \
            self.features[self.feature_index["front1_rel_y"]] = self.get_rel_xy(cr_scenario, vehicle_id, veh_id, startframe)
        else:
            self.features[self.feature_index["front1_veh_id"]] = -100
            self.features[self.feature_index["front1_rel_x"]] = -100
            self.features[self.feature_index["front1_rel_y"]] = -100

        rear_M, rear_Mdist, rear_Moffset, rear_Mdist2, rear_Moffset2, rear_Mori2 = ret["rear"]
        veh_id = rear_M
        if veh_id != -100:
            self.features[self.feature_index["rear1_veh_id"]] = veh_id
            self.features[self.feature_index["rear1_rel_x"]], \
            self.features[self.feature_index["rear1_rel_y"]] = self.get_rel_xy(cr_scenario, vehicle_id, veh_id, startframe)
        else:
            self.features[self.feature_index["rear1_veh_id"]] = -100
            self.features[self.feature_index["rear1_rel_x"]] = -100
            self.features[self.feature_index["rear1_rel_y"]] = -100

        Ls, Ldists, Loffsets, Ldists2, Loffsets2, Loris2 = ret["left"]
        for ii in range(len(Ls)):
            veh_id = Ls[ii]
            if veh_id != -100:
                self.features[self.feature_index["left{}_veh_id".format(ii+1)]] = veh_id
                self.features[self.feature_index["left{}_rel_x".format(ii+1)]], \
                self.features[self.feature_index["left{}_rel_y".format(ii+1)]] = self.get_rel_xy(cr_scenario, vehicle_id, veh_id, startframe)

            else:
                self.features[self.feature_index["left{}_veh_id".format(ii+1)]] = -100
                self.features[self.feature_index["left{}_rel_x".format(ii+1)]] = -100
                self.features[self.feature_index["left{}_rel_y".format(ii+1)]] = -100
        Rs, Rdists, Roffsets, Rdists2, Roffsets2, Roris2 = ret["right"]
        for ii in range(len(Rs)):
            veh_id = Rs[ii]
            if veh_id != -100:
                self.features[self.feature_index["right{}_veh_id".format(ii+1)]] = veh_id
                self.features[self.feature_index["right{}_rel_x".format(ii+1)]], \
                self.features[self.feature_index["right{}_rel_y".format(ii+1)]] = self.get_rel_xy(cr_scenario, vehicle_id, veh_id, startframe)

            else:
                self.features[self.feature_index["right{}_veh_id".format(ii+1)]] = -100
                self.features[self.feature_index["right{}_rel_x".format(ii+1)]] = -100
                self.features[self.feature_index["right{}_rel_y".format(ii+1)]] = -100
        return self.features


class NeighborFeatureExtractor1:
    def __init__(self):
        self.num_features = 6 * 8
        self.features = np.zeros(self.num_features, dtype=np.float32)
        self.feature_info = {}
        self.feature_names = feature_names = ["_veh_id", "_dist", "_offset", "_dist_ins", "_offset_ins", "_ori_ins"]
        feature_ranges = [{"high":2000,"low":1},
                          {"high":250.0, "low":-100.0},
                          {"high":4.0, "low":-4.0},
                          {"high":250.0, "low":-100.0},
                          {"high":250.0, "low":-100.0},
                          {"high":np.pi, "low":-np.pi}]
        self.neighbor_names = neighbor_names = ["front1", "left1", "left2", "left3", "right1", "right2", "right3", "rear1"]
        self.feature_index = {}
        index = 0
        for nn in neighbor_names:
            for i in range(len(feature_names)):
                self.feature_info[nn+feature_names[i]] = feature_ranges[i]
                self.feature_index[nn+feature_names[i]] = index
                index += 1
    def set_step_length_and_maneuver_horizon(self, step_length, horizon):
        self.step_length = step_length
        self.horizon = horizon
    def get_features(self, cr_scenario, vehicle_id, startframe):
        self.features = np.zeros(self.num_features, dtype=np.float32)
        ret = cr_scenario.get_attribute(NEIGHBOR_FORE_REAR_LEFT_MID_RIGHT, vehicle_id, startframe)
        # neighbors features: rel_pos, vel, length, width, rel_heading
        fore_M, fore_Mdist, fore_Moffset, fore_Mdist2, fore_Moffset2, fore_Mori2 = ret["front"]
        veh_id = fore_M
        if veh_id is not None:
            self.features[self.feature_index["front1_veh_id"]] = veh_id
            self.features[self.feature_index["front1_dist"]] = fore_Mdist
            self.features[self.feature_index["front1_offset"]] = fore_Moffset
            self.features[self.feature_index["front1_dist_ins"]] = fore_Mdist2
            self.features[self.feature_index["front1_offset_ins"]] = fore_Moffset2
            self.features[self.feature_index["front1_ori_ins"]] = fore_Mori2
        else:
            self.features[self.feature_index["front1_veh_id"]] = -100
            self.features[self.feature_index["front1_dist"]] = -100
            self.features[self.feature_index["front1_offset"]] = -100
            self.features[self.feature_index["front1_dist_ins"]] = -100
            self.features[self.feature_index["front1_offset_ins"]] = -100
            self.features[self.feature_index["front1_ori_ins"]] = -100

        rear_M, rear_Mdist, rear_Moffset, rear_Mdist2, rear_Moffset2, rear_Mori2 = ret["rear"]
        veh_id = rear_M
        if veh_id is not None:
            self.features[self.feature_index["rear1_veh_id"]] = veh_id
            self.features[self.feature_index["rear1_dist"]] = rear_Mdist
            self.features[self.feature_index["rear1_offset"]] = rear_Moffset
            self.features[self.feature_index["rear1_dist_ins"]] = rear_Mdist2
            self.features[self.feature_index["rear1_offset_ins"]] = rear_Moffset2
            self.features[self.feature_index["rear1_ori_ins"]] = rear_Mori2
        else:
            self.features[self.feature_index["rear1_veh_id"]] = -100
            self.features[self.feature_index["rear1_dist"]] = -100
            self.features[self.feature_index["rear1_offset"]] = -100
            self.features[self.feature_index["rear1_dist_ins"]] = -100
            self.features[self.feature_index["rear1_offset_ins"]] = -100
            self.features[self.feature_index["rear1_ori_ins"]] = -100

        Ls, Ldists, Loffsets, Ldists2, Loffsets2, Loris2 = ret["left"]
        for ii in range(len(Ls)):
            veh_id = Ls[ii]
            if veh_id is not None:
                self.features[self.feature_index["left{}_veh_id".format(ii+1)]] = veh_id
                self.features[self.feature_index["left{}_dist".format(ii+1)]] = Ldists[ii]
                self.features[self.feature_index["left{}_offset".format(ii+1)]] = Loffsets[ii]
                self.features[self.feature_index["left{}_dist_ins".format(ii+1)]] = Ldists2[ii]
                self.features[self.feature_index["left{}_offset_ins".format(ii+1)]] = Loffsets2[ii]
                self.features[self.feature_index["left{}_ori_ins".format(ii+1)]] = Loris2[ii]
            else:
                self.features[self.feature_index["left{}_veh_id".format(ii+1)]] = -100
                self.features[self.feature_index["left{}_dist".format(ii+1)]] = -100
                self.features[self.feature_index["left{}_offset".format(ii+1)]] = -100
                self.features[self.feature_index["left{}_dist_ins".format(ii+1)]] = -100
                self.features[self.feature_index["left{}_offset_ins".format(ii+1)]] = -100
                self.features[self.feature_index["left{}_ori_ins".format(ii+1)]] = -100
        Rs, Rdists, Roffsets, Rdists2, Roffsets2, Roris2 = ret["right"]
        for ii in range(len(Rs)):
            veh_id = Rs[ii]
            if veh_id is not None:
                self.features[self.feature_index["right{}_veh_id".format(ii+1)]] = veh_id
                self.features[self.feature_index["right{}_dist".format(ii+1)]] = Rdists[ii]
                self.features[self.feature_index["right{}_offset".format(ii+1)]] = Roffsets[ii]
                self.features[self.feature_index["right{}_dist_ins".format(ii+1)]] = Rdists2[ii]
                self.features[self.feature_index["right{}_offset_ins".format(ii+1)]] = Roffsets2[ii]
                self.features[self.feature_index["right{}_ori_ins".format(ii+1)]] = Roris2[ii]

            else:
                self.features[self.feature_index["right{}_veh_id".format(ii+1)]] = -100
                self.features[self.feature_index["right{}_dist".format(ii+1)]] = -100
                self.features[self.feature_index["right{}_offset".format(ii+1)]] = -100
                self.features[self.feature_index["right{}_dist_ins".format(ii+1)]] = -100
                self.features[self.feature_index["right{}_offset_ins".format(ii+1)]] = -100
                self.features[self.feature_index["right{}_ori_ins".format(ii+1)]] = -100
        return self.features

class VehicleToLaneletFeatureExtractor:
    def __init__(self, config):
        self.speed_limit = config["speed_limit"]
        self.num_features = 2
        self.feature_info = {"Current":{"high":1, "low":0},
                             "Target":{"high":1, "low":0}
                             }
        self.feature_index = {"Current":0,
                              "Target":1}
    #def set_range_and_horizon(self, max_s, horizon):
    #    self.max_s = max_s
    #    self.horizon = horizon
    def locate_on_grid(self, cr_scenario, nodes, vehicle_id, frame, initial_cands=None):
        ego_vehicle = cr_scenario.obstacles[frame][vehicle_id]
        ego_posG = ego_vehicle.initial_state.posG
        best_dist2 = float('inf')
        best_projection = None
        if initial_cands is None:
            for node_ind in nodes:
                pos = nodes[node_ind]
                dist2 = (pos.x-ego_posG.x)**2 + (pos.y-ego_posG.y)**2
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best_projection = node_ind
        else:
            for node_ind in initial_cands:
                pos = nodes[node_ind]
                dist2 = (pos.x-ego_posG.x)**2 + (pos.y-ego_posG.y)**2
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best_projection = node_ind
        return best_projection
    def adj_nodes(self, cr_scenario, node_ind, nodes, edges):
        children = []
        s = []
        cur_curvePt, cur_lanelet, cur_curve = get_curvePt_from_index(cr_scenario.lanelet_network, node_ind)
        for left in edges[2][node_ind]:
            left_curvePt,_,left_curve = get_curvePt_from_index(cr_scenario.lanelet_network, left)
            children.append(left)
            if left[1] == cur_lanelet.adj_left:
                left_posF = Frenet(cur_curvePt.pos.projF(cr_scenario.lanelet_network,
                                                          [left[1]]
                                                          ),
                                    cr_scenario.lanelet_network
                )
                adj_left_curvePt,_,_ = get_curvePt_from_index(cr_scenario.lanelet_network,left_posF.ind)
                s.append(left_curvePt.s - adj_left_curvePt.s)
            else:
                cur_left_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(cur_lanelet.adj_left)
                left_posF = Frenet(cur_curvePt.pos.projF(cr_scenario.lanelet_network,
                                                          [cur_lanelet.adj_left]
                                                          ),
                                    cr_scenario.lanelet_network
                )
                adj_left_curvePt,_,adj_left_curve = get_curvePt_from_index(cr_scenario.lanelet_network,left_posF.ind)

                if left[1] in cur_left_lanelet.successor:
                    s.append(left_curvePt.s+(adj_left_curve[-1].s-adj_left_curvePt.s))
                else:
                    assert(left[1] in cur_left_lanelet.predecessor)
                    s.append(-adj_left_curvePt.s-(left_curve[-1].s-left_curvePt.s))

        for right in edges[3][node_ind]:
            right_curvePt,_,right_curve = get_curvePt_from_index(cr_scenario.lanelet_network, right)
            children.append(right)
            if right[1] == cur_lanelet.adj_right:
                right_posF = Frenet(cur_curvePt.pos.projF(cr_scenario.lanelet_network,
                                                          [right[1]]
                                                          ),
                                    cr_scenario.lanelet_network
                )
                adj_right_curvePt,_,_ = get_curvePt_from_index(cr_scenario.lanelet_network,right_posF.ind)
                s.append(right_curvePt.s - adj_right_curvePt.s)
            else:
                cur_right_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(cur_lanelet.adj_right)
                right_posF = Frenet(cur_curvePt.pos.projF(cr_scenario.lanelet_network,
                                                          [cur_lanelet.adj_right]
                                                          ),
                                    cr_scenario.lanelet_network
                )
                adj_right_curvePt,_,adj_right_curve = get_curvePt_from_index(cr_scenario.lanelet_network,right_posF.ind)

                if right[1] in cur_right_lanelet.successor:
                    s.append(right_curvePt.s+(adj_right_curve[-1].s-adj_right_curvePt.s))
                else:
                    assert(right[1] in cur_right_lanelet.predecessor)
                    s.append(-adj_right_curvePt.s-(right_curve[-1].s-right_curvePt.s))

        for front in edges[0][node_ind]:
            front_curvePt,front_lane,_ = get_curvePt_from_index(cr_scenario.lanelet_network, front)
            children.append(front)
            if front[1] == node_ind[1]:
                s.append(front_curvePt.s - cur_curvePt.s)
            else:
                #print(front[1], front_lane.successor,front_lane.predecessor, cur_lanelet, cur_lanelet.successor, cur_lanelet.predecessor)
                #assert(front[1] in cur_lanelet.successor)
                if front[1] in cur_lanelet.successor:
                    s.append(front_curvePt.s + (cur_curve[-1].s-cur_curvePt.s))
                else:
                    front_posF = Frenet(cur_curvePt.pos.projF(cr_scenario.lanelet_network,
                                                              [front[1]]
                                                              ),
                                        cr_scenario.lanelet_network
                    )
                    adj_front_curvePt,_,_ = get_curvePt_from_index(cr_scenario.lanelet_network,front_posF.ind)
                    s.append(front_curvePt.s - adj_front_curvePt.s)
            #assert(s[-1] >= 0.0)
        return iter(zip(children, s))
    def reachable(self, s):
        #pos = nodes[node_ind]
        #if (pos.x - self.veh_cen_pos.x)**2+(pos.y - self.veh_cen_pos.y)**2 < self.veh_radius**2:
        #    continue
        #else:
        #    return False
        if s <= self.s_max and s >= self.s_min:
            return True
        else:
            return False

    def visited(self, visited_grids, node_idxs, grid_ind):
        grid_index = node_idxs[grid_ind]
        for vn in visited_grids:
            if grid_index == node_idxs[vn]:
                return True
        return False

    def furthest_longitudinal_reach(self, t, v0, v_s, v_max, a_max):
        if v0 < v_s and v0 < v_max:
            t1 = min((v_s - v0)/a_max, t)
        else:
            t1 = 0.
        v1 = a_max * t1 + v0
        if v1 >= v_s and v1 < v_max:
            t2 = min((v_max**2-v1**2)/(2*a_max*v_s), (t-t1))
        else:
            t2 = 0.
        t3 = t - t1 - t2
        x1 = (0.5*a_max*t1**2) + (v0*t1)
        if t2 > 0:
            x2 = (((2 * a_max * v_s * t2) + v1**2)**(3/2) - v1**3) / (3 * a_max * v_s)
        else:
            x2 = 0.
        x3 = v_max * t3
        x = x1 + x2 + x3
        return x

    def get_features(self, cr_scenario, nodes, node_idxs, edges, vehicle_id, startframe, endframe):
        ego_vehicle = cr_scenario.obstacles[startframe][vehicle_id]
        ego_posG = ego_vehicle.initial_state.posG
        ego_posF = ego_vehicle.initial_state.posF
        ego_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(ego_posF.ind[1])
        ego_curve = ego_lanelet.center_curve
        self.s_min = 0.0
        self.s_max = self.furthest_longitudinal_reach( (endframe-startframe)*cr_scenario.dt,
                                                  ego_vehicle.initial_state.velocity,
                                                  self.speed_limit,
                                                  40., 2.5)

        start_grid = self.locate_on_grid(cr_scenario, nodes, vehicle_id, startframe)
        ego_curvePt = lerp_curve(ego_curve[ego_posF.ind[0].i], ego_curve[ego_posF.ind[0].i+1], ego_posF.ind[0].t)
        reachable_grids = [start_grid]
        visited_grids = set()
        visited_grids.add(start_grid)
        stack = [(start_grid, 0., self.adj_nodes(cr_scenario, start_grid, nodes, edges))]
        while stack:
            cur_node, s, children = stack[-1]
            try:
                child, delta_s = next(children)
                if self.reachable(s+delta_s) and not self.visited(visited_grids, node_idxs, child):
                    visited_grids.add(child)
                    stack.append((child, s+delta_s, self.adj_nodes(cr_scenario, child, nodes, edges)))
                    reachable_grids.append(child)
            except StopIteration:
                stack.pop()

        if vehicle_id in cr_scenario.obstacles[endframe].keys():
            target_grid = self.locate_on_grid(cr_scenario, nodes, vehicle_id, endframe, reachable_grids)
        else:
            target_grid = self.locate_on_grid(cr_scenario, nodes, vehicle_id, startframe, reachable_grids)


        final_features = {}
        for grid_ind in reachable_grids:
            final_features[node_idxs[grid_ind]] = {}
            if node_idxs[start_grid] == node_idxs[grid_ind]:
                final_features[node_idxs[grid_ind]]["current"] = 1
            else:
                final_features[node_idxs[grid_ind]]["current"] = 0
            if node_idxs[target_grid] == node_idxs[grid_ind]:
                final_features[node_idxs[grid_ind]]["target"] = 1
            else:
                final_features[node_idxs[grid_ind]]["target"] = 0

        return final_features

class VehicleToLaneletFeatureExtractor2:
    def __init__(self):
        self.num_features = 10
        self.feature_info = {"isCurrent":{"high":1, "low":0},
                             "isTarget":{"high":1, "low":0},
                             "xs":{"high":40., "low":-40.},
                             "ys":{"high":40., "low":-40.},}
        self.feature_index = {"isCurrent":0,
                              "isTarget":1,
                              "xs":[2,3,4,5],
                              "ys":[6,7,8,9]}
    def set_range_and_horizon(self, max_s, horizon):
        self.max_s = max_s
        self.horizon = horizon
    def get_features(self, cr_scenario, vehicle_id, startframe):
        feature_inds = []
        ego_vehicle = cr_scenario.obstacles[startframe][vehicle_id]
        ego_posG = ego_vehicle.initial_state.posG
        delta_step = int(self.horizon/cr_scenario.dt)
        if vehicle_id in cr_scenario.obstacles[startframe-delta_step//3]:
            ego_posG1 = cr_scenario.obstacles[startframe-delta_step//3][vehicle_id].initial_state.posG
        else:
            ego_posG1 = ego_posG
        if vehicle_id in cr_scenario.obstacles[startframe-delta_step//3*2]:
            ego_posG2 = cr_scenario.obstacles[startframe-delta_step//3*2][vehicle_id].initial_state.posG
        else:
            ego_posG2 = ego_posG1
        ego_posF = ego_vehicle.initial_state.posF
        ego_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(ego_posF.ind[1])
        ego_curve = ego_lanelet.center_curve
        current_index = cr_scenario.locate_on_grid1(startframe, vehicle_id)
        ego_curvePt = lerp_curve(ego_curve[ego_posF.ind[0].i], ego_curve[ego_posF.ind[0].i+1], ego_posF.ind[0].t)
        lanelets_from_to_index = {}
        if ego_curvePt.s + self.max_s > ego_curve[-1].s:
            lanelets_from_to_index[ego_lanelet.lanelet_id] = (ego_posF.ind[0], CurveIndex(len(ego_curve)-2,1.0))
            for succ in ego_lanelet.successor:
                succ_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ)
                succ_curve = succ_lanelet.center_curve
                if ego_curvePt.s +self.max_s > ego_curve[-1].s + succ_curve[-1].s:
                    lanelets_from_to_index[succ] = (CurveIndex(0,0.0), CurveIndex(len(succ_curve)-2,1.0))
                    if ego_lanelet.adj_left is None and succ_lanelet.adj_left is not None:
                        succ_left_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ_lanelet.adj_left)
                        succ_left_curve = succ_left_lanelet.center_curve
                        lanelets_from_to_index[succ_lanelet.adj_left] = (CurveIndex(0,0.0), CurveIndex(len(succ_left_curve)-2,1.0))
                    if ego_lanelet.adj_right is None and succ_lanelet.adj_right is not None:
                        succ_right_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ_lanelet.adj_right)
                        succ_right_curve = succ_right_lanelet.center_curve
                        lanelets_from_to_index[succ_lanelet.adj_right] = (CurveIndex(0,0.0), CurveIndex(len(succ_right_curve)-2,1.0))

                    for succ_succ in succ_lanelet.successor:
                        succ_succ_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ_succ)
                        succ_succ_curve = succ_succ_lanelet.center_curve
                        to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_succ_curve, ego_curvePt.s+self.max_s-ego_curve[-1].s-succ_curve[-1].s)
                        lanelets_from_to_index[succ_succ] = (CurveIndex(0,0.0), to_index)
                        if succ_succ_lanelet.adj_left is not None:
                            succ_succ_left_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ_succ_lanelet.adj_left)
                            succ_succ_left_curve = succ_succ_left_lanelet.center_curve
                            to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_succ_left_curve, ego_curvePt.s+self.max_s-ego_curve[-1].s-succ_curve[-1].s)
                            lanelets_from_to_index[succ_succ_lanelet.adj_left] = (CurveIndex(0,0.0), to_index)
                        if succ_succ_lanelet.adj_right is not None:
                            succ_succ_right_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ_succ_lanelet.adj_right)
                            succ_succ_right_curve = succ_succ_right_lanelet.center_curve
                            to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_succ_right_curve, ego_curvePt.s+self.max_s-ego_curve[-1].s-succ_curve[-1].s)
                            lanelets_from_to_index[succ_succ_lanelet.adj_right] = (CurveIndex(0,0.0), to_index)
                else:
                    to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_curve, ego_curvePt.s+self.max_s-ego_curve[-1].s)
                    lanelets_from_to_index[succ] = (CurveIndex(0,0.0), to_index)
                    if ego_lanelet.adj_left is None and succ_lanelet.adj_left is not None:
                        succ_left_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ_lanelet.adj_left)
                        succ_left_curve = succ_left_lanelet.center_curve
                        to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_left_curve, ego_curvePt.s+self.max_s-ego_curve[-1].s)
                        lanelets_from_to_index[succ_lanelet.adj_left] = (CurveIndex(0,0.0), to_index)
                    if ego_lanelet.adj_right is None and succ_lanelet.adj_right is not None:
                        succ_right_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ_lanelet.adj_right)
                        succ_right_curve = succ_right_lanelet.center_curve
                        to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_right_curve, ego_curvePt.s+self.max_s-ego_curve[-1].s)
                        lanelets_from_to_index[succ_lanelet.adj_right] = (CurveIndex(0,0.0), to_index)
        else:
            to_index, _ = get_curve_index(ego_posF.ind[0], ego_curve, self.max_s)
            lanelets_from_to_index[ego_lanelet.lanelet_id] = (ego_posF.ind[0], to_index)
        if ego_lanelet.adj_left is not None:
            ego_left_posF = Frenet(ego_posG.projF(cr_scenario.lanelet_network, [ego_lanelet.adj_left]), cr_scenario.lanelet_network)
            ego_left_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(ego_left_posF.ind[1])
            ego_left_curve = ego_left_lanelet.center_curve
            ego_left_curvePt = lerp_curve(ego_left_curve[ego_left_posF.ind[0].i], ego_left_curve[ego_left_posF.ind[0].i+1], ego_left_posF.ind[0].t)
            if ego_left_curvePt.s + self.max_s > ego_left_curve[-1].s:
                lanelets_from_to_index[ego_left_lanelet.lanelet_id] = (ego_left_posF.ind[0], CurveIndex(len(ego_left_curve)-2,1.0))
                for succ in ego_left_lanelet.successor:
                    succ_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ)
                    succ_curve = succ_lanelet.center_curve
                    if ego_left_curvePt.s +self.max_s > ego_left_curve[-1].s + succ_curve[-1].s:
                        lanelets_from_to_index[succ] = (CurveIndex(0,0.0), CurveIndex(len(succ_curve)-2,1.0))
                        for succ_succ in succ_lanelet.successor:
                            succ_succ_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ_succ)
                            succ_succ_curve = succ_succ_lanelet.center_curve
                            to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_succ_curve, ego_left_curvePt.s+self.max_s-ego_left_curve[-1].s-succ_curve[-1].s)
                            lanelets_from_to_index[succ_succ] = (CurveIndex(0,0.0), to_index)
                    else:
                        to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_curve, ego_left_curvePt.s+self.max_s-ego_left_curve[-1].s)
                        lanelets_from_to_index[succ] = (CurveIndex(0,0.0), to_index)
            else:
                to_index, _ = get_curve_index(ego_left_posF.ind[0], ego_left_curve, self.max_s)
                lanelets_from_to_index[ego_left_lanelet.lanelet_id] = (ego_left_posF.ind[0], to_index)
        if ego_lanelet.adj_right is not None:
            ego_right_posF = Frenet(ego_posG.projF(cr_scenario.lanelet_network, [ego_lanelet.adj_right]), cr_scenario.lanelet_network)
            ego_right_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(ego_right_posF.ind[1])
            ego_right_curve = ego_right_lanelet.center_curve
            ego_right_curvePt = lerp_curve(ego_right_curve[ego_right_posF.ind[0].i], ego_right_curve[ego_right_posF.ind[0].i+1], ego_right_posF.ind[0].t)
            if ego_right_curvePt.s + self.max_s > ego_right_curve[-1].s:
                lanelets_from_to_index[ego_right_lanelet.lanelet_id] = (ego_right_posF.ind[0], CurveIndex(len(ego_right_curve)-2,1.0))
                for succ in ego_right_lanelet.successor:
                    succ_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ)
                    succ_curve = succ_lanelet.center_curve
                    if ego_right_curvePt.s +self.max_s > ego_right_curve[-1].s + succ_curve[-1].s:
                        lanelets_from_to_index[succ] = (CurveIndex(0,0.0), CurveIndex(len(succ_curve)-2,1.0))
                        for succ_succ in succ_lanelet.successor:
                            succ_succ_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ_succ)
                            succ_succ_curve = succ_succ_lanelet.center_curve
                            to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_succ_curve, ego_right_curvePt.s+self.max_s-ego_right_curve[-1].s-succ_curve[-1].s)
                            lanelets_from_to_index[succ_succ] = (CurveIndex(0,0.0), to_index)
                    else:
                        to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_curve, ego_right_curvePt.s+self.max_s-ego_right_curve[-1].s)
                        lanelets_from_to_index[succ] = (CurveIndex(0,0.0), to_index)
            else:
                to_index, _ = get_curve_index(ego_right_posF.ind[0], ego_right_curve, self.max_s)
                lanelets_from_to_index[ego_right_lanelet.lanelet_id] = (ego_right_posF.ind[0], to_index)

        for index in range(len(cr_scenario.grids)):
            grid = cr_scenario.grids[index]
            if index == current_index:
                feature_inds.append(index)
                continue
            for ind in grid.ind_list:
                if ind[1] in lanelets_from_to_index:
                    from_index, to_index = lanelets_from_to_index[ind[1]]
                    if from_index.i < ind[0].i or (from_index.i==ind[0].i and from_index.t < ind[0].t):
                        if to_index.i > ind[0].i or (to_index.i==ind[0].i and to_index.t > ind[0].t):
                            feature_inds.append(index)
                            break

        horizon_step = int(self.horizon/cr_scenario.dt)
        if vehicle_id in cr_scenario.obstacles[startframe+horizon_step].keys():
            target_index = cr_scenario.locate_on_grid1(startframe+horizon_step, vehicle_id, initial_cands=feature_inds)
        else:
            target_index = cr_scenario.locate_on_grid1(startframe, vehicle_id, initial_cands=feature_inds)


        assert current_index in feature_inds
        assert target_index in feature_inds
        final_features = {}
        for feature_ind in feature_inds:
            final_features[feature_ind] = {}
            if current_index == feature_ind:
                final_features[feature_ind]["current"] = 1
            else:
                final_features[feature_ind]["current"] = 0
            if target_index == feature_ind:
                final_features[feature_ind]["target"] = 1
            else:
                final_features[feature_ind]["target"] = 0
            points = []
            for pos in cr_scenario.grids[feature_ind].pos_list:
                pproj0 = pos.inertial2body(ego_posG)
                pproj1 = pos.inertial2body(ego_posG1)
                pproj2 = pos.inertial2body(ego_posG2)

                points.append((pproj0.x, pproj0.y, pproj0.th, pproj1.x, pproj1.y, pproj1.th, pproj2.x, pproj2.y, pproj2.th))

            final_features[feature_ind]["point"] = points
        return final_features

class VehicleToLaneletFeatureExtractor1:
    def __init__(self):
        self.num_features = 10
        self.feature_info = {"isCurrent":{"high":1, "low":0},
                             "isTarget":{"high":1, "low":0},
                             "xs":{"high":40., "low":-40.},
                             "ys":{"high":40., "low":-40.},}
        self.feature_index = {"isCurrent":0,
                              "isTarget":1,
                              "xs":[2,3,4,5],
                              "ys":[6,7,8,9]}
    def set_range_and_horizon(self, max_s, horizon):
        self.max_s = max_s
        self.horizon = horizon
    def get_features(self, cr_scenario, vehicle_id, startframe):
        feature_inds = []
        ego_vehicle = cr_scenario.obstacles[startframe][vehicle_id]
        ego_posG = ego_vehicle.initial_state.posG
        ego_posF = ego_vehicle.initial_state.posF
        ego_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(ego_posF.ind[1])
        ego_curve = ego_lanelet.center_curve
        current_index = cr_scenario.locate_on_grid1(startframe, vehicle_id)
        ego_curvePt = lerp_curve(ego_curve[ego_posF.ind[0].i], ego_curve[ego_posF.ind[0].i+1], ego_posF.ind[0].t)
        lanelets_from_to_index = {}
        if ego_curvePt.s + self.max_s > ego_curve[-1].s:
            lanelets_from_to_index[ego_lanelet.lanelet_id] = (ego_posF.ind[0], CurveIndex(len(ego_curve)-2,1.0))
            for succ in ego_lanelet.successor:
                succ_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ)
                succ_curve = succ_lanelet.center_curve
                if ego_curvePt.s +self.max_s > ego_curve[-1].s + succ_curve[-1].s:
                    lanelets_from_to_index[succ] = (CurveIndex(0,0.0), CurveIndex(len(succ_curve)-2,1.0))
                    if ego_lanelet.adj_left is None and succ_lanelet.adj_left is not None:
                        succ_left_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ_lanelet.adj_left)
                        succ_left_curve = succ_left_lanelet.center_curve
                        lanelets_from_to_index[succ_lanelet.adj_left] = (CurveIndex(0,0.0), CurveIndex(len(succ_left_curve)-2,1.0))
                    if ego_lanelet.adj_right is None and succ_lanelet.adj_right is not None:
                        succ_right_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ_lanelet.adj_right)
                        succ_right_curve = succ_right_lanelet.center_curve
                        lanelets_from_to_index[succ_lanelet.adj_right] = (CurveIndex(0,0.0), CurveIndex(len(succ_right_curve)-2,1.0))

                    for succ_succ in succ_lanelet.successor:
                        succ_succ_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ_succ)
                        succ_succ_curve = succ_succ_lanelet.center_curve
                        to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_succ_curve, ego_curvePt.s+self.max_s-ego_curve[-1].s-succ_curve[-1].s)
                        lanelets_from_to_index[succ_succ] = (CurveIndex(0,0.0), to_index)
                        if succ_succ_lanelet.adj_left is not None:
                            succ_succ_left_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ_succ_lanelet.adj_left)
                            succ_succ_left_curve = succ_succ_left_lanelet.center_curve
                            to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_succ_left_curve, ego_curvePt.s+self.max_s-ego_curve[-1].s-succ_curve[-1].s)
                            lanelets_from_to_index[succ_succ_lanelet.adj_left] = (CurveIndex(0,0.0), to_index)
                        if succ_succ_lanelet.adj_right is not None:
                            succ_succ_right_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ_succ_lanelet.adj_right)
                            succ_succ_right_curve = succ_succ_right_lanelet.center_curve
                            to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_succ_right_curve, ego_curvePt.s+self.max_s-ego_curve[-1].s-succ_curve[-1].s)
                            lanelets_from_to_index[succ_succ_lanelet.adj_right] = (CurveIndex(0,0.0), to_index)
                else:
                    to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_curve, ego_curvePt.s+self.max_s-ego_curve[-1].s)
                    lanelets_from_to_index[succ] = (CurveIndex(0,0.0), to_index)
                    if ego_lanelet.adj_left is None and succ_lanelet.adj_left is not None:
                        succ_left_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ_lanelet.adj_left)
                        succ_left_curve = succ_left_lanelet.center_curve
                        to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_left_curve, ego_curvePt.s+self.max_s-ego_curve[-1].s)
                        lanelets_from_to_index[succ_lanelet.adj_left] = (CurveIndex(0,0.0), to_index)
                    if ego_lanelet.adj_right is None and succ_lanelet.adj_right is not None:
                        succ_right_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ_lanelet.adj_right)
                        succ_right_curve = succ_right_lanelet.center_curve
                        to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_right_curve, ego_curvePt.s+self.max_s-ego_curve[-1].s)
                        lanelets_from_to_index[succ_lanelet.adj_right] = (CurveIndex(0,0.0), to_index)
        else:
            to_index, _ = get_curve_index(ego_posF.ind[0], ego_curve, self.max_s)
            lanelets_from_to_index[ego_lanelet.lanelet_id] = (ego_posF.ind[0], to_index)
        if ego_lanelet.adj_left is not None:
            ego_left_posF = Frenet(ego_posG.projF(cr_scenario.lanelet_network, [ego_lanelet.adj_left]), cr_scenario.lanelet_network)
            ego_left_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(ego_left_posF.ind[1])
            ego_left_curve = ego_left_lanelet.center_curve
            ego_left_curvePt = lerp_curve(ego_left_curve[ego_left_posF.ind[0].i], ego_left_curve[ego_left_posF.ind[0].i+1], ego_left_posF.ind[0].t)
            if ego_left_curvePt.s + self.max_s > ego_left_curve[-1].s:
                lanelets_from_to_index[ego_left_lanelet.lanelet_id] = (ego_left_posF.ind[0], CurveIndex(len(ego_left_curve)-2,1.0))
                for succ in ego_left_lanelet.successor:
                    succ_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ)
                    succ_curve = succ_lanelet.center_curve
                    if ego_left_curvePt.s +self.max_s > ego_left_curve[-1].s + succ_curve[-1].s:
                        lanelets_from_to_index[succ] = (CurveIndex(0,0.0), CurveIndex(len(succ_curve)-2,1.0))
                        for succ_succ in succ_lanelet.successor:
                            succ_succ_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ_succ)
                            succ_succ_curve = succ_succ_lanelet.center_curve
                            to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_succ_curve, ego_left_curvePt.s+self.max_s-ego_left_curve[-1].s-succ_curve[-1].s)
                            lanelets_from_to_index[succ_succ] = (CurveIndex(0,0.0), to_index)
                    else:
                        to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_curve, ego_left_curvePt.s+self.max_s-ego_left_curve[-1].s)
                        lanelets_from_to_index[succ] = (CurveIndex(0,0.0), to_index)
            else:
                to_index, _ = get_curve_index(ego_left_posF.ind[0], ego_left_curve, self.max_s)
                lanelets_from_to_index[ego_left_lanelet.lanelet_id] = (ego_left_posF.ind[0], to_index)
        if ego_lanelet.adj_right is not None:
            ego_right_posF = Frenet(ego_posG.projF(cr_scenario.lanelet_network, [ego_lanelet.adj_right]), cr_scenario.lanelet_network)
            ego_right_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(ego_right_posF.ind[1])
            ego_right_curve = ego_right_lanelet.center_curve
            ego_right_curvePt = lerp_curve(ego_right_curve[ego_right_posF.ind[0].i], ego_right_curve[ego_right_posF.ind[0].i+1], ego_right_posF.ind[0].t)
            if ego_right_curvePt.s + self.max_s > ego_right_curve[-1].s:
                lanelets_from_to_index[ego_right_lanelet.lanelet_id] = (ego_right_posF.ind[0], CurveIndex(len(ego_right_curve)-2,1.0))
                for succ in ego_right_lanelet.successor:
                    succ_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ)
                    succ_curve = succ_lanelet.center_curve
                    if ego_right_curvePt.s +self.max_s > ego_right_curve[-1].s + succ_curve[-1].s:
                        lanelets_from_to_index[succ] = (CurveIndex(0,0.0), CurveIndex(len(succ_curve)-2,1.0))
                        for succ_succ in succ_lanelet.successor:
                            succ_succ_lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(succ_succ)
                            succ_succ_curve = succ_succ_lanelet.center_curve
                            to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_succ_curve, ego_right_curvePt.s+self.max_s-ego_right_curve[-1].s-succ_curve[-1].s)
                            lanelets_from_to_index[succ_succ] = (CurveIndex(0,0.0), to_index)
                    else:
                        to_index, _ = get_curve_index(CurveIndex(0,0.0), succ_curve, ego_right_curvePt.s+self.max_s-ego_right_curve[-1].s)
                        lanelets_from_to_index[succ] = (CurveIndex(0,0.0), to_index)
            else:
                to_index, _ = get_curve_index(ego_right_posF.ind[0], ego_right_curve, self.max_s)
                lanelets_from_to_index[ego_right_lanelet.lanelet_id] = (ego_right_posF.ind[0], to_index)

        for index in range(len(cr_scenario.grids)):
            grid = cr_scenario.grids[index]
            if index == current_index:
                feature_inds.append(index)
                continue
            for ind in grid.ind_list:
                if ind[1] in lanelets_from_to_index:
                    from_index, to_index = lanelets_from_to_index[ind[1]]
                    if from_index.i < ind[0].i or (from_index.i==ind[0].i and from_index.t < ind[0].t):
                        if to_index.i > ind[0].i or (to_index.i==ind[0].i and to_index.t > ind[0].t):
                            feature_inds.append(index)
                            break

        horizon_step = int(self.horizon/cr_scenario.dt)
        if vehicle_id in cr_scenario.obstacles[startframe+horizon_step].keys():
            target_index = cr_scenario.locate_on_grid1(startframe+horizon_step, vehicle_id, initial_cands=feature_inds)
        else:
            target_index = cr_scenario.locate_on_grid1(startframe, vehicle_id, initial_cands=feature_inds)


        assert current_index in feature_inds
        assert target_index in feature_inds
        final_features = {}
        for feature_ind in feature_inds:
            final_features[feature_ind] = {}
            if current_index == feature_ind:
                final_features[feature_ind]["current"] = 1
            else:
                final_features[feature_ind]["current"] = 0
            if target_index == feature_ind:
                final_features[feature_ind]["target"] = 1
            else:
                final_features[feature_ind]["target"] = 0
            points = []
            for pos in cr_scenario.grids[feature_ind].pos_list:
                pproj = pos.inertial2body(ego_posG)
                points.append((pproj.x, pproj.y, pproj.th))

            final_features[feature_ind]["point"] = points
        return final_features

class LaneletToLaneletFeatureExtractor:
    def __init__(self):
        self.num_features = 5
        self.feature_info = {
                              "index":{"high":None, "low":None},
                              "relation" : {"high":1., "low":0.},
        }
        self.feature_index = {"index" : 0,
                              "relation" : [1,2,3,4],
        }

    def get_features(self, cr_scenario, grid_ind, indexs, edges, front_only=False):
        features = []
        for front in edges[0][grid_ind]:
            feature = np.zeros(self.num_features, dtype=np.float32)
            feature[self.feature_index["index"]] = indexs[front]
            feature[self.feature_index["relation"]] = [0,0,1,0]
            features.append(feature)
        if front_only:
            return features
        '''
        for back in edges[1][grid_ind]:
            feature = np.zeros(self.num_features, dtype=np.float32)
            feature[self.feature_index["index"]] = indexs[back]
            feature[self.feature_index["relation"]] = [0,0,0,0,1]
            features.append(feature)
        '''
        for left in edges[2][grid_ind]:
            feature = np.zeros(self.num_features, dtype=np.float32)
            feature[self.feature_index["index"]] = indexs[left]
            feature[self.feature_index["relation"]] = [1,0,0,0]
            features.append(feature)
        for right in edges[3][grid_ind]:
            feature = np.zeros(self.num_features, dtype=np.float32)
            feature[self.feature_index["index"]] = indexs[right]
            feature[self.feature_index["relation"]] = [0,1,0,0]
            features.append(feature)
        for conflict in edges[4][grid_ind]:
            feature = np.zeros(self.num_features, dtype=np.float32)
            feature[self.feature_index["index"]] = indexs[conflict]
            feature[self.feature_index["relation"]] = [0,0,0,1]
            features.append(feature)
        return features

class LaneletToLaneletFeatureExtractor1:
    def __init__(self):
        self.num_features = 5
        self.feature_info = {
                              "index":{"high":None, "low":None},
                              "relation" : {"high":1., "low":0.},
        }
        self.feature_index = {"index" : 0,
                              "relation" : [1,2,3,4],
        }
    def set_grid_length(self, grid_length):
        self.max_distance2 = (grid_length+2)**2
        self.grid_length = grid_length
    def get_features(self, cr_scenario, grid_index):
        features = []
        ego_grid = cr_scenario.grids[grid_index]
        for front in ego_grid.front_list:
            feature = np.zeros(self.num_features, dtype=np.float32)
            feature[self.feature_index["index"]] = front
            feature[self.feature_index["relation"]] = [0,0,1,0]
            features.append(feature)
        for left in ego_grid.left_list:
            feature = np.zeros(self.num_features, dtype=np.float32)
            feature[self.feature_index["index"]] = left
            feature[self.feature_index["relation"]] = [1,0,0,0]
            features.append(feature)
        for right in ego_grid.right_list:
            feature = np.zeros(self.num_features, dtype=np.float32)
            feature[self.feature_index["index"]] = right
            feature[self.feature_index["relation"]] = [0,1,0,0]
            features.append(feature)
        for conflict in ego_grid.conflict_list:
            feature = np.zeros(self.num_features, dtype=np.float32)
            feature[self.feature_index["index"]] = conflict
            feature[self.feature_index["relation"]] = [0,0,0,1]
            features.append(feature)
        return features

class LaneletFeatureExtractor:
    def __init__(self):
        self.num_features = 9
        self.features = np.zeros(self.num_features, dtype=np.float32)
        self.feature_info = {
                             "marker_left" : {"high": 1., "low":0.},
                             "marker_right" : {"high": 1., "low":0.},
                             "priority" : {"high":1., "low":0.},
                             "x": {"high":200, "low":-200},
                             "y": {"high":200, "low":-200}
                             }
        self.feature_index = {
                             "marker_left" : [0,1,2],
                             "marker_right" : [3,4,5],
                             "priority" : 6,
                             "x" : 7,
                             "y" : 8
                             }
    def get_rel_xy(self, cr_scenario, veh, grid_pos, startframe):
        veh_state = cr_scenario.obstacles[startframe][veh]
        grid_proj = grid_pos.inertial2body(veh_state.initial_state.posG)
        grid_x, grid_y = grid_proj.x, grid_proj.y
        return grid_x, grid_y

    def get_features(self, cr_scenario, grid_ind, grid_pos, vehicle_id, startframe):
        self.features = np.zeros(self.num_features, dtype=np.float32)
        lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(grid_ind[1])
        if lanelet.line_marking_left_vertices ==  LineMarking.DASHED:
            self.features[self.feature_index["marker_left"]] = [1,0,0]
        elif lanelet.line_marking_left_vertices ==  LineMarking.SOLID:
            self.features[self.feature_index["marker_left"]] = [0,1,0]
        elif lanelet.line_marking_left_vertices ==  LineMarking.NO_MARKING:
            self.features[self.feature_index["marker_left"]] = [0,0,1]

        if lanelet.line_marking_right_vertices ==  LineMarking.DASHED:
            self.features[self.feature_index["marker_right"]] = [1,0,0]
        elif lanelet.line_marking_right_vertices ==  LineMarking.SOLID:
            self.features[self.feature_index["marker_right"]] = [0,1,0]
        elif lanelet.line_marking_right_vertices ==  LineMarking.NO_MARKING:
            self.features[self.feature_index["marker_right"]] = [0,0,1]

        if  LaneletType.ACCESS_RAMP in lanelet.lanelet_type or LaneletType.EXIT_RAMP in lanelet.lanelet_type:
            self.features[self.feature_index["priority"]] = 0.0
        else:
            self.features[self.feature_index["priority"]] = 1.0

        self.features[self.feature_index["x"]], \
        self.features[self.feature_index["y"]] = self.get_rel_xy(cr_scenario, vehicle_id, grid_pos, startframe)
        return self.features

class LaneletFeatureExtractor1:
    def __init__(self):
        self.num_features = 7#15
        self.features = np.zeros(self.num_features, dtype=np.float32)
        self.feature_info = {
                             "marker_left" : {"high": 1., "low":0.},
                             "marker_right" : {"high": 1., "low":0.},
                             "priority" : {"high":1., "low":0.},
                             }
        self.feature_index = {

                             "marker_left" : [0,1,2],
                             "marker_right" : [3,4,5],
                             "priority" : 6,
                             }
    def get_features(self, cr_scenario, grid_index, vehicle_id, startframe):
        self.features = np.zeros(self.num_features, dtype=np.float32)
        grid = cr_scenario.grids[grid_index]
        marker_left_list = []
        marker_right_list = []
        priority_list = []
        for ind in grid.ind_list:
            lanelet = cr_scenario.lanelet_network.find_lanelet_by_id(ind[1])
            if lanelet.line_marking_left_vertices ==  LineMarking.DASHED:
                marker_left_list.append(1)
            elif lanelet.line_marking_left_vertices ==  LineMarking.SOLID:
                marker_left_list.append(2)
            elif lanelet.line_marking_left_vertices ==  LineMarking.NO_MARKING:
                marker_left_list.append(3)

            if lanelet.line_marking_right_vertices ==  LineMarking.DASHED:
                marker_right_list.append(1)
            elif lanelet.line_marking_right_vertices ==  LineMarking.SOLID:
                marker_right_list.append(2)
            elif lanelet.line_marking_right_vertices ==  LineMarking.NO_MARKING:
                marker_right_list.append(3)

            if  LaneletType.ACCESS_RAMP in lanelet.lanelet_type or LaneletType.EXIT_RAMP in lanelet.lanelet_type:
                priority_list.append(0)
            else:
                priority_list.append(1)
        try:
            marker_left = max(set(marker_left_list), key=marker_left_list.count)
        except:
            print(grid.ind_list, grid.pos_list, grid.id)
            raise
        marker_right = max(set(marker_right_list), key=marker_right_list.count)
        priority = max(set(priority_list), key=priority_list.count)
        if marker_left == 1:
            self.features[self.feature_index["marker_left"]] = [1,0,0]
        elif marker_left == 2:
            self.features[self.feature_index["marker_left"]] = [0,1,0]
        elif marker_left == 3:
            self.features[self.feature_index["marker_left"]] = [0,0,1]

        if marker_right == 1:
            self.features[self.feature_index["marker_right"]] = [1,0,0]
        elif marker_right == 2:
            self.features[self.feature_index["marker_right"]] = [0,1,0]
        elif marker_right == 3:
            self.features[self.feature_index["marker_right"]] = [0,0,1]

        self.features[self.feature_index["priority"]] = priority
        ###

        return self.features

class WellBehavedFeatureExtractor:
    def __init__(self):
        self.num_features = 5
        self.features = np.zeros(self.num_features, dtype=np.float32)
        self.feature_info = {"is_colliding" : {"high":1.,    "low":0.},
                             "out_of_lane"  : {"high":1.,    "low":0.},
                             "negative_velocity" : {"high":1., "low":0.},
                             "distance_road_edge_left" : {"high":50.,    "low":-50.},
                             "distance_road_edge_right" : {"high":50.,    "low":-50.}}
        self.feature_index = {"is_colliding" : 0,
                             "out_of_lane"  : 1,
                             "negative_velocity" : 2,
                             "distance_road_edge_left" : 3,
                             "distance_road_edge_right" : 4}

    def get_features(self, cr_scenario,  vehicle_id, startframe):
        ego_vehicle = cr_scenario.obstacles[startframe][vehicle_id]
        d_ml = cr_scenario.get_attribute(LINE_MARKING_LEFT_DIST, vehicle_id, startframe)
        d_mr = cr_scenario.get_attribute(LINE_MARKING_RIGHT_DIST, vehicle_id, startframe)
        self.features[0] = cr_scenario.get_attribute(IS_COLLIDING, vehicle_id, startframe)
        self.features[1] = d_ml < -0.8 or d_mr < -0.8
        self.features[2] = ego_vehicle.initial_state.velocity < 0.0
        self.features[3] = cr_scenario.get_attribute(ROAD_DIST_LEFT, vehicle_id, startframe)
        self.features[4] = cr_scenario.get_attribute(ROAD_DIST_RIGHT, vehicle_id, startframe)
        return self.features



class HistoryFeatureExtractor:
    def __init__(self):
        self.num_features = 5*5
        self.features = np.zeros(self.num_features, dtype=np.float32)
        self.feature_info = {}
        self.feature_names = feature_names = ["_relative_offset", "_relative_distance", "_velocity", "_relative_heading", "_time_step"]
        feature_ranges = [{"high":2.0, "low":-2.0}, {"high":40.0, "low":-40.0},
                          {"high":40.0, "low":0.0}, {"high":0.05, "low":-0.05},
                          {"high":5.0,"low":1.0}]
        self.history_names = history_names = ["1sec", "2sec", "3sec", "4sec", "5sec"]
        self.feature_index = {}
        index = 0
        for nn in history_names:
            for i in range(len(feature_names)):
                self.feature_info[nn+feature_names[i]] = feature_ranges[i]
                self.feature_index[nn+feature_names[i]] = index
                index += 1


    def get_features(self, cr_scenario,  vehicle_id, startframe):
        ego = cr_scenario.obstacles[startframe][vehicle_id]
        ego_lane_id = ego.initial_state.posF.ind[1]
        index = 0
        for t in range(1, 6):
            if vehicle_id in cr_scenario.obstacles[startframe - t*10].keys():
                past_veh = cr_scenario.obstacles[startframe-10*t][vehicle_id]
                past_lane_id = past_veh.initial_state.posF.ind[1]
                if ego_lane_id == past_lane_id:
                    self.features[index] = past_veh.initial_state.posF.d - ego.initial_state.posF.d
                    self.features[index+1] = past_veh.initial_state.posF.s - ego.initial_state.posF.s
                    self.features[index+2] = past_veh.initial_state.velocity
                    self.features[index+3] = past_veh.initial_state.posF.phi
                    self.features[index+4] = t*1.0
                else:
                    posG = past_veh.initial_state.get_posG()
                    posF_adjust = Frenet(posG.projF(cr_scenario.lanelet_network, [ego_lane_id]), cr_scenario.lanelet_network)
                    self.features[index] = posF_adjust.d - ego.initial_state.posF.d
                    self.features[index+1] = posF_adjust.s - ego.initial_state.posF.s
                    self.features[index+2] = past_veh.initial_state.velocity
                    self.features[index+3] = posF_adjust.phi
                    self.features[index+4] = t*1.0
            else:
                for i in range(1, 10):
                    if vehicle_id in cr_scenario.obstacles[startframe-(t-1)*10-(10-i)].keys():
                        past_veh = cr_scenario.obstacles[startframe-(t-1)*10-(10-i)][vehicle_id]
                        past_lane_id = past_veh.initial_state.posF.ind[1]
                        if ego_lane_id == past_lane_id:
                            self.features[index] = past_veh.initial_state.posF.d - ego.initial_state.posF.d
                            self.features[index+1] = past_veh.initial_state.posF.s - ego.initial_state.posF.s
                            self.features[index+2] = past_veh.initial_state.velocity
                            self.features[index+3] = past_veh.initial_state.posF.phi
                            self.features[index+4] = t-1 + (10-i)*0.1
                        else:
                            posG = past_veh.initial_state.get_posG()
                            posF_adjust = Frenet(posG.projF(cr_scenario.lanelet_network, [ego_lane_id]), cr_scenario.lanelet_network)
                            self.features[index] = posF_adjust.d - ego.initial_state.posF.d
                            self.features[index+1] = posF_adjust.s - ego.initial_state.posF.s
                            self.features[index+2] = past_veh.initial_state.velocity
                            self.features[index+3] = posF_adjust.phi
                            self.features[index+4] = t-1 + (10-i)*0.1
                        break
                for i in range(index, len(self.num_features)):
                    self.features[i] = -1
                break
            index += 5
        return self.features

class MultiFeatureExtractor:
    def __init__(self, extractors):

        self.extractors = extractors
        self.lengths = [subext.num_features for subext in self.extractors]
        self.num_features = np.sum(self.lengths)
        self.features = np.zeros(self.num_features, dtype=np.float32)
        self.feature_index = {}
        for index, subext in enumerate(self.extractors):
            for feat_name, sub_index in subext.feature_index.items():
                if index == 0:
                    self.feature_index[feat_name] = sub_index
                else:
                    self.feature_index[feat_name] = sub_index + int(np.sum(self.lengths[0:index]))
    def set_step_length_and_maneuver_horizon(self, step_length, horizon):
        for subext in self.extractors:
            try:
                subext.set_step_length_and_maneuver_horizon(step_length, horizon)
            except:
                pass
    def set_origin(self, cr_scenario, egoid, step):
        for subext in self.extractors:
            try:
                subext.set_origin(cr_scenario, egoid, step)
            except:
                pass

    def get_features(self, cr_scenario, vehicle_id, startframe):
        feature_index = 0
        for subext, feature_len in zip(self.extractors, self.lengths):
            stop = feature_index + feature_len
            self.features[feature_index:stop] = subext.get_features(cr_scenario, vehicle_id, startframe)
            feature_index += feature_len
        return self.features
    def get_features_by_name(self, features, names):
        ret = []
        for n in names:
            if n in self.feature_index.keys():
                try:
                    ret.extend(features[self.feature_index[n]])
                except:
                    ret.append(features[self.feature_index[n]])
        return ret
