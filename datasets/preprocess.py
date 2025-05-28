import torch
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
import math
import pickle
from models.smart.utils import wrap_angle
import os

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

"""
 This function returns the threshold for collision. If any of two circles' origins' distance between two vehicles is lower than the threshold, it is considered as they have a collision at that timestamp.
 w1, w2 is scalar value which represents the width of vehicle 1 and vehicle 2.
"""
def return_collision_threshold(w1, w2):
    return (w1 + w2) / np.sqrt(3.8)


def cal_polygon_contour(x, y, theta, width, length):
    left_front_x = x + 0.5 * length * torch.cos(theta) - 0.5 * width * torch.sin(theta)
    left_front_y = y + 0.5 * length * torch.sin(theta) + 0.5 * width * torch.cos(theta)
    left_front = torch.stack((left_front_x, left_front_y), dim=1)

    right_front_x = x + 0.5 * length * torch.cos(theta) + 0.5 * width * torch.sin(theta)
    right_front_y = y + 0.5 * length * torch.sin(theta) - 0.5 * width * torch.cos(theta)
    right_front = torch.stack((right_front_x, right_front_y), dim=1)

    right_back_x = x - 0.5 * length * torch.cos(theta) + 0.5 * width * torch.sin(theta)
    right_back_y = y - 0.5 * length * torch.sin(theta) - 0.5 * width * torch.cos(theta)
    right_back = torch.stack((right_back_x, right_back_y), dim=1)

    left_back_x = x - 0.5 * length * torch.cos(theta) - 0.5 * width * torch.sin(theta)
    left_back_y = y - 0.5 * length * torch.sin(theta) + 0.5 * width * torch.cos(theta)
    left_back = torch.stack((left_back_x, left_back_y), dim=1)

    polygon_contour = torch.stack(
        [left_front, right_front, right_back, left_back], dim=1)

    return polygon_contour


def interplating_polyline(polylines, heading, distance=0.5, split_distace=5):
    # Calculate the cumulative distance along the path, up-sample the polyline to 0.5 meter
    dist_along_path_list = [[0]]
    polylines_list = [[polylines[0]]]
    for i in range(1, polylines.shape[0]):
        euclidean_dist = euclidean(polylines[i, :2], polylines[i - 1, :2])
        heading_diff = min(abs(max(heading[i], heading[i - 1]) - min(heading[1], heading[i - 1])),
                           abs(max(heading[i], heading[i - 1]) - min(heading[1], heading[i - 1]) + math.pi))
        if heading_diff > math.pi / 4 and euclidean_dist > 3:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        elif heading_diff > math.pi / 8 and euclidean_dist > 3:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        elif heading_diff > 0.1 and euclidean_dist > 3:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        elif euclidean_dist > 10:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        else:
            dist_along_path_list[-1].append(dist_along_path_list[-1][-1] + euclidean_dist)
            polylines_list[-1].append(polylines[i])
    # plt.plot(polylines[:, 0], polylines[:, 1])
    # plt.savefig('tmp.jpg')
    new_x_list = []
    new_y_list = []
    multi_polylines_list = []
    for idx in range(len(dist_along_path_list)):
        if len(dist_along_path_list[idx]) < 2:
            continue
        dist_along_path = np.array(dist_along_path_list[idx])
        polylines_cur = np.array(polylines_list[idx])
        # Create interpolation functions for x and y coordinates
        fx = interp1d(dist_along_path, polylines_cur[:, 0])
        fy = interp1d(dist_along_path, polylines_cur[:, 1])
        # fyaw = interp1d(dist_along_path, heading)

        # Create an array of distances at which to interpolate
        new_dist_along_path = np.arange(0, dist_along_path[-1], distance)
        new_dist_along_path = np.concatenate([new_dist_along_path, dist_along_path[[-1]]])
        # Use the interpolation functions to generate new x and y coordinates
        new_x = fx(new_dist_along_path)
        new_y = fy(new_dist_along_path)
        # new_yaw = fyaw(new_dist_along_path)
        new_x_list.append(new_x)
        new_y_list.append(new_y)

        # Combine the new x and y coordinates into a single array
        new_polylines = np.vstack((new_x, new_y)).T
        polyline_size = int(split_distace / distance)
        if new_polylines.shape[0] >= (polyline_size + 1):
            padding_size = (new_polylines.shape[0] - (polyline_size + 1)) % polyline_size
            final_index = (new_polylines.shape[0] - (polyline_size + 1)) // polyline_size + 1
        else:
            padding_size = new_polylines.shape[0]
            final_index = 0
        multi_polylines = None
        new_polylines = torch.tensor(new_polylines)
        new_heading = torch.atan2(new_polylines[1:, 1] - new_polylines[:-1, 1],
                                  new_polylines[1:, 0] - new_polylines[:-1, 0])
        new_heading = torch.cat([new_heading, new_heading[-1:]], -1)[..., None]
        new_polylines = torch.cat([new_polylines, new_heading], -1)
        if new_polylines.shape[0] >= (polyline_size + 1):
            multi_polylines = new_polylines.unfold(dimension=0, size=polyline_size + 1, step=polyline_size)
            multi_polylines = multi_polylines.transpose(1, 2)
            multi_polylines = multi_polylines[:, ::5, :]
        if padding_size >= 3:
            last_polyline = new_polylines[final_index * polyline_size:]
            last_polyline = last_polyline[torch.linspace(0, last_polyline.shape[0] - 1, steps=3).long()]
            if multi_polylines is not None:
                multi_polylines = torch.cat([multi_polylines, last_polyline.unsqueeze(0)], dim=0)
            else:
                multi_polylines = last_polyline.unsqueeze(0)
        if multi_polylines is None:
            continue
        multi_polylines_list.append(multi_polylines)
    if len(multi_polylines_list) > 0:
        multi_polylines_list = torch.cat(multi_polylines_list, dim=0)
    else:
        multi_polylines_list = None
    return multi_polylines_list


def average_distance_vectorized(point_set1, centroids):
    dists = np.sqrt(np.sum((point_set1[:, None, :, :] - centroids[None, :, :, :]) ** 2, axis=-1))
    return np.mean(dists, axis=2)


def assign_clusters(sub_X, centroids):
    distances = average_distance_vectorized(sub_X, centroids)
    return np.argmin(distances, axis=1)


class TokenProcessor:

    def __init__(self, token_size):
        module_dir = os.path.dirname(os.path.dirname(__file__))
        self.agent_token_path = os.path.join(module_dir, f'models/smart/tokens/cluster_frame_5_{token_size}.pkl')
        self.map_token_traj_path = os.path.join(module_dir, 'models/smart/tokens/map_traj_token5.pkl')
        self.noise = False
        self.disturb = False
        self.shift = 5
        self.get_trajectory_token()
        self.training = False
        self.current_step = 10

    def preprocess(self, data):
        data = self.convert_data(data)
        data = self.tokenize_map(data)
        # data = self.get_interaction_labels_sparse(data)
        # del data['city']
        if 'polygon_is_intersection' in data['map_polygon']:
            del data['map_polygon']['polygon_is_intersection']
        if 'route_type' in data['map_polygon']:
            del data['map_polygon']['route_type']
        return data

    def get_trajectory_token(self):
        agent_token_data = pickle.load(open(self.agent_token_path, 'rb'))
        map_token_traj = pickle.load(open(self.map_token_traj_path, 'rb'))
        self.trajectory_token = agent_token_data['token']
        self.trajectory_token_all = agent_token_data['token_all']
        self.map_token = {'traj_src': map_token_traj['traj_src'], }
        self.token_last = {}
        for k, v in self.trajectory_token_all.items():
            v = np.array(v, dtype=np.float64)
            token_last = torch.tensor(v[:, -2:]).to(torch.float)
            diff_xy = token_last[:, 0, 0] - token_last[:, 0, 3]
            theta = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])
            cos, sin = theta.cos(), theta.sin()
            rot_mat = theta.new_zeros(token_last.shape[0], 2, 2)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = -sin
            rot_mat[:, 1, 0] = sin
            rot_mat[:, 1, 1] = cos
            agent_token = torch.bmm(token_last[:, 1], rot_mat)
            agent_token -= token_last[:, 0].mean(1)[:, None, :]
            self.token_last[k] = agent_token.numpy()

    def clean_heading(self, data):
        heading = data['agent']['heading']
        valid = data['agent']['valid_mask']
        pi = torch.tensor(torch.pi)
        n_vehicles, n_frames = heading.shape

        heading_diff_raw = heading[:, :-1] - heading[:, 1:]
        heading_diff = torch.remainder(heading_diff_raw + pi, 2 * pi) - pi
        heading_diff[heading_diff > pi] -= 2 * pi
        heading_diff[heading_diff < -pi] += 2 * pi

        valid_pairs = valid[:, :-1] & valid[:, 1:]

        for i in range(n_frames - 1):
            change_needed = (torch.abs(heading_diff[:, i:i + 1]) > 1.0) & valid_pairs[:, i:i + 1]

            heading[:, i + 1][change_needed.squeeze()] = heading[:, i][change_needed.squeeze()]

            if i < n_frames - 2:
                heading_diff_raw = heading[:, i + 1] - heading[:, i + 2]
                heading_diff[:, i + 1] = torch.remainder(heading_diff_raw + pi, 2 * pi) - pi
                heading_diff[heading_diff[:, i + 1] > pi] -= 2 * pi
                heading_diff[heading_diff[:, i + 1] < -pi] += 2 * pi


    def convert_data(self, data):
        ori_obs_traj_xy = data.veh_xseq[:,:,:2]
        batch_size = ori_obs_traj_xy.shape[0]
        obs_traj_xy = torch.zeros((batch_size, 20, 2), dtype=ori_obs_traj_xy.dtype, device=ori_obs_traj_xy.device)
        obs_traj_xy[:, 1::2, :] = ori_obs_traj_xy
        obs_traj_xy[:, 2::2, :] = (ori_obs_traj_xy[:, 1:, :2] + ori_obs_traj_xy[:, :-1, :2]) * 0.5
        delta = obs_traj_xy[:, 1:, :] - obs_traj_xy[:, :-1, :]
        dx = delta[:, :, 0]
        dy = delta[:, :, 1]
        obs_heading = torch.atan2(dy, dx)
        obs_info = data.veh_xseq[:,:,4].bool()
        obs_valid_mask = torch.zeros((batch_size, 20), dtype=torch.bool, device=obs_traj_xy.device)
        obs_valid_mask[:, 1::2] = obs_info
        obs_valid_mask[:, 2::2] = obs_info[:, :-1] & obs_info[:, 1:]
        ori_fut_traj_xy = data.veh_yseq[:,:,:2]
        ori_curr_fut_traj_xy = torch.cat([ori_obs_traj_xy[:, -1:, :], ori_fut_traj_xy], dim=1)
        fut_traj_xy = torch.zeros((batch_size, 30, 2), dtype=ori_fut_traj_xy.dtype, device=ori_fut_traj_xy.device)
        fut_traj_xy[:, 1::2, :] = ori_fut_traj_xy
        fut_traj_xy[:, 0::2, :] = (ori_fut_traj_xy + ori_curr_fut_traj_xy[:, :-1, :2]) * 0.5
        fut_info = data.veh_has_preds
        fut_valid_mask = torch.zeros((batch_size, 30), dtype=torch.bool, device=fut_traj_xy.device)
        cur_fut_info = torch.cat([obs_info[:, -1:], fut_info], dim=1)
        fut_valid_mask[:, 1::2] = fut_info
        fut_valid_mask[:, 0::2] = fut_info & cur_fut_info[:, :-1]
        position = torch.cat([obs_traj_xy, fut_traj_xy], dim=1)
        valid_mask = torch.cat([obs_valid_mask, fut_valid_mask], dim=1)
        delta = position[:, 1:, :] - position[:, :-1, :]
        delta = torch.cat([delta, delta[:, -1:, :]], dim=1)  # [batch_size, num_steps, 2]
        dx = delta[:, :, 0]
        dy = delta[:, :, 1]
        heading = torch.atan2(dy, dx)
        velocity = delta / 0.1  # assuming 10Hz sampling rate
        agent_type = torch.zeros(batch_size, dtype=torch.uint8, device=position.device)
        agent_category = 3* torch.ones(batch_size, dtype=torch.uint8, device=position.device) 
        position = position[:, 9:]
        heading = heading[:, 9:]
        velocity = velocity[:, 9:]
        valid_mask = valid_mask[:, 9:]

        shape = data.veh_shape # [num_agents, 2]
        height  = 1.64
        # new_shape = torch.zeros((shape.shape[0], position.shape[1], 3), dtype=torch.float, device=position.device) # [num_agents, num_steps, 3]
        # for i in range(shape.shape[0]):
        #     length = shape[i, 0].item()
        #     width = shape[i, 1].item()
        #     new_shape[i, :, 0] = length
        #     new_shape[i, :, 1] = width
        #     new_shape[i, :, 2] = height
        length_width = shape.unsqueeze(1).expand(-1, position.shape[1], -1)  # [num_agents, num_steps, 2]
        height_tensor = torch.full((shape.shape[0], position.shape[1], 1), height, device=position.device)
        new_shape = torch.cat([length_width, height_tensor], dim=-1)  # [num_ag

        new_data = {'agent':{
            'position': position,
            'heading': heading,
            'velocity': velocity,
            'valid_mask': valid_mask,
            'type': agent_type,
            'category': agent_category,
            'num_nodes': batch_size,
            'shape': new_shape,
            'av_index': 0
        },
        }
        # self.clean_heading(new_data)
        matching_extra_mask = (new_data['agent']['valid_mask'][:, self.current_step] == True) * (
                new_data['agent']['valid_mask'][:, self.current_step - 5] == False)
        token_index, token_contour = self.match_token(position, valid_mask, heading,
                                                              'veh', agent_category,
                                                              matching_extra_mask)
        new_data['agent']['token_idx'] = token_index
        new_data['agent']['token_contour'] = token_contour
        token_pos = token_contour.mean(dim=2)
        new_data['agent']['token_pos'] = token_pos
        diff_xy = token_contour[:, :, 0, :] - token_contour[:, :, 3, :]
        new_data['agent']['token_heading'] = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])
        valid_mask_shift = valid_mask.unfold(1, self.shift + 1, self.shift)
        token_valid_mask = valid_mask_shift[:, :, 0] * valid_mask_shift[:, :, -1]
        new_data['agent']['agent_valid_mask'] = token_valid_mask
        vel = torch.cat([token_pos.new_zeros(new_data['agent']['num_nodes'], 1, 2),
                         ((token_pos[:, 1:] - token_pos[:, :-1]) / (0.1 * self.shift))], dim=1)
        vel_valid_mask = torch.cat([torch.zeros(token_valid_mask.shape[0], 1, dtype=torch.bool, device=token_valid_mask.device),
                                    (token_valid_mask * token_valid_mask.roll(shifts=1, dims=1))[:, 1:]], dim=1)
        vel[~vel_valid_mask] = 0
        vel[new_data['agent']['valid_mask'][:, self.current_step], 1] = new_data['agent']['velocity'][
                                                                    new_data['agent']['valid_mask'][:, self.current_step],
                                                                    self.current_step, :2]

        new_data['agent']['token_velocity'] = vel
        ####### map data ########
        
        map_data = {
            'map_polygon' : {},
            'map_point': {},
            ('map_point', 'to', 'map_polygon'): {},
            ('map_polygon', 'to', 'map_polygon'): {},
        }
        num_polygons = data.lane_ctrs.size(0)
        map_data['map_polygon']['num_nodes'] = num_polygons
        map_data['map_polygon']['type'] = torch.full((num_polygons,), 3, dtype=torch.uint8, device=position.device)  # 3 for road polygon
        map_data['map_polygon']['light_type'] = torch.zeros(num_polygons, dtype=torch.bool, device=position.device) # no traffic light
        num_points = torch.tensor([3 for _ in range(num_polygons)], dtype=torch.long, device=position.device)
        point_to_polygon_edge_index = torch.stack(
        [torch.arange(num_points.sum(), dtype=torch.long, device=position.device),
            torch.arange(num_polygons, dtype=torch.long, device=position.device).repeat_interleave(num_points)], dim=0)
        map_data['map_point']['num_nodes'] = num_points.sum().item()
        point_position, point_orientation = [], []
        for i in range(num_polygons):
            ctr = data.lane_ctrs[i]
            vec = data.lane_vecs[i]
            point_position.append(ctr - vec * 0.25)
            point_position.append(ctr)
            point_position.append(ctr + vec * 0.25)
            point_orientation.append(torch.atan2(vec[1], vec[0]))
            point_orientation.append(torch.atan2(vec[1], vec[0]))
            point_orientation.append(torch.atan2(vec[1], vec[0]))

        map_data['map_point']['position'] = torch.stack(point_position, dim=0)
        map_data['map_point']['orientation'] = torch.stack(point_orientation, dim=0)
        # map_data['map_point']['magnitude'] = torch.cat(point_magnitude, dim=0)
        map_data['map_point']['type'] = torch.full((map_data['map_point']['num_nodes'],),  16, dtype=torch.uint8, device=position.device) # 16 for road point
        polygon_to_polygon_edge_index = []
        polygon_to_polygon_type = []
        poly_inds, succ_inds = data.lane_suc_edge_index[0]
        polygon_to_polygon_edge_index.append(
            torch.stack([succ_inds,
                        poly_inds], dim=0))
        polygon_to_polygon_type.append(
            torch.full((succ_inds.shape[0],), 2, dtype=torch.uint8, device=position.device))
            # 0 for road edge
        poly_inds, pred_inds = data.lane_pre_edge_index[0]
        polygon_to_polygon_edge_index.append(
            torch.stack([pred_inds,
                        poly_inds], dim=0))
        polygon_to_polygon_type.append(
            torch.full((pred_inds.shape[0],), 1, dtype=torch.uint8, device=position.device))  # 1 for road edge
        
        poly_inds, left_inds = data.lane_left_edge_index
        polygon_to_polygon_edge_index.append(
            torch.stack([left_inds,
                        poly_inds], dim=0))
        polygon_to_polygon_type.append(
            torch.full((left_inds.shape[0],), 3, dtype=torch.uint8, device=position.device))  # 3 for road edge
        
        poly_inds, right_inds = data.lane_right_edge_index
        polygon_to_polygon_edge_index.append(
            torch.stack([right_inds,
                        poly_inds], dim=0))
        polygon_to_polygon_type.append(
            torch.full((right_inds.shape[0],), 4, dtype=torch.uint8, device=position.device))  # 4 for road edge
        if len(polygon_to_polygon_edge_index) != 0:
            polygon_to_polygon_edge_index = torch.cat(polygon_to_polygon_edge_index, dim=1)
            polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
        else:
            polygon_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long)
            polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)

        map_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = polygon_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['type'] = polygon_to_polygon_type

        for k, v in map_data.items():
            new_data[k] = v
        return new_data 
    
    def tokenize_agent(self, data):
        if data['agent']["velocity"].shape[1] == 90:
            print(data['scenario_id'], data['agent']["velocity"].shape)
        interplote_mask = (data['agent']['valid_mask'][:, self.current_step] == False) * (
                data['agent']['position'][:, self.current_step, 0] != 0)
        if data['agent']["velocity"].shape[-1] == 2:
            data['agent']["velocity"] = torch.cat([data['agent']["velocity"],
                                                   torch.zeros(data['agent']["velocity"].shape[0],
                                                               data['agent']["velocity"].shape[1], 1)], dim=-1)
        vel = data['agent']["velocity"][interplote_mask, self.current_step]
        data['agent']['position'][interplote_mask, self.current_step - 1, :3] = data['agent']['position'][
                                                                                interplote_mask, self.current_step,
                                                                                :3] - vel * 0.1
        data['agent']['valid_mask'][interplote_mask, self.current_step - 1:self.current_step + 1] = True
        data['agent']['heading'][interplote_mask, self.current_step - 1] = data['agent']['heading'][
            interplote_mask, self.current_step]
        data['agent']["velocity"][interplote_mask, self.current_step - 1] = data['agent']["velocity"][
            interplote_mask, self.current_step]

        data['agent']['type'] = data['agent']['type'].to(torch.uint8)

        self.clean_heading(data)
        matching_extra_mask = (data['agent']['valid_mask'][:, self.current_step] == True) * (
                data['agent']['valid_mask'][:, self.current_step - 5] == False)

        interplote_mask_first = (data['agent']['valid_mask'][:, 0] == False) * (data['agent']['position'][:, 0, 0] != 0)
        data['agent']['valid_mask'][interplote_mask_first, 0] = True

        agent_pos = data['agent']['position'][:, :, :2]
        valid_mask = data['agent']['valid_mask']

        valid_mask_shift = valid_mask.unfold(1, self.shift + 1, self.shift)
        token_valid_mask = valid_mask_shift[:, :, 0] * valid_mask_shift[:, :, -1]
        agent_type = data['agent']['type']
        agent_category = data['agent']['category']
        agent_heading = data['agent']['heading']
        vehicle_mask = agent_type == 0
        cyclist_mask = agent_type == 2
        ped_mask = agent_type == 1

        veh_pos = agent_pos[vehicle_mask, :, :]
        veh_valid_mask = valid_mask[vehicle_mask, :]
        veh_token_valid_mask = token_valid_mask[vehicle_mask, :]
        cyc_pos = agent_pos[cyclist_mask, :, :]
        cyc_valid_mask = valid_mask[cyclist_mask, :]
        cyc_token_valid_mask = token_valid_mask[cyclist_mask, :]
        ped_pos = agent_pos[ped_mask, :, :]
        ped_valid_mask = valid_mask[ped_mask, :]
        ped_token_valid_mask = token_valid_mask[ped_mask, :]

        veh_token_index, veh_token_contour = self.match_token(veh_pos, veh_valid_mask, agent_heading[vehicle_mask],
                                                              'veh', agent_category[vehicle_mask],
                                                              matching_extra_mask[vehicle_mask])
        ped_token_index, ped_token_contour = self.match_token(ped_pos, ped_valid_mask, agent_heading[ped_mask], 'ped',
                                                              agent_category[ped_mask], matching_extra_mask[ped_mask])
        cyc_token_index, cyc_token_contour = self.match_token(cyc_pos, cyc_valid_mask, agent_heading[cyclist_mask],
                                                              'cyc', agent_category[cyclist_mask],
                                                              matching_extra_mask[cyclist_mask])

        target_label = torch.zeros((agent_pos.shape[0], veh_token_contour.shape[1], 2)).to(torch.int64)
        for i in range(target_label.shape[0]):
            for j in range(target_label.shape[1]):
                target_label[i, j, 0] = i
                target_label[i, j, 1] = j

        veh_target_label = target_label[vehicle_mask]   
        ped_target_label = target_label[ped_mask]
        cyc_target_label = target_label[cyclist_mask]

        veh_source_token_index, veh_target_token_index, veh_source_token_pos, veh_target_token_pos, veh_t, veh_token_pos_at_t, veh_token_vel_at_t = self.optimal_transport(veh_token_index, 'veh', veh_target_label, veh_token_valid_mask)
        ped_source_token_index, ped_target_token_index, ped_source_token_pos, ped_target_token_pos, ped_t, ped_token_pos_at_t, ped_token_vel_at_t = self.optimal_transport(ped_token_index, 'ped', ped_target_label, ped_token_valid_mask)
        cyc_source_token_index, cyc_target_token_index, cyc_source_token_pos, cyc_target_token_pos, cyc_t, cyc_token_pos_at_t, cyc_token_vel_at_t = self.optimal_transport(cyc_token_index, 'cyc', cyc_target_label, cyc_token_valid_mask)

        token_index = torch.zeros((agent_pos.shape[0], veh_token_index.shape[1])).to(torch.int64)
        token_index[vehicle_mask] = veh_token_index
        token_index[ped_mask] = ped_token_index
        token_index[cyclist_mask] = cyc_token_index

        token_time = torch.zeros((agent_pos.shape[0], veh_t.shape[1], 1)).to(torch.float)
        token_time[vehicle_mask] = veh_t
        token_time[ped_mask] = ped_t
        token_time[cyclist_mask] = cyc_t

        token_pos_at_t = torch.zeros((agent_pos.shape[0], veh_token_pos_at_t.shape[1], veh_token_pos_at_t.shape[2])).to(torch.float)
        token_pos_at_t[vehicle_mask] = veh_token_pos_at_t
        token_pos_at_t[ped_mask] = ped_token_pos_at_t
        token_pos_at_t[cyclist_mask] = cyc_token_pos_at_t

        token_vel_at_t = torch.zeros((agent_pos.shape[0], veh_token_vel_at_t.shape[1], veh_token_vel_at_t.shape[2])).to(torch.float)
        token_vel_at_t[vehicle_mask] = veh_token_vel_at_t
        token_vel_at_t[ped_mask] = ped_token_vel_at_t
        token_vel_at_t[cyclist_mask] = cyc_token_vel_at_t

        source_token_index = torch.zeros((agent_pos.shape[0], veh_source_token_index.shape[1])).to(torch.int64)
        source_token_index[vehicle_mask] = veh_source_token_index
        source_token_index[ped_mask] = ped_source_token_index
        source_token_index[cyclist_mask] = cyc_source_token_index

        target_token_index = torch.zeros((agent_pos.shape[0], veh_target_token_index.shape[1], 2)).to(torch.int64)
        target_token_index[vehicle_mask] = veh_target_token_index
        target_token_index[ped_mask] = ped_target_token_index
        target_token_index[cyclist_mask] = cyc_target_token_index

        source_token_pos = torch.zeros((agent_pos.shape[0], veh_source_token_pos.shape[1], veh_source_token_pos.shape[2])).to(torch.float)
        source_token_pos[vehicle_mask] = veh_source_token_pos
        source_token_pos[ped_mask] = ped_source_token_pos
        source_token_pos[cyclist_mask] = cyc_source_token_pos

        target_token_pos = torch.zeros((agent_pos.shape[0], veh_target_token_pos.shape[1], veh_target_token_pos.shape[2])).to(torch.float)
        target_token_pos[vehicle_mask] = veh_target_token_pos
        target_token_pos[ped_mask] = ped_target_token_pos
        target_token_pos[cyclist_mask] = cyc_target_token_pos

        token_contour = torch.zeros((agent_pos.shape[0], veh_token_contour.shape[1],
                                     veh_token_contour.shape[2], veh_token_contour.shape[3]))
        token_contour[vehicle_mask] = veh_token_contour
        token_contour[ped_mask] = ped_token_contour
        token_contour[cyclist_mask] = cyc_token_contour

        trajectory_token_veh = torch.from_numpy(self.trajectory_token['veh']).clone().to(torch.float)
        trajectory_token_ped = torch.from_numpy(self.trajectory_token['ped']).clone().to(torch.float)
        trajectory_token_cyc = torch.from_numpy(self.trajectory_token['cyc']).clone().to(torch.float)

        agent_token_traj = torch.zeros((agent_pos.shape[0], trajectory_token_veh.shape[0], 4, 2))
        agent_token_traj[vehicle_mask] = trajectory_token_veh
        agent_token_traj[ped_mask] = trajectory_token_ped
        agent_token_traj[cyclist_mask] = trajectory_token_cyc

        if not self.training:
            token_valid_mask[matching_extra_mask, 1] = True

        data['agent']['token_idx'] = token_index
        data['agent']['token_contour'] = token_contour
        token_pos = token_contour.mean(dim=2)
        data['agent']['token_pos'] = token_pos
        diff_xy = token_contour[:, :, 0, :] - token_contour[:, :, 3, :]
        data['agent']['token_heading'] = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])
        data['agent']['agent_valid_mask'] = token_valid_mask

        data['agent']['flow_matching'] = {}
        data['agent']['flow_matching']['source_token_index'] = source_token_index
        data['agent']['flow_matching']['target_token_index'] = target_token_index
        data['agent']['flow_matching']['source_token_pos'] = source_token_pos
        data['agent']['flow_matching']['target_token_pos'] = target_token_pos
        data['agent']['flow_matching']['token_pos_at_t'] = token_pos_at_t
        data['agent']['flow_matching']['token_time'] = token_time
        data['agent']['flow_matching']['token_vel_at_t'] = token_vel_at_t

        vel = torch.cat([token_pos.new_zeros(data['agent']['num_nodes'], 1, 2),
                         ((token_pos[:, 1:] - token_pos[:, :-1]) / (0.1 * self.shift))], dim=1)
        vel_valid_mask = torch.cat([torch.zeros(token_valid_mask.shape[0], 1, dtype=torch.bool),
                                    (token_valid_mask * token_valid_mask.roll(shifts=1, dims=1))[:, 1:]], dim=1)
        vel[~vel_valid_mask] = 0
        vel[data['agent']['valid_mask'][:, self.current_step], 1] = data['agent']['velocity'][
                                                                    data['agent']['valid_mask'][:, self.current_step],
                                                                    self.current_step, :2]

        data['agent']['token_velocity'] = vel

        return data

    def match_token(self, pos, valid_mask, heading, category, agent_category, extra_mask):
        agent_token_src = self.trajectory_token[category]
        token_last = self.token_last[category]
        if self.shift <= 2:
            if category == 'veh':
                width = 1.0
                length = 2.4
            elif category == 'cyc':
                width = 0.5
                length = 1.5
            else:
                width = 0.5
                length = 0.5
        else:
            if category == 'veh':
                width = 2.0
                length = 4.8
            elif category == 'cyc':
                width = 1.0
                length = 2.0
            else:
                width = 1.0
                length = 1.0

        prev_heading = heading[:, 0]
        prev_pos = pos[:, 0]
        agent_num, num_step, feat_dim = pos.shape
        token_num, token_contour_dim, feat_dim = agent_token_src.shape
        agent_token_src = agent_token_src.reshape(1, token_num * token_contour_dim, feat_dim).repeat(agent_num, 0)
        token_last = token_last.reshape(1, token_num * token_contour_dim, feat_dim).repeat(extra_mask.sum().item(), 0)
        token_index_list = []
        token_contour_list = []
        prev_token_idx = None

        for i in range(self.shift, pos.shape[1], self.shift):
            theta = prev_heading
            cur_heading = heading[:, i]
            cur_pos = pos[:, i]
            cos, sin = theta.cos(), theta.sin()
            rot_mat = theta.new_zeros(agent_num, 2, 2)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            agent_token_world = torch.bmm(torch.tensor(agent_token_src, device=rot_mat.device).to(torch.float), rot_mat).reshape(agent_num,
                                                                                                              token_num,
                                                                                                              token_contour_dim,
                                                                                                              feat_dim)
            agent_token_world += prev_pos[:, None, None, :]

            cur_contour = cal_polygon_contour(cur_pos[:, 0], cur_pos[:, 1], cur_heading, width, length)
            # agent_token_index = torch.tensor(np.argmin(
            #     np.mean(np.sqrt(np.sum((cur_contour[:, None, ...] - agent_token_world.numpy()) ** 2, axis=-1)), axis=2),
            #     axis=-1), device=agent_token_world.device)
            dists = torch.norm(
                cur_contour[:, None, ...] - agent_token_world, dim=-1
            )  # [B, N, 4]
            mean_dists = dists.mean(dim=2)  # [B, N]
            agent_token_index = torch.argmin(mean_dists, dim=1)  # [B]

            # if prev_token_idx is not None and self.noise:
            #     same_idx = prev_token_idx == agent_token_index
            #     same_idx[:] = True
            #     topk_indices = np.argsort(
            #         np.mean(np.sqrt(np.sum((cur_contour[:, None, ...] - agent_token_world.numpy()) ** 2, axis=-1)),
            #                 axis=2), axis=-1)[:, :5]
            #     sample_topk = np.random.choice(range(0, topk_indices.shape[1]), topk_indices.shape[0])
            #     agent_token_index[same_idx] = \
            #         torch.tensor(topk_indices[np.arange(topk_indices.shape[0]), sample_topk], device=agent_token_world.device)[same_idx]

            token_contour_select = agent_token_world[torch.arange(agent_num, device=agent_token_world.device), agent_token_index]

            diff_xy = token_contour_select[:, 0, :] - token_contour_select[:, 3, :]

            prev_heading = heading[:, i].clone()
            prev_heading[valid_mask[:, i - self.shift]] = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])[
                valid_mask[:, i - self.shift]]

            prev_pos = pos[:, i].clone()
            prev_pos[valid_mask[:, i - self.shift]] = token_contour_select.mean(dim=1)[valid_mask[:, i - self.shift]]
            prev_token_idx = agent_token_index
            token_index_list.append(agent_token_index[:, None])
            token_contour_list.append(token_contour_select[:, None, ...])

        token_index = torch.cat(token_index_list, dim=1)
        token_contour = torch.cat(token_contour_list, dim=1)

        # extra matching
        if not self.training:
            theta = heading[extra_mask, self.current_step - 1]
            prev_pos = pos[extra_mask, self.current_step - 1]
            cur_pos = pos[extra_mask, self.current_step]
            cur_heading = heading[extra_mask, self.current_step]
            cos, sin = theta.cos(), theta.sin()
            rot_mat = theta.new_zeros(extra_mask.sum(), 2, 2)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            agent_token_world = torch.bmm(torch.tensor(token_last, device=rot_mat.device).to(torch.float), rot_mat).reshape(
                extra_mask.sum(), token_num, token_contour_dim, feat_dim)
            agent_token_world += prev_pos[:, None, None, :]

            cur_contour = cal_polygon_contour(cur_pos[:, 0], cur_pos[:, 1], cur_heading, width, length)
            # agent_token_index = torch.tensor(np.argmin(
            #     np.mean(np.sqrt(np.sum((cur_contour[:, None, ...] - agent_token_world.numpy()) ** 2, axis=-1)), axis=2),
            #     axis=-1), device=agent_token_world.device)
            dists = torch.norm(
                cur_contour[:, None, ...] - agent_token_world, dim=-1
            )  # [B, N, 4]
            mean_dists = dists.mean(dim=2)  # [B, N]
            agent_token_index = torch.argmin(mean_dists, dim=1)  # [B]

            token_contour_select = agent_token_world[torch.arange(extra_mask.sum(), device=agent_token_world.device), agent_token_index]

            token_index[extra_mask, 1] = agent_token_index
            token_contour[extra_mask, 1] = token_contour_select
        return token_index, token_contour


    def tokenize_map(self, data):
        data['map_polygon']['type'] = data['map_polygon']['type'].to(torch.uint8)
        data['map_point']['type'] = data['map_point']['type'].to(torch.uint8)
        pt2pl = data[('map_point', 'to', 'map_polygon')]['edge_index']
        pt_type = data['map_point']['type'].to(torch.uint8)
        pt_side = torch.zeros_like(pt_type)
        pt_pos = data['map_point']['position'][:, :2]
        data['map_point']['orientation'] = wrap_angle(data['map_point']['orientation'])
        pt_heading = data['map_point']['orientation']
        split_polyline_type = []
        split_polyline_pos = []
        split_polyline_theta = []
        split_polyline_side = []
        pl_idx_list = []
        split_polygon_type = []
        data['map_point']['type'].unique()

        for i in sorted(torch.unique(pt2pl[1])):
            index = pt2pl[0, pt2pl[1] == i]
            polygon_type = data['map_polygon']["type"][i]
            cur_side = pt_side[index]
            cur_type = pt_type[index]
            cur_pos = pt_pos[index]
            cur_heading = pt_heading[index]

            for side_val in torch.unique(cur_side):
                for type_val in torch.unique(cur_type):
                    if type_val == 13:
                        continue
                    indices = torch.where((cur_side == side_val) & (cur_type == type_val))[0]
                    if len(indices) <= 2:
                        continue
                    split_polyline = interplating_polyline(cur_pos[indices].numpy(), cur_heading[indices].numpy())
                    if split_polyline is None:
                        continue
                    new_cur_type = cur_type[indices][0]
                    new_cur_side = cur_side[indices][0]
                    map_polygon_type = polygon_type.repeat(split_polyline.shape[0])
                    new_cur_type = new_cur_type.repeat(split_polyline.shape[0])
                    new_cur_side = new_cur_side.repeat(split_polyline.shape[0])
                    cur_pl_idx = torch.Tensor([i])
                    new_cur_pl_idx = cur_pl_idx.repeat(split_polyline.shape[0])
                    split_polyline_pos.append(split_polyline[..., :2])
                    split_polyline_theta.append(split_polyline[..., 2])
                    split_polyline_type.append(new_cur_type)
                    split_polyline_side.append(new_cur_side)
                    pl_idx_list.append(new_cur_pl_idx)
                    split_polygon_type.append(map_polygon_type)

        split_polyline_pos = torch.cat(split_polyline_pos, dim=0)
        split_polyline_theta = torch.cat(split_polyline_theta, dim=0)
        split_polyline_type = torch.cat(split_polyline_type, dim=0)
        split_polyline_side = torch.cat(split_polyline_side, dim=0)
        split_polygon_type = torch.cat(split_polygon_type, dim=0)
        pl_idx_list = torch.cat(pl_idx_list, dim=0)
        vec = split_polyline_pos[:, 1, :] - split_polyline_pos[:, 0, :]
        data['map_save'] = {}
        data['pt_token'] = {}
        data['map_save']['traj_pos'] = split_polyline_pos
        data['map_save']['traj_theta'] = split_polyline_theta[:, 0]  # torch.arctan2(vec[:, 1], vec[:, 0])
        data['map_save']['pl_idx_list'] = pl_idx_list
        data['pt_token']['type'] = split_polyline_type
        data['pt_token']['side'] = split_polyline_side
        data['pt_token']['pl_type'] = split_polygon_type
        data['pt_token']['num_nodes'] = split_polyline_pos.shape[0]
        return data

    def get_interaction_labels_sparse(self, data):
        """
        Computes ground-truth sparse DAG interaction labels for SMART-style data.
        Adds: data['agent']['ig_labels_sparse'] as a 1D array [num_edges]
        """

        # Parameters
        future_start = self.future_start_step if hasattr(self, "future_start_step") else 11
        future_len = data['agent']['position'].shape[1] - future_start

        # Extract features (convert to numpy)
        feat_locs = data['agent']['position'][:, self.current_step+1:, :2].cpu().numpy()
        feat_vels = data['agent']['velocity'][:, self.current_step+1:, :2].cpu().numpy()
        feat_psirads = data['agent']['heading'][:, self.current_step+1:].unsqueeze(-1).cpu().numpy()
        has_obss = data['agent']['valid_mask'][:, self.current_step+1:].cpu().numpy()
        agenttypes = data['agent']['type'].cpu().numpy()
        is_valid_agent = data['agent']['valid_mask'][:, self.current_step].cpu().numpy()

        # Generate shape tensor for each agent
        def get_shape(t):
            if t == 0:  # vehicle
                return (4.0, 1.8)
            elif t == 1:  # pedestrian
                return (0.7, 0.7)
            elif t == 2:  # cyclist
                return (0.7, 0.7)
            else:
                return (1.0, 1.0)

        shapes = np.stack([get_shape(t) for t in agenttypes], axis=0)
        shapes = np.tile(shapes[:, None, :], (1, future_len, 1))  # [N, T, 2]

        # Call the original logic (you must have `return_circle_list`, `return_collision_threshold` defined)
        ig_labels = self._compute_ig_labels_sparse(
            feat_locs, feat_vels, feat_psirads, shapes, has_obss, is_valid_agent, agenttypes
        )

        # Attach to data dict
        data['agent']['ig_labels'] = torch.tensor(ig_labels, dtype=torch.int64)
        return data

    def _compute_ig_labels_sparse(self, feat_locs, feat_vels, feat_psirads, shapes, has_obss, is_valid_agent, agenttypes):

        '''
            feat_locs: [N, T, 2], where N is the number of agents, T is the number of future timesteps
            feat_vels: [N, T, 2], where N is the number of agents, T is the number of future timesteps
            feat_psirads: [N, T, 1], where N is the number of agents, T is the number of future timesteps
            shapes: [N, T, 2], where N is the number of agents, T is the number of future timesteps
            has_obss: [N, T], where N is the number of agents, T is the number of future timesteps
            is_valid_agent: [N], where N is the number of agents
            agenttypes: [N], where N is the number of agents
        '''
        
        N = feat_locs.shape[0]
        labels = np.zeros((N, N))
        orig_trajs = feat_locs 

        circle_lists = []
        for i in range(N):
            agenttype_i = agenttypes[i]
            shape_i = shapes[i][0]
            length = shape_i[0]
            width = shape_i[1]
            
            traj_i = orig_trajs[i][has_obss[i] == 1]
            psirad_i = feat_psirads[i][has_obss[i] == 1]
            # shape is [30, c, 2], where c is the number of circles prescribed to vehicle i (depends on the size/shape of vehicle i)
            circle_lists.append(return_circle_list(traj_i[:, 0], traj_i[:, 1], length, width, psirad_i[:, 0]))
        
        for a in range(1, N):
            for b in range(a):
                agenttype_a = agenttypes[a]
                
                shape_a = shapes[a][0]
                width_a = shape_a[1]
                
                    
                agenttype_b = agenttypes[b]
                shape_b = shapes[b][0]
                width_b = shape_b[1]
                
                # for each (unordered) pairs of vehicles, we check if they are interacting
                # by checking if there is a collision at any pair of future timesteps. 
                circle_list_a = circle_lists[a]
                circle_list_b = circle_lists[b]

                # threshold determined according to widths of vehicles
                thresh = return_collision_threshold(width_a, width_b)

                dist = np.expand_dims(np.expand_dims(circle_list_a, axis=1), axis=2) - np.expand_dims(np.expand_dims(circle_list_b, axis=0), axis=3)
                dist = np.linalg.norm(dist, axis=-1, ord=2)
                
                # [T_a, T_b, num_circles_a, num_circles_b], where T_a is the number of ground-truth future positions present in a's trajectory, and b defined similarly.
                is_coll = dist < thresh
                is_coll_cumul = is_coll.sum(2).sum(2)
                # binary mask of shape [T_a, T_b]
                is_coll_mask = is_coll_cumul > 0

                if is_coll_mask.sum() < 1:
                    continue

                # fill in for indices (0) that do not have a ground-truth position
                for en, ind in enumerate(has_obss[a]):
                    if ind == 0:
                        is_coll_mask = np.insert(is_coll_mask, en, 0, axis=0)

                for en, ind in enumerate(has_obss[b]):
                    if ind == 0:
                        is_coll_mask = np.insert(is_coll_mask, en, 0, axis=1)  

                # assert is_coll_mask.shape == (30, 30)

                # [P, 2], first index is a, second is b; P is number of colliding pairs
                coll_ids = np.argwhere(is_coll_mask == 1)
                # only preserve the colliding pairs that are within 2.5 seconds (= 25 timesteps) of eachother
                valid_coll_mask = np.abs(coll_ids[:, 0] - coll_ids[:, 1]) <= 40 #40/25/10

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