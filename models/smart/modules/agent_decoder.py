import pickle
from typing import Dict, Mapping, Optional
import torch
import torch.nn as nn
from models.smart.layers import MLPLayer
from models.smart.layers.attention_layer import AttentionLayer
from models.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from torch_cluster import radius, radius_graph
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import dense_to_sparse, subgraph
from models.smart.utils import angle_between_2d_vectors, weight_init, wrap_angle
import math
import numpy as np

def cal_polygon_contour(x, y, theta, width, length):
    left_front_x = x + 0.5 * length * math.cos(theta) - 0.5 * width * math.sin(theta)
    left_front_y = y + 0.5 * length * math.sin(theta) + 0.5 * width * math.cos(theta)
    left_front = (left_front_x, left_front_y)

    right_front_x = x + 0.5 * length * math.cos(theta) + 0.5 * width * math.sin(theta)
    right_front_y = y + 0.5 * length * math.sin(theta) - 0.5 * width * math.cos(theta)
    right_front = (right_front_x, right_front_y)

    right_back_x = x - 0.5 * length * math.cos(theta) + 0.5 * width * math.sin(theta)
    right_back_y = y - 0.5 * length * math.sin(theta) - 0.5 * width * math.cos(theta)
    right_back = (right_back_x, right_back_y)

    left_back_x = x - 0.5 * length * math.cos(theta) - 0.5 * width * math.sin(theta)
    left_back_y = y - 0.5 * length * math.sin(theta) + 0.5 * width * math.cos(theta)
    left_back = (left_back_x, left_back_y)
    polygon_contour = [left_front, right_front, right_back, left_back]

    return polygon_contour


class SMARTAgentDecoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 token_data: Dict,
                 token_size=512,
                 if_control=False) -> None:
        super(SMARTAgentDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.if_control = if_control

        input_dim_x_a = 2
        input_dim_r_t = 4
        input_dim_r_pt2a = 3
        input_dim_r_a2a = 3
        input_dim_token = 8

        ######################## embedding layers 嵌入层 ########################
        # agent type: 类型嵌入 (type_a_emb)：用于嵌入智能体的类型。
        self.type_a_emb = nn.Embedding(4, hidden_dim)

        # agent shape: 形状嵌入 (shape_emb)：用于嵌入智能体的形状。
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)

        # agent position: 将智能体的位置特征转换为傅里叶嵌入向量 具体来说，它是一个 FourierEmbedding 层，负责将输入的二维位置特征（如 x 和 y 坐标）转换为高维的嵌入表示。
        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        
        # agent, time: 将时间维度上的相对位置特征转换为傅里叶嵌入表示。具体来说，它处理的是智能体在不同时间步之间的相对位置和方向信息。
        self.r_t_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        
        # point 2 agent: 将地图到智能体的相对位置特征转换为傅里叶嵌入表示。具体来说，它处理的是地图元素（如道路、建筑物等）与智能体之间的相对位置和方向信息。
        self.r_pt2a_emb = FourierEmbedding(input_dim=input_dim_r_pt2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        
        # agent 2 agent: 将智能体之间的相对位置特征转换为傅里叶嵌入表示。具体来说，它处理的是不同智能体之间的相对位置和方向信息。
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        
        # 用于嵌入token数据
        self.token_emb_veh = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.token_emb_ped = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.token_emb_cyc = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        
        # 融合嵌入 (fusion_emb)：用于融合不同特征的嵌入。
        self.fusion_emb = MLPEmbedding(input_dim=self.hidden_dim * 2, hidden_dim=self.hidden_dim)

        ######################## attention layers 注意力层 ########################
        # time 时间注意力层：用于处理时间维度上的注意力
        self.t_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        # point 2 agent 地图到智能体注意力层：用于处理地图到智能体的注意力
        self.pt2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        # agent 2 agent 智能体到智能体注意力层 (a2a_attn_layers)：用于处理智能体之间的注意力
        self.a2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )

        ######################## prediction head 预测头 ########################
        self.token_size = token_size
        self.token_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=self.token_size)
        

        self.trajectory_token = token_data['token']# 2048，4，2 训练用
        self.trajectory_token_traj = token_data['traj'] # 这个没用
        self.trajectory_token_all = {k:v.astype(np.float64) for k, v in token_data['token_all'].items()} # 2048，6（时间维度？），4，2推理用
        self.apply(weight_init)
        self.shift = 5
        self.beam_size = 5
        self.hist_mask = True

    def transform_rel(self, token_traj, prev_pos, prev_heading=None):
        if prev_heading is None:
            diff_xy = prev_pos[:, :, -1, :] - prev_pos[:, :, -2, :]
            prev_heading = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])

        num_agent, num_step, traj_num, traj_dim = token_traj.shape
        cos, sin = prev_heading.cos(), prev_heading.sin()
        rot_mat = torch.zeros((num_agent, num_step, 2, 2), device=prev_heading.device)
        rot_mat[:, :, 0, 0] = cos
        rot_mat[:, :, 0, 1] = -sin
        rot_mat[:, :, 1, 0] = sin
        rot_mat[:, :, 1, 1] = cos
        agent_diff_rel = torch.bmm(token_traj.view(-1, traj_num, 2), rot_mat.view(-1, 2, 2)).view(num_agent, num_step, traj_num, traj_dim)
        agent_pred_rel = agent_diff_rel + prev_pos[:, :, -1:, :]
        return agent_pred_rel
    


    ######################## embedding 嵌入 ########################

    def agent_token_embedding(self, data, agent_category, agent_token_index, pos_a, head_vector_a, inference=False):
        num_agent, num_step, traj_dim = pos_a.shape
        motion_vector_a = torch.cat([pos_a.new_zeros(data['agent']['num_nodes'], 1, self.input_dim),
                                     pos_a[:, 1:] - pos_a[:, :-1]], dim=1)

        ###### agent token embedding
        agent_type = data['agent']['type']
        veh_mask = (agent_type == 0)
        cyc_mask = (agent_type == 2)
        ped_mask = (agent_type == 1)

        trajectory_token_veh = torch.tensor(self.trajectory_token['veh']).clone().to(pos_a.device).to(torch.float)
        self.agent_token_emb_veh = self.token_emb_veh(trajectory_token_veh.view(trajectory_token_veh.shape[0], -1)) # veh

        trajectory_token_ped = torch.tensor(self.trajectory_token['ped']).clone().to(pos_a.device).to(torch.float)
        self.agent_token_emb_ped = self.token_emb_ped(trajectory_token_ped.view(trajectory_token_ped.shape[0], -1)) # ped
        
        trajectory_token_cyc = torch.tensor(self.trajectory_token['cyc']).clone().to(pos_a.device).to(torch.float)
        self.agent_token_emb_cyc = self.token_emb_cyc(trajectory_token_cyc.view(trajectory_token_cyc.shape[0], -1)) # cyc

        if inference:
            agent_token_traj_all = torch.zeros((num_agent, self.token_size, self.shift + 1, 4, 2), device=pos_a.device)
            trajectory_token_all_veh = torch.tensor(self.trajectory_token_all['veh']).clone().to(pos_a.device).to(
                torch.float)
            trajectory_token_all_ped = torch.tensor(self.trajectory_token_all['ped']).clone().to(pos_a.device).to(
                torch.float)
            trajectory_token_all_cyc = torch.tensor(self.trajectory_token_all['cyc']).clone().to(pos_a.device).to(
                torch.float)
            agent_token_traj_all[veh_mask] = torch.cat(
                [trajectory_token_all_veh[:, :self.shift], trajectory_token_veh[:, None, ...]], dim=1)
            agent_token_traj_all[ped_mask] = torch.cat(
                [trajectory_token_all_ped[:, :self.shift], trajectory_token_ped[:, None, ...]], dim=1)
            agent_token_traj_all[cyc_mask] = torch.cat(
                [trajectory_token_all_cyc[:, :self.shift], trajectory_token_cyc[:, None, ...]], dim=1)

        agent_token_emb = torch.zeros((num_agent, num_step, self.hidden_dim), device=pos_a.device)
        agent_token_emb[veh_mask] = self.agent_token_emb_veh[agent_token_index[veh_mask]]
        agent_token_emb[ped_mask] = self.agent_token_emb_ped[agent_token_index[ped_mask]]
        agent_token_emb[cyc_mask] = self.agent_token_emb_cyc[agent_token_index[cyc_mask]]

        agent_token_traj = torch.zeros((num_agent, num_step, self.token_size, 4, 2), device=pos_a.device)
        agent_token_traj[veh_mask] = trajectory_token_veh
        agent_token_traj[ped_mask] = trajectory_token_ped
        agent_token_traj[cyc_mask] = trajectory_token_cyc

        vel = data['agent']['token_velocity']

        ####### type and shape embedding
        categorical_embs = [
            self.type_a_emb(data['agent']['type'].long()).repeat_interleave(repeats=num_step,
                                                                            dim=0),

            self.shape_emb(data['agent']['shape'][:, self.num_historical_steps - 1, :]).repeat_interleave(
                repeats=num_step,
                dim=0)
        ]
        feature_a = torch.stack(
            [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),
             ], dim=-1)

        ###### agent position embedding
        x_a = self.x_a_emb(continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
                           categorical_embs=categorical_embs)
        x_a = x_a.view(-1, num_step, self.hidden_dim)

        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)

        if inference:
            return feat_a, agent_token_traj, agent_token_traj_all, agent_token_emb, categorical_embs
        else:
            return feat_a, agent_token_traj

    # not used
    # def agent_predict_next(self, data, agent_category, feat_a):
    #     num_agent, num_step, traj_dim = data['agent']['token_pos'].shape
    #     agent_type = data['agent']['type']
    #     veh_mask = (agent_type == 0)  # * agent_category==3
    #     cyc_mask = (agent_type == 2)  # * agent_category==3
    #     ped_mask = (agent_type == 1)  # * agent_category==3
    #     token_res = torch.zeros((num_agent, num_step, self.token_size), device=agent_category.device)
    #     token_res[veh_mask] = self.token_predict_head(feat_a[veh_mask])
    #     token_res[cyc_mask] = self.token_predict_cyc_head(feat_a[cyc_mask])
    #     token_res[ped_mask] = self.token_predict_walker_head(feat_a[ped_mask])
    #     return token_res

    # # not used
    # def agent_predict_next_inf(self, data, agent_category, feat_a):
    #     num_agent, traj_dim = feat_a.shape
    #     agent_type = data['agent']['type']

    #     veh_mask = (agent_type == 0)  # * agent_category==3
    #     cyc_mask = (agent_type == 2)  # * agent_category==3
    #     ped_mask = (agent_type == 1)  # * agent_category==3

    #     token_res = torch.zeros((num_agent, self.token_size), device=agent_category.device)
    #     token_res[veh_mask] = self.token_predict_head(feat_a[veh_mask])
    #     token_res[cyc_mask] = self.token_predict_cyc_head(feat_a[cyc_mask])
    #     token_res[ped_mask] = self.token_predict_walker_head(feat_a[ped_mask])

    #     return token_res
    
    ######################## build edge: ??? ########################

    def build_temporal_edge(self, pos_a, head_a, head_vector_a, num_agent, mask, inference_mask=None):
        pos_t = pos_a.reshape(-1, self.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)
        hist_mask = mask.clone()

        if self.hist_mask and self.training:
            hist_mask[
                torch.arange(mask.shape[0]).unsqueeze(1), torch.randint(0, mask.shape[1], (num_agent, 10))] = False
            mask_t = hist_mask.unsqueeze(2) & hist_mask.unsqueeze(1)
        elif inference_mask is not None:
            mask_t = hist_mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        else:
            mask_t = hist_mask.unsqueeze(2) & hist_mask.unsqueeze(1)

        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
             rel_head_t,
             edge_index_t[0] - edge_index_t[1]], dim=-1)
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index_t, r_t

    def build_interaction_edge(self, pos_a, head_a, head_vector_a, batch_s, mask_s):
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        edge_index_a2a = radius_graph(x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False,
                                      max_num_neighbors=300)
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
             rel_head_a2a], dim=-1)
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)
        return edge_index_a2a, r_a2a

    def build_map2agent_edge(self, data, num_step, agent_category, pos_a, head_a, head_vector_a, mask,
                             batch_s, batch_pl):
        mask_pl2a = mask.clone()
        mask_pl2a = mask_pl2a.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        pos_pl = data['pt_token']['position'][:, :self.input_dim].contiguous()
        orient_pl = data['pt_token']['orientation'].contiguous()
        pos_pl = pos_pl.repeat(num_step, 1)
        orient_pl = orient_pl.repeat(num_step)
        edge_index_pl2a = radius(x=pos_s[:, :2], y=pos_pl[:, :2], r=self.pl2a_radius,
                                 batch_x=batch_s, batch_y=batch_pl, max_num_neighbors=300)
        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
        r_pl2a = torch.stack(
            [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),
             rel_orient_pl2a], dim=-1)
        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        return edge_index_pl2a, r_pl2a

    def forward(self,
                data: HeteroData,
                map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pos_a = data['agent']['token_pos']
        head_a = data['agent']['token_heading']
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        num_agent, num_step, traj_dim = pos_a.shape
        agent_category = data['agent']['category']
        agent_token_index = data['agent']['token_idx']

        ################# 生成智能体的token嵌入： agent token embedding #####################
        feat_a, agent_token_traj = self.agent_token_embedding(data, agent_category, agent_token_index, pos_a, head_vector_a)
        # feat_a: [N, 18, 128]
        # agent_token_traj: [N, 18, 2048, 4, 2]

        agent_valid_mask = data['agent']['agent_valid_mask'].clone()
        # eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1]
        # agent_valid_mask[~eval_mask] = False

        ################################## 构建边 ##################################
        # 1 调用 build_temporal_edge 方法，构建时间维度上的边。
        mask = agent_valid_mask
        edge_index_t, r_t = self.build_temporal_edge(pos_a, head_a, head_vector_a, num_agent, mask)

        if isinstance(data, Batch):
            batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * t
                                 for t in range(num_step)], dim=0)
            batch_pl = torch.cat([data['pt_token']['batch'] + data.num_graphs * t
                                  for t in range(num_step)], dim=0)
        else:
            batch_s = torch.arange(num_step,
                                   device=pos_a.device).repeat_interleave(data['agent']['num_nodes'])
            batch_pl = torch.arange(num_step,
                                    device=pos_a.device).repeat_interleave(data['pt_token']['num_nodes'])

        # 2 调用 build_interaction_edge 方法，构建智能体之间的交互边。
        mask_s = mask.transpose(0, 1).reshape(-1)
        edge_index_a2a, r_a2a = self.build_interaction_edge(pos_a, head_a, head_vector_a, batch_s, mask_s)

        # 3 调用 build_map2agent_edge 方法，构建地图到智能体的边。
        mask[agent_category != 3] = False
        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(data, num_step, agent_category, pos_a, head_a,
                                                            head_vector_a, mask, batch_s, batch_pl)

        ########################## attention layers 注意力机制 ############################
        for i in range(self.num_layers): # num_layers=6, 

            # 1 时间注意力层
            feat_a = feat_a.reshape(-1, self.hidden_dim)
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
            
            # 2 point 2 agent 注意力层
            feat_a = feat_a.reshape(-1, num_step,
                                    self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            feat_a = self.pt2a_attn_layers[i]((map_enc['x_pt'].repeat_interleave(
                repeats=num_step, dim=0).reshape(-1, num_step, self.hidden_dim).transpose(0, 1).reshape(
                    -1, self.hidden_dim), feat_a), r_pl2a, edge_index_pl2a)
            
            # 3 agent 2 agent 注意力层
            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.reshape(num_step, -1, self.hidden_dim).transpose(0, 1)


        ########################## token prediction head ############################
        num_agent, num_step, hidden_dim, traj_num, traj_dim = agent_token_traj.shape
        next_token_prob = self.token_predict_head(feat_a)
        next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1)
        _, next_token_idx = torch.topk(next_token_prob_softmax, k=10, dim=-1)

        next_token_index_gt = agent_token_index.roll(shifts=-1, dims=1)
        next_token_eval_mask = mask.clone()
        next_token_eval_mask = next_token_eval_mask * next_token_eval_mask.roll(shifts=-1, dims=1) * next_token_eval_mask.roll(shifts=1, dims=1)
        next_token_eval_mask[:, -1] = False

        return {'x_a': feat_a,
                'next_token_idx': next_token_idx,
                'next_token_prob': next_token_prob,
                'next_token_idx_gt': next_token_index_gt,
                'next_token_eval_mask': next_token_eval_mask,
                }

    def inference(self,
                  data: HeteroData,
                  map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # TOTAL_STEPS = 301
        # self.num_historical_steps=11， self.shift5

        eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1] # [N]
        pos_a = data['agent']['token_pos'].clone() # [N, T, 2]
        head_a = data['agent']['token_heading'].clone() # [N, T]
        num_agent, num_step, traj_dim = pos_a.shape
        pos_a[:, (self.num_historical_steps - 1) // self.shift:] = 0 # 将pos_a这个tensor除了history的部分都清空，后面填充？
        head_a[:, (self.num_historical_steps - 1) // self.shift:] = 0 # 将head_a这个tensor除了history的部分都清空，后面填充？
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1) # [N, T, 2]

        ### agent valid mask
        agent_valid_mask = data['agent']['agent_valid_mask'].clone() # [N, T]
        agent_valid_mask[:, (self.num_historical_steps - 1) // self.shift:] = True # 将除了history的部分都设为valid，即往后全都要预测
        agent_valid_mask[~eval_mask] = False # 那些不需要evaluate的车子全视作invalid

        # agent_valid_mask = torch.ones((data['agent']['valid_mask'].shape[0], TOTAL_STEPS//(10)*2), dtype=torch.bool)

        ################# 生成智能体的token嵌入： agent token embedding #####################
        agent_token_index = data['agent']['token_idx'] # [N, T], ground truth
        agent_category = data['agent']['category'] # [N]
        feat_a, agent_token_traj, agent_token_traj_all, agent_token_emb, categorical_embs = self.agent_token_embedding(
            data,
            agent_category,
            agent_token_index,
            pos_a,
            head_vector_a,
            inference=True)
        # feat_a: [N, T, 128]; agent_token_traj: [N, T, 2048, 4, 2]; agent_token_traj_all: [N, T, 2048, 6, 4, 2]
        # agent_token_emb: [N, T, 128]; # categorial_embs: [[252, 128]] (?)

        agent_type = data["agent"]["type"] # [N]
        veh_mask = (agent_type == 0)  # * agent_category==3
        cyc_mask = (agent_type == 2)  # * agent_category==3
        ped_mask = (agent_type == 1)  # * agent_category==3
        av_mask = data["agent"]["av_index"]

        self.num_recurrent_steps_val = data["agent"]['position'].shape[1]-self.num_historical_steps
        # a number, when default, it is 80, 表示后续的0.1Hz的step数
        # self.num_recurrent_steps_val = TOTAL_STEPS-self.num_historical_steps # 总共30秒
        pred_traj = torch.zeros(data["agent"].num_nodes, self.num_recurrent_steps_val, 2, device=feat_a.device) # [N, TT, 2]，0.1Hz，存放预测出来的轨迹
        pred_head = torch.zeros(data["agent"].num_nodes, self.num_recurrent_steps_val, device=feat_a.device) # [N, TT] 0.1Hz，存放预测出来的heading
        pred_prob = torch.zeros(data["agent"].num_nodes, self.num_recurrent_steps_val // self.shift, device=feat_a.device) # [N, T-2]，0.5Hz，存放预测的token的概率


        ###################### main loop over time ######################
        next_token_idx_list = []
        mask = agent_valid_mask.clone() # [N, T]
        feat_a_t_dict = {}
        ### 
        
        if self.if_control:
            control_token_idx = 31  # 最短轨迹

            # 有效的车辆
            inference_count = mask.sum(dim=1)  # [N]
            inference_count[~veh_mask] = -1  # 非车辆设为无效

            # 计算平均速度（欧氏范数）
            velocity = data['agent']['token_velocity']  # [N, T//5, 2]
            mean_speed = velocity.norm(dim=2).mean(dim=1)  # [N]

            # 设置一个速度阈值，例如平均速度 > 2.0 m/s
            speed_threshold = 4.0
            fast_mask = mean_speed > speed_threshold

            # 联合条件：是车辆且速度快
            valid_mask = veh_mask & fast_mask
            inference_count[~valid_mask] = -1  # 其他设为无效

            if (inference_count > -1).any():
                # 如果有有效的车辆，选出有效车辆中推理次数最多的
                control_index = torch.argmax(inference_count).item()
            else:
                # 如果没有有效车辆，选择速度最快的车辆
                max_speed_index = torch.argmax(mean_speed).item()
                control_index = max_speed_index
        
        
        for t in range(self.num_recurrent_steps_val // self.shift): # 0.5Hz, default t = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            ################################## 构建边 ##################################
            # In the inference stage, we only infer the current stage for recurrent
            # 1 调用 build_temporal_edge 方法，构建时间维度上的边。
            # 输出: edge_index_t: [2, 11]; r_t: [11, 128], related to map
            if t == 0:
                inference_mask = mask.clone() # [N,T//5]
                inference_mask[:, (self.num_historical_steps - 1) // self.shift + t:] = False # 若t刚开始
            else:
                inference_mask = torch.zeros_like(mask)
                inference_mask[:, (self.num_historical_steps - 1) // self.shift + t - 1] = True # 否则，只将当前那0.5Hz的step设成true（即，只infer当前step）
            edge_index_t, r_t = self.build_temporal_edge(pos_a, head_a, head_vector_a, num_agent, mask, inference_mask)
            
            # 2 调用 build_map2agent_edge 方法，构建地图到智能体的边。
            # 输出：edge_index_pl2a: [2, #related to map] r_pl2a: [#related to map, 128]
            if isinstance(data, Batch):
                batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * t
                                     for t in range(num_step)], dim=0) # [252] = [N * T]
                batch_pl = torch.cat([data['pt_token']['batch'] + data.num_graphs * t
                                      for t in range(num_step)], dim=0) # [#related to map]
            else:
                batch_s = torch.arange(num_step,
                                       device=pos_a.device).repeat_interleave(data['agent']['num_nodes'])
                batch_pl = torch.arange(num_step,
                                        device=pos_a.device).repeat_interleave(data['pt_token']['num_nodes'])
           
            edge_index_pl2a, r_pl2a = self.build_map2agent_edge(data, num_step, agent_category, pos_a, head_a,
                                                                head_vector_a,
                                                                inference_mask, batch_s,
                                                                batch_pl)
            
            # 3 调用 build_interaction_edge 方法，构建智能体之间的交互边。
            # 输出：edge_index_a2a: [2, 196?] r_a2a: [196?, 128]
            mask_s = inference_mask.transpose(0, 1).reshape(-1) #[252] = [N * T]，将inference mask展开
            edge_index_a2a, r_a2a = self.build_interaction_edge(pos_a, head_a, head_vector_a,
                                                                batch_s, mask_s)
            

            ########################## attention layers 注意力机制 ############################
            for i in range(self.num_layers):
                if i in feat_a_t_dict:
                    feat_a = feat_a_t_dict[i]
                # 1 时间注意力层
                # 输入：feat_a（agent特征，来自agent token embedding）, r_t（来自temporal edge）, edge_index_t（来自temporal edge）
                # 输出：更新后的feat_a
                feat_a = feat_a.reshape(-1, self.hidden_dim) # 转换形状为 [N*T, 128]
                feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
                
                # 2 point 2 agent 注意力层
                # 输入：feat_a, map_encoding(来自地图), r_pl2a（来自map2agent edge）,edge_index_pl2a（来自map2agent edge）
                # 输出：更新后的feat_a
                feat_a = feat_a.reshape(-1, num_step,
                                        self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                feat_a = self.pt2a_attn_layers[i]((map_enc['x_pt'].repeat_interleave(
                    repeats=num_step, dim=0).reshape(-1, num_step, self.hidden_dim).transpose(0, 1).reshape(
                        -1, self.hidden_dim), feat_a), r_pl2a, edge_index_pl2a) # [N*T, 128]

                # 3 agent 2 agent 注意力层
                # 输入：feat_a，r_a2a（来自interaction_edge），edge_index_a2a（来自interaction_edge）
                # 输出：更新后的feat_a
                feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
                feat_a = feat_a.reshape(num_step, -1, self.hidden_dim).transpose(0, 1) # [N, T, 128]

                if i+1 not in feat_a_t_dict:
                    feat_a_t_dict[i+1] = feat_a
                else:
                    feat_a_t_dict[i+1][:, (self.num_historical_steps - 1) // self.shift - 1 + t] = feat_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t]

            ########################## token prediction head ############################
            next_token_prob = self.token_predict_head(feat_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t])
            # 输入：feat_a的[N,T,128]一个step，[N,128] -->> next_token_prob [14,2048]

            next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1) # [N, 2048]

            topk_prob, next_token_idx = torch.topk(next_token_prob_softmax, k=self.beam_size, dim=-1) # [N, topk] 和[N, topk]，分别表示概率前topk的token的概率以及索引值

            ########################## 以下是从next token复原出轨迹信息 ############################
            ### 随机控制一辆车
            if self.if_control:
                
                # 把选好的index 赋值给next_token_idx,(赋值topk个一样的，抽取流程不修改)
                next_token_idx[control_index] = torch.full(
                                                (self.beam_size,),
                                                control_token_idx, # token index，来自提前选择
                                                dtype=torch.long, 
                                                device=next_token_idx.device
                                            )
                

            expanded_index = next_token_idx[..., None, None, None].expand(-1, -1, 6, 4, 2) # [N, topk, 6, 4, 2]
            next_token_traj = torch.gather(agent_token_traj_all, 1, expanded_index) # [N, topk, 6, 4, 2]

            # 复原theta
            theta = head_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t] # [N]，head_a的其中一个time step，这个time step是当前要预测的这个的上一个
            cos, sin = theta.cos(), theta.sin()
            rot_mat = torch.zeros((num_agent, 2, 2), device=theta.device)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            agent_diff_rel = torch.bmm(next_token_traj.view(-1, 4, 2),
                                       rot_mat[:, None, None, ...].repeat(1, self.beam_size, self.shift + 1, 1, 1).view(
                                           -1, 2, 2)).view(num_agent, self.beam_size, self.shift + 1, 4, 2) # [N, topk, 6, 4, 2]
            agent_pred_rel = agent_diff_rel + pos_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t, :][:, None, None, None, ...] # [N,topk, 6, 4, 2]，加上当前的pos_a

            sample_index = torch.multinomial(topk_prob, 1).to(agent_pred_rel.device) # [N, 1], 从topk中采样，赋值
            agent_pred_rel = agent_pred_rel.gather(dim=1,
                                                   index=sample_index[..., None, None, None].expand(-1, -1, 6, 4,
                                                                                                    2))[:, 0, ...] # 选出topk采样的最优解
            pred_prob[:, t] = topk_prob.gather(dim=-1, index=sample_index)[:, 0] # [N,T//5-2]，赋值-1
            if self.if_control:
                # 把控制车的概率改为-1，方便返回区分
                pred_prob[control_index, t] = -1
                
                
            pred_traj[:, t * 5:(t + 1) * 5] = agent_pred_rel[:, 1:, ...].clone().mean(dim=2) # [N, T, 2]
            diff_xy = agent_pred_rel[:, 1:, 0, :] - agent_pred_rel[:, 1:, 3, :]
            pred_head[:, t * 5:(t + 1) * 5] = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])

            pos_a[:, (self.num_historical_steps - 1) // self.shift + t] = agent_pred_rel[:, -1, ...].clone().mean(dim=1)
            diff_xy = agent_pred_rel[:, -1, 0, :] - agent_pred_rel[:, -1, 3, :]
            theta = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])
            head_a[:, (self.num_historical_steps - 1) // self.shift + t] = theta
            next_token_idx = next_token_idx.gather(dim=1, index=sample_index)
            next_token_idx = next_token_idx.squeeze(-1)
            next_token_idx_list.append(next_token_idx[:, None])

            # agent_token_emb: [N, T, 128]
            agent_token_emb[veh_mask, (self.num_historical_steps - 1) // self.shift + t] = self.agent_token_emb_veh[
                next_token_idx[veh_mask]]
            agent_token_emb[ped_mask, (self.num_historical_steps - 1) // self.shift + t] = self.agent_token_emb_ped[
                next_token_idx[ped_mask]]
            agent_token_emb[cyc_mask, (self.num_historical_steps - 1) // self.shift + t] = self.agent_token_emb_cyc[
                next_token_idx[cyc_mask]]
            motion_vector_a = torch.cat([pos_a.new_zeros(data['agent']['num_nodes'], 1, self.input_dim),
                                         pos_a[:, 1:] - pos_a[:, :-1]], dim=1)

            head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1) # [N, T, 2]

            # 复原velocity
            vel = motion_vector_a.clone() / (0.1 * self.shift) # [N, T, 2]
            vel[:, (self.num_historical_steps - 1) // self.shift + 1 + t:] = 0 # 当前所预测的时刻往后那些都设为0
            motion_vector_a[:, (self.num_historical_steps - 1) // self.shift + 1 + t:] = 0
            
            # x_a
            x_a = torch.stack(
                [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2])], dim=-1)

            x_a = self.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)),
                               categorical_embs=categorical_embs)
            x_a = x_a.view(-1, num_step, self.hidden_dim)

            feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
            feat_a = self.fusion_emb(feat_a)
        ###################### end: main loop over time ######################

        agent_valid_mask[agent_category != 3] = False

        return {
            'pos_a': pos_a[:, (self.num_historical_steps - 1) // self.shift:], # [N, T, 2] -> [N, T-2, 2]
            'head_a': head_a[:, (self.num_historical_steps - 1) // self.shift:], # [N, T] -> [N, T-2]
            'gt': data['agent']['position'][:, self.num_historical_steps:, :self.input_dim].contiguous(), # [N, 80, 2]
            'valid_mask': agent_valid_mask[:, (self.num_historical_steps - 1) // self.shift:], #这句话是错的
            'pred_traj': pred_traj, # [N, 80, 2]
            'pred_head': pred_head, # [N, 80]
            'next_token_idx': torch.cat(next_token_idx_list, dim=-1), # [N, T-2]
            'next_token_idx_gt': agent_token_index.roll(shifts=-1, dims=1), # [N, T]
            'next_token_eval_mask': data['agent']['agent_valid_mask'],
            'pred_prob': pred_prob,
            'vel': vel # [N, T, 2]
        }

    