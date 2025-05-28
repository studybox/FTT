import torch 
import torch.nn as nn
import torch.nn.functional as F
from models.laneGCN.layers import LinRes
from torch_scatter import scatter_max
import math
import numpy as np


class SubGraph(nn.Module):
    def __init__(self, in_channels, num_subgraph_layres=3, hidden_unit=64):
        super(SubGraph, self).__init__()
        self.num_subgraph_layres = num_subgraph_layres
        self.hidden_unit = hidden_unit
        self.out_channels = hidden_unit

        lane_layers = []
        veh_layers = []
        veh_in_channels, lane_in_channels  = in_channels
        for i in range(num_subgraph_layres):
            lane_layers.append(LinRes(lane_in_channels, hidden_unit, norm="GN", ng=1))
            lane_in_channels = hidden_unit * 2
        self.lane_layers = nn.ModuleList(lane_layers)

        for i in range(num_subgraph_layres):
            veh_layers.append(LinRes(veh_in_channels, hidden_unit, norm="GN", ng=1))
            veh_in_channels = hidden_unit * 2
        self.veh_layers = nn.ModuleList(veh_layers)

        self.lane_linear = nn.Linear(hidden_unit * 2, hidden_unit)
        self.veh_linear = nn.Linear(hidden_unit * 2, hidden_unit)

    def forward(self, sub_data):
        lane_x = sub_data.lane_feat
        veh_x = sub_data.veh_feat
        lane_cluster = sub_data.lane_cluster
        veh_cluster = sub_data.veh_cluster
        batch_lane = sub_data.batch_lane
        batch_veh = sub_data.batch_veh

        for layer in self.lane_layers:
            lane_x = layer(lane_x)
            lane_agg, _ = scatter_max(lane_x, lane_cluster, dim=0, dim_size=len(batch_lane))
            lane_x = torch.cat([lane_x, lane_agg[lane_cluster]], dim=-1)
        lane_x = self.lane_linear(lane_x)
        lane_x, _ = scatter_max(lane_x, lane_cluster, dim=0, dim_size=len(batch_lane))

        for layer in self.veh_layers:
            veh_x = layer(veh_x)
            veh_agg, _ = scatter_max(veh_x, veh_cluster, dim=0, dim_size=len(batch_veh))
            veh_x = torch.cat([veh_x, veh_agg[veh_cluster]], dim=-1)
        veh_x = self.veh_linear(veh_x)
        veh_x, _ = scatter_max(veh_x, veh_cluster, dim=0, dim_size=len(batch_veh))

        batch_size = batch_veh.max().item()+1

        x = []
        max_len = max(sub_data.valid_lens)
        for b in range(batch_size):
            v = veh_x[batch_veh==b]
            len_v = len(v)
            v = torch.cat([v, torch.ones(len_v,1).type('torch.cuda.FloatTensor'),
                              torch.zeros(len_v,1).type('torch.cuda.FloatTensor')], -1)
            l = lane_x[batch_lane==b]
            len_l = len(l)
            l = torch.cat([l, torch.zeros(len_l,1).type('torch.cuda.FloatTensor'),
                              torch.ones(len_l,1).type('torch.cuda.FloatTensor')], -1)

            z = torch.zeros(max_len-len_v-len_l, self.hidden_unit+2).type('torch.cuda.FloatTensor')
            x.append(torch.cat([v,l,z], dim=0))

        return torch.stack(x, dim=0)

        #return F.normalize(x, p=2.0, dim=1)

class GlobalGraph(nn.Module):
    def __init__(self, in_channels,
                 global_graph_width,
                 num_global_layers=1,
                 need_scale=False):
        super(GlobalGraph, self).__init__()
        self.in_channels = in_channels
        self.global_graph_width = global_graph_width

        layers = []

        in_channels = self.in_channels
        for i in range(num_global_layers):
            layers.append(SelfAttentionFCLayer(in_channels,
                                                 self.global_graph_width,
                                                 need_scale)
            )

            in_channels = self.global_graph_width
        self.layers = nn.ModuleList(layers)

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x


class SelfAttentionFCLayer(nn.Module):
    """
    Self-attention layer. no scale_factor d_k
    """

    def __init__(self, in_channels, global_graph_width, need_scale=False):
        super(SelfAttentionFCLayer, self).__init__()
        self.in_channels = in_channels
        self.graph_width = global_graph_width
        self.q_lin = nn.Linear(in_channels, global_graph_width)
        self.k_lin = nn.Linear(in_channels, global_graph_width)
        self.v_lin = nn.Linear(in_channels, global_graph_width)
        self.scale_factor_d = 1 + \
            int(np.sqrt(self.in_channels)) if need_scale else 1

    def forward(self, x, valid_lens):
        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)

        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.graph_width)
        attention_weights = self.masked_softmax(scores, valid_lens)
        x = torch.bmm(attention_weights, value)
        return x
    @staticmethod
    def masked_softmax(X, valid_lens):
        """
        masked softmax for attention scores
        args:
                X: 3-D tensor, valid_len: 1-D or 2-D tensor
        """
        if valid_lens is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            shape = X.shape

            # Fill masked elements with a large negative, whose exp is 0
            mask = torch.zeros_like(X, dtype=torch.bool)
            for batch_id, cnt in enumerate(valid_lens):
                mask[batch_id, :, cnt:] = True
                mask[batch_id, cnt:] = True
            X_masked = X.masked_fill(mask, -1e12)
            return nn.functional.softmax(X_masked, dim=-1) * (1 - mask.float())

class Vectornet(nn.Module):
    def __init__(self, in_channels,
                 num_subgraph_layres=3,
                 num_global_graph_layer=1,
                 subgraph_width=64,
                 global_graph_width=64
                 ):
        super(Vectornet, self).__init__()
        # some params
        self.num_subgraph_layres = num_subgraph_layres
        self.global_graph_width = global_graph_width

        # subgraph feature extractor
        self.subgraph = SubGraph(in_channels, num_subgraph_layres, subgraph_width)

        # global graph
        self.global_graph = GlobalGraph(self.subgraph.out_channels + 2,
                                        self.global_graph_width,
                                        num_global_layers=num_global_graph_layer)

    def forward(self, batch):
        """
        args:
            data (Data): [x, y, cluster, edge_index, valid_len]
        """
        batch_size = batch.num_graphs
        #time_step_len = data.time_step_len[0].int()
        valid_lens = batch.valid_lens

        #id_embedding = data.identifier

        sub_graph_out = self.subgraph(batch)

        # reconstruct the batch global interaction graph data
        global_graph_out = self.global_graph(sub_graph_out, valid_lens=valid_lens)

        veh_lens = batch.veh_lens
        out = []
        for b in range(batch_size):
            out.append(global_graph_out[b, :veh_lens[b]])

        return torch.cat(out, dim=0)

class TargetPred(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dim: int = 64,
                 m: int = 10,
                 device=torch.device("cpu")):
        super(TargetPred, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.M = m          # output candidate target

        self.device = device

        self.prob_mlp = nn.Sequential(
            nn.Linear(in_channels + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.mean_mlp = nn.Sequential(
            nn.Linear(in_channels + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, feat_in: torch.Tensor, tar_candidate: torch.Tensor, candidate_mask=None):
        """
        predict the target end position of the target agent from the target candidates
        :param feat_in:        the encoded trajectory features, [batch_size, inchannels]
        :param tar_candidate:  the target position candidate (x, y), [batch_size, N, 2]
        :param candidate_mask: the mask of valid target candidate
        :return:
        """
        # dimension must be [batch size, 1, in_channels]
        feat_in = feat_in.unsqueeze(1)
        batch_size, n, _ = tar_candidate.size()

        # stack the target candidates to the end of input feature
        feat_in_repeat = torch.cat([feat_in.repeat(1, n, 1), tar_candidate], dim=2)
        # compute probability for each candidate
        prob_tensor = self.prob_mlp(feat_in_repeat).squeeze(2)
        tar_candit_prob = self.masked_softmax(prob_tensor, candidate_mask)        # [batch_size, n_tar, 1]
        tar_offset_mean = self.mean_mlp(feat_in_repeat)                          # [batch_size, n_tar, 2]

        return tar_candit_prob, tar_offset_mean

    @staticmethod
    def masked_softmax(X, valid_lens):
        """
        masked softmax for attention scores
        args:
                X: 3-D tensor, valid_len: 1-D or 2-D tensor
        """
        #print(valid_lens)
        if valid_lens is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            shape = X.shape

            # Fill masked elements with a large negative, whose exp is 0
            mask = torch.zeros_like(X, dtype=torch.bool)
            for batch_id, cnt in enumerate(valid_lens):
                mask[batch_id, cnt:] = True
            X_masked = X.masked_fill(mask, -1e2)
            return nn.functional.log_softmax(X_masked, dim=-1)# * (1 - mask.float())

    def loss(self,
             feat_in: torch.Tensor,
             tar_candidate: torch.Tensor,
             candidate_gt: torch.Tensor,
             offset_gt: torch.Tensor,
             candidate_mask=None):
        """
        compute the loss for target prediction, classification gt is binary labels,
        only the closest candidate is labeled as 1
        :param feat_in: encoded feature for the target candidate, [batch_size, inchannels]
        :param tar_candidate: the target candidates for predicting the end position of the target agent, [batch_size, N, 2]
        :param candidate_gt: target prediction ground truth, classification gt and offset gt, [batch_size, N]
        :param offset_gt: the offset ground truth, [batch_size, 2]
        :param candidate_mask:
        :return:
        """
        batch_size, n, _ = tar_candidate.size()
        _, num_cand = candidate_gt.size()

        assert num_cand == n, "The num target candidate and the ground truth one-hot vector is not aligned: {} vs {};".format(n, num_cand)
        # pred prob and compute cls loss
        tar_candit_prob, tar_offset_mean = self.forward(feat_in, tar_candidate, candidate_mask)
        # classfication loss in n candidates
        n_candidate_loss = F.cross_entropy(tar_candit_prob.transpose(1, 2), candidate_gt.long(), reduction='sum')
        # classification loss in m selected candidates
        _, indices = tar_candit_prob[:, :, 1].topk(self.M, dim=1)
        batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(self.M)]).T
        offset_loss = F.smooth_l1_loss(tar_offset_mean[candidate_gt.bool()], offset_gt, reduction='sum')
        return n_candidate_loss + offset_loss, tar_candidate[batch_idx, indices], tar_offset_mean[batch_idx, indices]

class MotionEstimation(nn.Module):
    def __init__(self,
                 in_channels,
                 horizon=15,
                 hidden_dim=64):
        """
        estimate the trajectories based on the predicted targets
        :param in_channels:
        :param horizon:
        :param hidden_dim:
        """
        super(MotionEstimation, self).__init__()
        self.in_channels = in_channels
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        self.traj_pred = nn.Sequential(
            nn.Linear(in_channels + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * 2)
        )

    def forward(self, feat_in: torch.Tensor, loc_in: torch.Tensor):
        """
        predict the trajectory according to the target location
        :param feat_in: encoded feature vector for the target agent, torch.Tensor, [batch_size, in_channels]
        :param loc_in: end location, torch.Tensor, [batch_size, M, 2] or [batch_size, 1, 2]
        :return: [batch_size, M, horizon * 2] or [batch_size, 1, horizon * 2]
        """
        assert feat_in.size()[-1] == self.in_channels, "[MotionEstimation]: Error feature, mismatch in the feature channels!"

        batch_size, M, _ = loc_in.size()
        if M > 1:
            # target candidates
            feat_in = feat_in.unsqueeze(1)
            inputs = torch.cat([feat_in.repeat(1, M, 1), loc_in], dim=2)
        else:
            # targt ground truth
            inputs = torch.cat([feat_in, loc_in.squeeze(1)], dim=-1)

        return self.traj_pred(inputs)

    def loss(self, feat_in: torch.Tensor, loc_gt: torch.Tensor, traj_gt: torch.Tensor):
        """
        compute loss according to the ground truth target location input
        :param feat_in: feature input of the target agent, torch.Tensor, [batch_size, in_channels]
        :param loc_gt: final target location gt, torch.Tensor, [batch_size, 2]
        :param traj_gt: the gt trajectory, torch.Tensor, [batch_size, horizon * 2]
        :param reduction: reduction of the loss, str
        :return:
        """
        assert feat_in.dim() == 3, "[MotionEstimation]: Error in feature input dimension."
        assert traj_gt.dim() == 2, "[MotionEstimation]: Error in trajectory gt dimension."
        batch_size, _, _ = feat_in.size()

        traj_pred = self.forward(feat_in, loc_gt.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(traj_pred, traj_gt, reduction='sum')
        return loss

class TrajScoreSelection(nn.Module):
    def __init__(self,
                 feat_channels,
                 horizon=30,
                 hidden_dim=64,
                 temper=0.01,
                 device=torch.device("cpu")):
        """
        init trajectories scoring and selection module
        :param feat_channels: int, number of channels
        :param horizon: int, prediction horizon, prediction time x pred_freq
        :param hidden_dim: int, hidden dimension
        :param temper: float, the temperature
        """
        super(TrajScoreSelection, self).__init__()
        self.feat_channels = feat_channels
        self.horizon = horizon
        self.temper = temper

        self.device = device

        self.score_mlp = nn.Sequential(
            nn.Linear(feat_channels + horizon * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feat_in: torch.Tensor, traj_in: torch.Tensor):
        """
        forward function
        :param feat_in: input feature tensor, torch.Tensor, [batch_size, feat_channels]
        :param traj_in: candidate trajectories, torch.Tensor, [batch_size, M, horizon * 2]
        :return: [batch_size, M]
        """
        feat_in = feat_in.unsqueeze(1)
        assert feat_in.dim() == 3, "[TrajScoreSelection]: Error in input feature dimension."
        assert traj_in.dim() == 3, "[TrajScoreSelection]: Error in candidate trajectories dimension"

        batch_size, M, _ = traj_in.size()
        input_tenor = torch.cat([feat_in.repeat(1, M, 1), traj_in], dim=2)

        return F.softmax(self.score_mlp(input_tenor).squeeze(-1), dim=-1)

    def loss(self, feat_in, traj_in, traj_gt):
        """
        compute loss
        :param feat_in: input feature, torch.Tensor, [batch_size, feat_channels]
        :param traj_in: candidate trajectories, torch.Tensor, [batch_size, M, horizon * 2]
        :param traj_gt: gt trajectories, torch.Tensor, [batch_size, horizon * 2]
        :return:
        """
        # batch_size = traj_in.shape[0]

        # compute ground truth score
        score_gt = F.softmax(-distance_metric(traj_in, traj_gt)/self.temper, dim=1)
        score_pred = self.forward(feat_in, traj_in)

        logprobs = - torch.log(score_pred)

        loss = torch.sum(torch.mul(logprobs, score_gt))
        return loss

class TNTLoss(nn.Module):
    """
        The loss function for train TNT, loss = a1 * Targe_pred_loss + a2 * Traj_reg_loss + a3 * Score_loss
    """
    def __init__(self,
                 lambda1,
                 lambda2,
                 lambda3,
                 horizon,
                 m,
                 k,
                 temper=0.01,
                 reduction='sum',
                 device=torch.device("cpu")):
        """
        lambda1, lambda2, lambda3: the loss coefficient;
        temper: the temperature for computing the score gt;
        aux_loss: with the auxiliary loss or not;
        reduction: loss reduction, "sum" or "mean" (batch mean);
        """
        super(TNTLoss, self).__init__()
        self.horizon = horizon
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        #self.candidate_loss = nn.CrossEntropyLoss(reduction="none")
        self.offset_loss = nn.SmoothL1Loss(reduction="none")
        self.motion_loss = nn.SmoothL1Loss()
        #self.score_loss = nn.CrossEntropyLoss(reduction="none")


        self.m = m
        self.k = k

        self.reduction = reduction
        self.temper = temper

        self.device = device

    def forward(self, pred_dict, gt_dict):
        num_vehs = gt_dict['candidate_gt'].size()[0]
        has_preds = gt_dict['has_preds']
        has_target = has_preds[:, -1]
        last = has_preds.float() + 0.1 * torch.arange(self.horizon).float().to(
            has_preds.device
        ) / float(self.horizon)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        loss = 0.0
        #cls_loss = self.candidate_loss(pred_dict['candidate_gt'], gt_dict['candidate_gt'])
        row_idcs = torch.arange(num_vehs).to(self.device)

        cls_loss = -pred_dict['target_logprob'][row_idcs, gt_dict['candidate_gt']]
        cls_loss = cls_loss[has_target].mean()

        offset = pred_dict['offset'][row_idcs, gt_dict['candidate_gt']]
        offset_loss = self.offset_loss(offset, gt_dict['offset'])
        offset_loss = offset_loss[has_target].mean()
        loss += self.lambda1 * (cls_loss + offset_loss)

        # compute motion estimation loss
        reg_loss = self.motion_loss(pred_dict['traj_with_gt'][mask][has_preds[mask]],
                                    gt_dict['y_rel'][mask][has_preds[mask]])
        #print(reg_loss.size(), mask.size(), has_preds.size())
        loss += self.lambda2 * reg_loss

        # compute scoring gt and loss

        score_gt = F.softmax(-distance_metric(pred_dict['traj'][mask],
                                              gt_dict['y'][mask],
                                              has_preds[mask])/self.temper, dim=-1).detach()
        score_loss = torch.sum(torch.mul(- torch.log(pred_dict['score'][mask]), score_gt)) / num_vehs
        #score_loss = F.binary_cross_entropy(pred_dict['score'], score_gt, reduction='sum')
        loss += self.lambda3 * score_loss

        loss_dict = {"tar_cls_loss": cls_loss,
                     "tar_offset_loss": offset_loss,
                     "traj_loss": reg_loss,
                     "sloss": score_loss}

        return loss, loss_dict


def distance_metric(traj_candidate: torch.Tensor, traj_gt: torch.Tensor, has_preds=None):
    """
        compute the distance between the candidate trajectories and gt trajectory
        :param traj_candidate: torch.Tensor, [batch_size, M, horizon * 2] or [M, horizon * 2]
        :param traj_gt: torch.Tensor, [batch_size, horizon * 2] or [1, horizon * 2]
        :return: distance, torch.Tensor, [batch_size, M] or [1, M]
        """
    assert traj_gt.dim() == 2, "Error dimension in ground truth trajectory"
    if traj_candidate.dim() == 3:
        # batch case
        pass

    elif traj_candidate.dim() == 2:
        traj_candidate = traj_candidate.unsqueeze(1)
    else:
        raise NotImplementedError

    assert traj_candidate.size()[2] == traj_gt.size()[1], "Miss match in prediction horizon!"

    _, M, horizon_2_times = traj_candidate.size()
    dis = torch.pow(traj_candidate - traj_gt.unsqueeze(1), 2).view(-1, M, int(horizon_2_times / 2), 2)
    if has_preds is not None:
        #print(dis.size(), has_preds.size())

        dis[~has_preds.unsqueeze(1).repeat(1,M,1)] = 0
    dis, _ = torch.max(torch.sum(dis, dim=3), dim=2)

    return dis