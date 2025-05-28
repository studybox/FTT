import torch
import torch.nn as nn

from torch.nn import functional as F
from models.ftt.layers import TransformerConv
from models.laneGCN.layers import LinRes

class TargetsPredAttNet(nn.Module):
    def __init__(self, config):
        super(TargetsPredAttNet, self).__init__()
        self.config = config
        self.output_dim = 2
        self.num_preds = config["num_preds"]
        dropout = config["args"].dropout
        self.num_blocks = config["num_blocks"]

        self.node_blocks = nn.ModuleList(
            [
                TransformerConv(config["n_map"],
                                config["n_map"],
                                heads=config["n_head"],
                                dropout=dropout)
                for _ in range(config["num_blocks"])
            ]
        )
        self.target_blocks = nn.ModuleList(
            [
                TransformerConv((config["n_map"], config["n_actor"]),
                                config["n_actor"],
                                heads=config["n_head"],
                                dropout=dropout)
                for _ in range(config["num_blocks"])
            ]
        )

        self.hidden2target = nn.Sequential(
            LinRes(config["n_actor"]+2, config["n_actor"], norm="GN", ng=1),
            nn.Linear(config["n_actor"], 1)
        )

        self.hidden2offset = nn.Sequential(
            LinRes(config["n_actor"]+2, config["n_actor"], norm="GN", ng=1),
            nn.Linear(config["n_actor"], 2)
        )

        self.path_blocks = nn.ModuleList(
                [
                   TransformerConv(config["n_path"],
                                   config["n_path"],
                                   heads=config["n_head"],
                                   dropout=dropout)
                   for _ in range(config["num_blocks"])
                ]
        )
        self.traj_blocks = nn.ModuleList(
            [
                TransformerConv((config["n_path"], config["n_actor"]),
                                config["n_actor"],
                                heads=config["n_head"],
                                dropout=dropout)
                for _ in range(config["num_blocks"])
            ]
        )
        self.spatial_embedding = nn.Linear(3, config["n_path"])

        self.hidden2pos = nn.Sequential(
            LinRes(config["n_actor"], config["n_actor"], norm="GN", ng=1),
            nn.Linear(config["n_actor"], 2),
        )

    def forward(self, actors, nodes, initial_poses, target_poses_gt, candidates,
                node_edge_index, node_target_edge_index,
                traj_edge_index, node_traj_edge_index, actor_traj_index, ranks=None):
        batch = actors.size(0)
        # give nodes a sequential order info
        for block in self.node_blocks:
            nodes, ia = block(nodes, node_edge_index, return_attention_weights=True)

        targets = actors
        for block in self.target_blocks:
            targets = block((nodes, targets), node_target_edge_index)

        n = candidates.size(1)
        target_cands = torch.cat([targets.unsqueeze(1).repeat(1, n, 1), candidates], dim=2)
        target_logits = self.hidden2target(target_cands.reshape(batch*n, -1))
        target_logits = target_logits.reshape(batch, n)
        target_offsets = self.hidden2offset(target_cands.reshape(batch*n, -1))
        target_offsets = target_offsets.reshape(batch, n, 2)
        # make K-th prediction
        row_idcs = torch.arange(batch).cuda()
        _, top_idcs = target_logits.topk(self.config['num_mods'], dim=1)
        if ranks is None:
            col_idcs = top_idcs[:, 0]
        else:
            col_idcs = top_idcs[row_idcs, ranks]
        target_poses = candidates[row_idcs, col_idcs] + target_offsets[row_idcs, col_idcs]
        target_probs = F.softmax(target_logits, dim=-1)
        target_prob = target_probs[row_idcs, col_idcs]
        # give paths a sequential order info
        if self.training:
            paths = self.generate_frenet_path(initial_poses, target_poses_gt)
        else:
            paths = self.generate_frenet_path(initial_poses, target_poses)
        paths = self.spatial_embedding(paths)
        for block in self.path_blocks:
            paths = block(paths, traj_edge_index)

        trajs = actors[actor_traj_index] + paths
        for block in self.traj_blocks:
            trajs = block((nodes, trajs), node_traj_edge_index)
        trajs = torch.reshape(self.hidden2pos(trajs), [-1, self.config["num_preds"], 2])
        return trajs, target_poses, target_prob, target_logits, target_offsets, target_probs

    def generate_frenet_path(self, initial_poses, target_poses):
        T = self.config["dT"] * self.config["num_preds"]
        s = target_poses[:,0]
        vs0 = initial_poses[:,0]
        sa = 2*(s - vs0*T)/T**2

        d = target_poses[:,1]
        d0 = initial_poses[:, 1]
        vd0 = initial_poses[:, 2]

        da = 2*(d-d0 - vd0*T)/T**2

        pe = torch.zeros(initial_poses.size(0), self.config["num_preds"], 3).cuda()
        t = torch.arange(1, self.config["num_preds"]+1).cuda() * self.config["dT"]
        pe[:, :, 0] = t.view(1,-1)*torch.ones_like(s).view(-1,1)
        pe[:, :, 1] = t.view(1,-1)*vs0.view(-1,1) + (t**2).view(1,-1)*0.5*sa.view(-1,1)
        pe[:, :, 2] = t.view(1,-1)*vd0.view(-1,1) + (t**2).view(1,-1)*0.5*da.view(-1,1) + d0.view(-1,1)
        return torch.reshape(pe, [-1, 3])