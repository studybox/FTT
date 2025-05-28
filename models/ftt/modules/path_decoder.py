import torch
import torch.nn as nn
from models.laneGCN.layers import LinRes

class PathPredNet(nn.Module):
    def __init__(self, config, latent=True):
        super(PathPredNet, self).__init__()
        self.config = config
        self.latent = latent
        if latent:
            size = config["n_actor"]+config["n_map"]+config["act_continuous_size"]+config["pat_continuous_size"]
        else:
            size = config["n_actor"]+config["n_map"]

        self.logit = nn.Sequential(
            LinRes(size, size//2, norm="GN", ng=1),
            nn.Linear(size//2, 1),
        )
    def forward(self, actors, paths, Z_act, Z_pat, graph):
        if self.latent:
            paths = torch.cat([paths, Z_pat, actors[graph["veh->path"]["u"]], Z_act[graph["veh->path"]["u"]]], -1)
        else:
            paths = torch.cat([paths, actors[graph["veh->path"]["u"]]], -1)
        logits = self.logit(paths)
        return logits