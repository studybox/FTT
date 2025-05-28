import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.laneGCN.layers import ActorNet, MapNet, A2M, M2M, M2A, A2A, LinRes, ScoreNet
from models.laneGCN.utils.losses import LaneGCNLoss
from .utils import relative_to_abs, cal_ade, cal_fde, cal_mr, cal_diversity

class LaneGCN(pl.LightningModule):
    def __init__(self, config):
        # code adopted from https://github.com/uber-research/LaneGCN/blob/master/lanegcn.py
        super(LaneGCN, self).__init__()
        self.config = config
        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)
        self.learning_rate = config["args"].learning_rate
        self.l2_weight_decay = config["args"].l2_weight_decay
        self.learning_rate_decay = config["args"].learning_rate_decay

        self.a2m = A2M(config)
        self.m2m = M2M(config)
        self.m2a = M2A(config)
        self.a2a = A2A(config)

        if self.config["num_mods"] == 1:
            self.pred_net = nn.Sequential(
                LinRes(config["n_actor"], config["n_actor"], norm="GN", ng=1),
                nn.Linear(config["n_actor"], 2 * config["num_preds"]),
            )
        else:
            self.pred_net = nn.ModuleList(
                [nn.Sequential(
                    LinRes(config["n_actor"], config["n_actor"], norm="GN", ng=1),
                    nn.Linear(config["n_actor"], 2 * config["num_preds"]),
                ) for m in range(self.config['num_mods'])
                ]
            )
        self.score_net = ScoreNet(config)
        self.score_loss = LaneGCNLoss(config)
        self.traj_loss = nn.SmoothL1Loss()
        self.val_seen_metrics = {}
        self.val_unseen_metrics = {}

    def forward(self, batch):
        # construct actor feature
        actors =  torch.cat([batch.obs_traj_rel, batch.obs_info], -1).permute(1, 2, 0)
        actor_idcs = batch.veh_batch
        actor_ctrs = batch.veh_ctrs
        actors = self.actor_net(actors)

        # construct map features
        graph = laneGCN_graph_gather(batch["graphs"])
        nodes, node_idcs, node_ctrs = self.map_net(graph)
        # actor-map fusion cycle
        nodes = self.a2m(nodes, graph, actors, actor_idcs, actor_ctrs)
        nodes = self.m2m(nodes, graph)
        actors = self.m2a(actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)
        actors = self.a2a(actors, actor_idcs, actor_ctrs)

        # prediction
        if self.config["num_mods"] == 1:
            out = torch.reshape(self.pred_net(actors), [-1, self.config["num_preds"], 2])
        else:
            preds = []
            for m in range(self.config['num_mods']):
                preds.append(torch.reshape(self.pred_net[m](actors), [-1, self.config["num_preds"], 2]))
            score_out = self.score_net(actors, actor_idcs, actor_ctrs, preds)
            out = score_out["pred"]
            score_loss = self.score_loss(score_out, batch.fut_traj, batch.has_preds)
            score_loss = score_loss['cls_loss']/(score_loss['num_cls']+1e-10)
        # masked losses
        gt_preds = batch.fut_traj_rel.permute(1, 0, 2)
        has_preds = batch.has_preds.permute(1, 0)

        last = has_preds.float() + 0.1 * torch.arange(self.config['num_preds']).float().to(
            has_preds.device
        ) / float(self.config['num_preds'])
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0
        has_preds = has_preds[mask]

        traj_loss = self.traj_loss(out[mask][has_preds], gt_preds[mask][has_preds])

        if self.config["num_mods"] == 1:
            ret = {'pred': out.permute(1, 0, 2),
                   'loss':traj_loss}
        else:
            ret = {'pred': out.permute(1, 0, 2),
                   'loss':traj_loss+score_loss,
                   'traj_loss':traj_loss,
                   'sloss':score_loss,
                   'score':score_out['cls_unsorted'],
                   'reg':preds
                   }
        return ret

    def training_step(self, batch, batch_idx):
        ret = self.forward(batch)
        loss = ret['loss']
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss
    
    def on_validation_start(self):
        self.val_seen_metrics = initialize_metrics()
        self.val_unseen_metrics = initialize_metrics()
        

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ret = self.forward(batch)
        loss = ret['loss']
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        metrics = self.compute_metrics(ret, batch, dataloader_idx)
        val_label = "val_seen" if dataloader_idx == 0 else "val_unseen"
        self.log(f'val_minADE_{val_label}', metrics['sum_minADE@3s'][-1]/metrics['count_average@3s'][-1], prog_bar=True, on_step=False, on_epoch=True, batch_size=metrics['count_average@3s'][-1])
        self.log(f'val_minFDE_{val_label}', metrics['sum_minFDE@3s'][-1]/metrics['count_final@3s'][-1], prog_bar=True, on_step=False, on_epoch=True, batch_size=metrics['count_final@3s'][-1])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.learning_rate_decay)
        return [optimizer], [lr_scheduler]
    
    def load_params_from_file(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        if 'best_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['best_state_dict'])
        else:
            self.load_state_dict(checkpoint['state_dict'])
    
    def compute_metrics(self, predictions, batch, dataloader_idx):
        metrics = self.val_seen_metrics if dataloader_idx == 0 else self.val_unseen_metrics
        
        if "reg" in predictions:
            # K = 6
            K_pred_traj_fake_rel = predictions['reg']
            ades = [[], [], []]
            fdes = [[], [], []]
            mrs = [[], [], []]
            K_total_preds = [0, 0, 0]
            K_total_final_preds = [0, 0, 0]
            K_pred_traj_final = []
            
            pred_traj_gt = batch.fut_traj

            for k in range(self.config['num_mods']):
                pred_rel = K_pred_traj_fake_rel[k].permute(1, 0, 2)
                pred_traj = relative_to_abs(pred_rel, batch.obs_traj[-1])

                K_pred_traj_final.append(pred_traj[-1])
                for tt in range(3):
                    ade = cal_ade(
                        pred_traj_gt[:self.config["num_preds"]//3*(tt+1)-1],
                        pred_traj[:self.config["num_preds"]//3*(tt+1)-1],
                        batch.has_preds[:self.config["num_preds"]//3*(tt+1)-1], mode='raw'
                    )
                    fde = cal_fde(
                        pred_traj_gt[:self.config["num_preds"]//3*(tt+1)-1],
                        pred_traj[:self.config["num_preds"]//3*(tt+1)-1],
                        batch.has_preds[:self.config["num_preds"]//3*(tt+1)-1], mode='raw'
                    )
                    mr = cal_mr(
                        pred_traj_gt[:self.config["num_preds"]//3*(tt+1)-1],
                        pred_traj[:self.config["num_preds"]//3*(tt+1)-1],
                        batch.has_preds[:self.config["num_preds"]//3*(tt+1)-1], mode='raw'
                    )
                    ades[tt].append(ade)
                    fdes[tt].append(fde)
                    mrs[tt].append(mr)
                    K_total_preds[tt] = batch.has_preds[:self.config["num_preds"]//3*(tt+1)-1].sum()
                    K_total_final_preds[tt] = batch.has_preds[self.config["num_preds"]//3*(tt+1)-1].sum()
            for tt in range(3):
                metrics["sum_minADE@{}s".format(tt+1)].append(torch.stack(ades[tt], 1).min(1)[0].sum().item())
                metrics["sum_minFDE@{}s".format(tt+1)].append(torch.stack(fdes[tt], 1).min(1)[0].sum().item())
                metrics["sum_minMR@{}s".format(tt+1)].append(torch.stack(mrs[tt], 1).min(1)[0].sum().item())
                metrics["count_final@{}s".format(tt+1)].append(K_total_final_preds[tt].item())
                metrics["count_average@{}s".format(tt+1)].append(K_total_preds[tt].item())

            metrics["diversity"].append(cal_diversity(torch.stack(K_pred_traj_final, 0)))
            metrics["diversity_N"].append(batch.has_preds.size(1))
        return metrics


def initialize_metrics():
    return {
            "sum_minADE@1s":[],
            "sum_minFDE@1s":[],
            "sum_minMR@1s":[],
            "count_final@1s":[],
            "count_average@1s":[],
            "sum_minADE@2s":[],
            "sum_minFDE@2s":[],
            "sum_minMR@2s":[],
            "count_final@2s":[],
            "count_average@2s":[],
            "sum_minADE@3s":[],
            "sum_minFDE@3s":[],
            "sum_minMR@3s":[],
            "count_final@3s":[],
            "count_average@3s":[],
            "diversity":[],
            "diversity_N":[],
            "raw_ades":[],
            "raw_curvatures":[],
            "raw_laterals":[],
            "raw_reg_frenets":[],
        }

def laneGCN_graph_gather(graphs):
    batch_size = len(graphs)
    node_idcs = []
    count = 0
    counts = []
    for i in range(batch_size):
        counts.append(count)
        idcs = torch.arange(count, count + graphs[i]["num_nodes"]).to(
            graphs[i]["feats"].device
        )
        node_idcs.append(idcs)
        count = count + graphs[i]["num_nodes"]

    graph = dict()
    graph["idcs"] = node_idcs
    graph["ctrs"] = [x["ctrs"] for x in graphs]
    graph["obs_idcs"] = [x["start"] for x in graphs]
    graph["path"] = []
    for k in range(len(graphs[0]["path"])):
        graph["path"].append([x["path"][k] for x in graphs])

    for key in ["feats", "pris"]:
        graph[key] = torch.cat([x[key] for x in graphs], 0)

    for k1 in ["pre", "suc"]:
        graph[k1] = []
        for i in range(len(graphs[0]["pre"])):
            graph[k1].append(dict())
            for ki, k2 in enumerate(["u", "v"]):
                graph[k1][i][k2] = torch.cat(
                    [graphs[j][k1][i][ki] + counts[j] for j in range(batch_size)], 0
                )

    for k1 in ["left", "right"]:
        graph[k1] = dict()
        for ki, k2 in enumerate(["u", "v"]):
            temp = [graphs[i][k1][ki] + counts[i] for i in range(batch_size)]
            temp = [
                x if x.dim() > 0 else graph["pre"][0]["u"].new().resize_(0)
                for x in temp
            ]
            graph[k1][k2] = torch.cat(temp)
    return graph