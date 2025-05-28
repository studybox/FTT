from models.tnt.layers import Vectornet, TargetPred, MotionEstimation, TrajScoreSelection, TNTLoss, distance_metric
import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.ftt.utils.util import relative_to_curve, relative_to_abs, cal_ade, cal_fde, cal_mr, cal_diversity

class TNT(pl.LightningModule):
    def __init__(self, config):
        # adopted from https://github.com/Henry1iu/TNT-Trajectory-Predition/blob/main/core/model/TNT.py
        super(TNT, self).__init__()
        self.learning_rate = config["args"].learning_rate
        self.l2_weight_decay = config["args"].l2_weight_decay
        self.learning_rate_decay = config["args"].learning_rate_decay
        self.config = config

        self.horizon = config["num_preds"]
        self.m = 7
        self.k = config["num_mods"]

        self.lambda1 = 0.1
        self.lambda2 = 1.0
        self.lambda3 = 0.1
        
        in_channels = (3,8)
        num_subgraph_layers = 3
        subgraph_width = 64
        num_global_graph_layer = 1
        global_graph_width = 64
        target_pred_hid = 64
        motion_esti_hid = 64
        score_sel_hid = 64
        temperature = 0.01

        self.vectornet = Vectornet(
            in_channels=in_channels,
            num_subgraph_layres=num_subgraph_layers,
            subgraph_width=subgraph_width,
            num_global_graph_layer=num_global_graph_layer,
            global_graph_width=global_graph_width,

        )
        self.target_pred_layer = TargetPred(
            in_channels=global_graph_width,
            hidden_dim=target_pred_hid,
            m=self.m,
            device=self.device
        )
        self.motion_estimator = MotionEstimation(
            in_channels=global_graph_width,
            horizon=self.horizon,
            hidden_dim=motion_esti_hid
        )
        self.traj_score_layer = TrajScoreSelection(
            feat_channels=global_graph_width,
            horizon=self.horizon,
            hidden_dim=score_sel_hid,
            temper=temperature,
            device=self.device
        )
        self._init_weight()
        self.loss = TNTLoss(
            self.lambda1, self.lambda2, self.lambda3,self.horizon,
            self.m, self.k, temperature,
             device=self.device
        )

    def traj_selection(self, traj_in, traj_in_rel, score, threshold=0.01):
        """
        select the top k trajectories according to the score and the distance
        :param traj_in: candidate trajectories, [batch, M, horizon * 2]
        :param score: score of the candidate trajectories, [batch, M]
        :param threshold: float, the threshold for exclude traj prediction
        :return: [batch_size, k, horizon * 2]
        """
        # re-arrange trajectories according the the descending order of the score

        _, batch_order = score.sort(descending=True)
        batch_sz = score.size(0)
        traj_pred = torch.cat([traj_in[i, order] for i, order in enumerate(batch_order)], dim=0).view(-1, self.m, self.horizon * 2)
        traj_pred_rel = torch.cat([traj_in_rel[i, order] for i, order in enumerate(batch_order)], dim=0).view(-1, self.m, self.horizon * 2)

        traj_selected = traj_pred[:, :self.k].clone()
        traj_selected_rel = traj_pred_rel[:, :self.k].clone()   # [batch_size, k, horizon * 2]

        # check the distance between them, NMS, stop only when enough trajs collected
        for batch_id in range(traj_pred.shape[0]):                              # one batch for a time
            traj_cnt = 1
            thres = threshold
            while traj_cnt < self.k:
                for j in range(1, self.m):
                    dis = distance_metric(traj_selected[batch_id, :traj_cnt], traj_pred[batch_id, j].unsqueeze(0))
                    if not torch.any(dis < thres):
                        traj_selected[batch_id, traj_cnt] = traj_pred[batch_id, j].clone()
                        traj_selected_rel[batch_id, traj_cnt] = traj_pred_rel[batch_id, j].clone()

                        traj_cnt += 1
                    if traj_cnt >= self.k:
                        break
                thres /= 2.0

        return traj_selected_rel.permute(1,0,2).reshape(self.k, batch_sz,-1 ,2)

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


    def forward(self, batch):
        """
        output prediction for training
        :param data: observed sequence data
        :return: dict{
                        "target_prob":  the predicted probability of each target candidate,
                        "offset":       the predicted offset of the target position from the gt target candidate,
                        "traj_with_gt": the predicted trajectory with the gt target position as the input,
                        "traj":         the predicted trajectory without the gt target position,
                        "score":        the predicted score for each predicted trajectory,
                     }
        """

        target_candidate = batch.candidate   # [batch_size, N, 2]
        candidate_mask = batch.candidate_mask

        # feature encoding
        target_feat = self.vectornet(batch)

        # predict prob. for each target candidate, and corresponding offest
        target_logprob, offset = self.target_pred_layer(target_feat, target_candidate, candidate_mask)
        #print(target_prob[:2])
        # predict the trajectory given the target gt
        target_gt = batch.target_gt.view(-1, 1, 2)
        traj_rel_with_gt = self.motion_estimator(target_feat, target_gt)
        # predict the trajectories for the M most-likely predicted target, and the score
        _, indices = target_logprob.topk(self.m, dim=1)
        num_vehs = batch.target_gt.size(0)
        batch_idx = torch.vstack([torch.arange(0, num_vehs, device=self.device) for _ in range(self.m)]).T
        target_pred_se, offset_pred_se = target_candidate[batch_idx, indices], offset[batch_idx, indices]
        trajs_rel = self.motion_estimator(target_feat, target_pred_se + offset_pred_se)

        last_pos = batch.actor_ctrs.view(-1,1,1,2)
        #print(trajs_rel.size(), last_pos.size())

        trajs = torch.cumsum(trajs_rel.view(-1,self.m,self.horizon,2), dim=2) + last_pos
        trajs = trajs.view(-1, self.m, self.horizon*2)

        score = self.traj_score_layer(target_feat, trajs)

        reg = self.traj_selection(trajs, trajs_rel, score)
        ret = {
            "target_logprob": target_logprob,
            "offset": offset,
            "traj_with_gt": traj_rel_with_gt.view(-1, self.horizon, 2),
            "traj": trajs,
            "score": score,
            "reg":reg,
            "pred":traj_rel_with_gt.reshape(-1, self.horizon, 2).permute(1,0,2)#reg[0].permute(1,0,2)
            }

        gt = {
            "candidate_gt": batch.candidate_gt,
            "offset": batch.offset_gt.view(-1, 2),
            "y_rel": batch.fut_traj_rel.permute(1,0,2),
            "y":batch.fut_traj.permute(1,0,2).reshape(-1, self.horizon*2),
            "has_preds":batch.has_preds.permute(1,0)
        }
        loss, loss_dict = self.loss(ret, gt)
        ret["loss"] = loss
        for key in loss_dict:
            ret[key] = loss_dict[key]
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
        if "frenet_pred" in predictions:
            frenet_pred = predictions["frenet_pred"]
            curves = predictions["curves"]
            curves_gt = predictions["curves_gt"]
            if "converted_pred" in predictions:
                pred_traj_fake_rel = predictions["converted_pred"]
            else:
                pred_traj_fake_rel = relative_to_curve(frenet_pred, batch.obs_traj[-1], curves)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, batch.obs_traj[-1])
            if "converted_gt" in predictions:
                pred_traj_gt_rel = predictions["converted_gt"]
            else:
                pred_traj_gt_rel = relative_to_curve(batch.fut_traj_fre, batch.obs_traj[-1], curves_gt)
            pred_traj_gt = relative_to_abs(pred_traj_gt_rel, batch.obs_traj[-1])

            ade = cal_ade(
                    pred_traj_gt,
                    pred_traj_fake,
                    batch.has_preds,
                    mode='raw'
            )
            curvatures = predictions["converted_curvature"]
            metrics["raw_curvatures"].append(curvatures[batch.has_preds].cpu().numpy())
            metrics["raw_ades"].append(ade.cpu().numpy())
            metrics["raw_laterals"].append(frenet_pred[:, :, 1][batch.has_preds].cpu().numpy())

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