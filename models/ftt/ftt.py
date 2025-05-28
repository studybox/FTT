import pytorch_lightning as pl
import torch
from .modules import FrenetPathMultiTargetGCN
from models.ftt.utils.util import relative_to_curve, relative_to_abs, cal_ade, cal_fde, cal_mr, cal_diversity

class FTT(pl.LightningModule):
    def __init__(self, config):
        super(FTT, self).__init__()
        self.model = FrenetPathMultiTargetGCN(config)
        self.config = config
        self.learning_rate = config["args"].learning_rate
        self.l2_weight_decay = config["args"].l2_weight_decay
        self.learning_rate_decay = config["args"].learning_rate_decay

        self.val_seen_metrics = {}
        self.val_unseen_metrics = {}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        ret = self.model(batch)
        loss = ret['loss']
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss
    
    def on_validation_start(self):
        self.val_seen_metrics = initialize_metrics()
        self.val_unseen_metrics = initialize_metrics()
        

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ret = self.model(batch)
        loss = ret['loss']
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        metrics = self.compute_metrics(ret, batch, dataloader_idx)
        val_label = "val_seen" if dataloader_idx == 0 else "val_unseen"
        self.log(f'val_minADE_{val_label}', metrics['sum_minADE@3s'][-1]/metrics['count_average@3s'][-1], prog_bar=True, on_step=False, on_epoch=True, batch_size=metrics['count_average@3s'][-1])
        self.log(f'val_minFDE_{val_label}', metrics['sum_minFDE@3s'][-1]/metrics['count_final@3s'][-1], prog_bar=True, on_step=False, on_epoch=True, batch_size=metrics['count_final@3s'][-1])

    # def on_validation_end(self):
    #     print("Validation metrics for seen dataset:")
    #     for tt in range(3):
    #         ade = np.sum(self.val_seen_metrics["sum_minADE@{}s".format(tt+1)]) / np.sum(self.val_seen_metrics["count_average@{}s".format(tt+1)])
    #         fde = np.sum(self.val_seen_metrics["sum_minFDE@{}s".format(tt+1)]) / np.sum(self.val_seen_metrics["count_final@{}s".format(tt+1)])
    #         mr = np.sum(self.val_seen_metrics["sum_minMR@{}s".format(tt+1)]) / np.sum(self.val_seen_metrics["count_final@{}s".format(tt+1)])
            
    #         print(f"  minADE@{tt+1}s: {ade:.4f}, minFDE@{tt+1}s: {fde:.4f}, minMR@{tt+1}s: {mr:.4f}")
    #     diversity = np.sum(self.val_seen_metrics["diversity"]) / np.sum(self.val_seen_metrics["diversity_N"])
    #     print(f"  diversity: {diversity:.4f}")
    #     print("Validation metrics for unseen dataset:")
    #     for tt in range(3):
    #         ade = np.sum(self.val_unseen_metrics["sum_minADE@{}s".format(tt+1)]) / np.sum(self.val_unseen_metrics["count_average@{}s".format(tt+1)])
    #         fde = np.sum(self.val_unseen_metrics["sum_minFDE@{}s".format(tt+1)]) / np.sum(self.val_unseen_metrics["count_final@{}s".format(tt+1)])
    #         mr = np.sum(self.val_unseen_metrics["sum_minMR@{}s".format(tt+1)]) / np.sum(self.val_unseen_metrics["count_final@{}s".format(tt+1)])
            
    #         print(f"  minADE@{tt+1}s: {ade:.4f}, minFDE@{tt+1}s: {fde:.4f}, minMR@{tt+1}s: {mr:.4f}")
    #     diversity = np.sum(self.val_unseen_metrics["diversity"]) / np.sum(self.val_unseen_metrics["diversity_N"])
    #     print(f"  diversity: {diversity:.4f}")
    #     print("Validation finished.")
        # save the "raw_ades" and "raw_curvatures" to a file
        # np.save("raw_ades_seen.npy", np.concatenate(self.val_seen_metrics["raw_ades"],0), allow_pickle=True)
        # np.save("raw_curvatures_seen.npy", np.concatenate(self.val_seen_metrics["raw_curvatures"],0), allow_pickle=True)
        # np.save("raw_ades_unseen.npy", np.concatenate(self.val_unseen_metrics["raw_ades"],0), allow_pickle=True)
        # np.save("raw_curvatures_unseen.npy", np.concatenate(self.val_unseen_metrics["raw_curvatures"],0), allow_pickle=True)
        # np.save("raw_laterals_seen.npy", np.concatenate(self.val_seen_metrics["raw_laterals"],0), allow_pickle=True)
        # np.save("raw_laterals_unseen.npy", np.concatenate(self.val_unseen_metrics["raw_laterals"],0), allow_pickle=True)
        # np.save("raw_reg_frenets_seen.npy", np.concatenate(self.val_seen_metrics["raw_reg_frenets"],0), allow_pickle=True)
        # np.save("raw_reg_frenets_unseen.npy", np.concatenate(self.val_unseen_metrics["raw_reg_frenets"],0), allow_pickle=True)
        
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
            if "curves_gt" in predictions:
                curves_gt = predictions["curves_gt"]
                pred_traj_gt_rel = relative_to_curve(batch.fut_traj_fre, batch.obs_traj[-1], curves_gt)
                pred_traj_gt = relative_to_abs(pred_traj_gt_rel, batch.obs_traj[-1])
            else:
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
            metrics['raw_reg_frenets'].append(predictions['reg_frenets'].cpu().numpy())
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