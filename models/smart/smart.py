import contextlib
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from models.smart.metrics import minADE
from models.smart.metrics import minFDE
from models.smart.metrics import TokenCls
# from smart.metrics.waymo_metrics import cal_waymo_metrics
from .modules.smart_decoder import SMARTDecoder
from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np
import pickle
from collections import defaultdict
import os
# from waymo_open_dataset.protos import sim_agents_submission_pb2
# import dgl
# import networkx as nx
import os
import yaml
import easydict
from models.smart.metrics.utils import cal_ade, cal_fde, cal_mr, cal_diversity

# def joint_scene_from_states(states, object_ids) -> sim_agents_submission_pb2.JointScene:
#     states = states.numpy()
#     simulated_trajectories = []
#     for i_object in range(len(object_ids)):
#         simulated_trajectories.append(sim_agents_submission_pb2.SimulatedTrajectory(
#             center_x=states[i_object, :, 0], center_y=states[i_object, :, 1],
#             center_z=states[i_object, :, 2], heading=states[i_object, :, 3],
#             object_id=object_ids[i_object].item()
#         ))
#     return sim_agents_submission_pb2.JointScene(simulated_trajectories=simulated_trajectories)


def load_config_act(path):
    """ load config file"""
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return easydict.EasyDict(cfg)

class SMART(pl.LightningModule):

    def __init__(self, model_config) -> None:
        super(SMART, self).__init__()
        self.save_hyperparameters()
        model_config = load_config_act("models/smart/configs/validation.yaml").Model
        self.model_config = model_config
        self.warmup_steps = model_config.warmup_steps
        self.lr = model_config.lr
        self.total_steps = model_config.total_steps
        self.dataset = model_config.dataset
        self.input_dim = model_config.input_dim
        self.hidden_dim = model_config.hidden_dim
        self.output_dim = model_config.output_dim
        self.output_head = model_config.output_head
        self.num_historical_steps = model_config.num_historical_steps
        self.current_step = self.num_historical_steps - 1
        self.num_future_steps = model_config.decoder.num_future_steps
        self.num_freq_bands = model_config.num_freq_bands
        self.vis_map = False
        self.noise = True
        module_dir = os.path.dirname(os.path.dirname(__file__))
        self.map_token_traj_path = os.path.join(module_dir, 'smart/tokens/map_traj_token5.pkl')
        self.init_map_token()
        self.token_path = os.path.join(module_dir, 'smart/tokens/cluster_frame_5_2048.pkl')
        token_data = self.get_trajectory_token()
        self.encoder = SMARTDecoder(
            dataset=model_config.dataset,
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            num_historical_steps=model_config.num_historical_steps,
            num_freq_bands=model_config.num_freq_bands,
            num_heads=model_config.num_heads,
            head_dim=model_config.head_dim,
            dropout=model_config.dropout,
            num_map_layers=model_config.decoder.num_map_layers,
            num_agent_layers=model_config.decoder.num_agent_layers,
            pl2pl_radius=model_config.decoder.pl2pl_radius,
            pl2a_radius=model_config.decoder.pl2a_radius,
            a2a_radius=model_config.decoder.a2a_radius,
            time_span=model_config.decoder.time_span,
            map_token={'traj_src': self.map_token['traj_src']},
            token_data=token_data,
            token_size=model_config.decoder.token_size,
        )
        self.minADE0 = minADE(max_guesses=1)
        self.minFDE0 = minFDE(max_guesses=1)
        self.minADE1 = minADE(max_guesses=1)
        self.minFDE1 = minFDE(max_guesses=1)
        self.TokenCls = TokenCls(max_guesses=1)

        self.test_predictions = dict()
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.map_cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.inference_token = True
        self.test_waymo = False
        self.rollout_num = 1
        self.shift = 5
        self.num_modes = 6
        self.num_preds = 15

    def get_trajectory_token(self):
        token_data = pickle.load(open(self.token_path, 'rb'))
        self.trajectory_token = token_data['token']
        self.trajectory_token_traj = token_data['traj']
        self.trajectory_token_all = token_data['token_all']
        return token_data

    def init_map_token(self):
        self.argmin_sample_len = 3
        map_token_traj = pickle.load(open(self.map_token_traj_path, 'rb'))
        self.map_token = {'traj_src': np.array(map_token_traj['traj_src']), }
        traj_end_theta = np.arctan2(self.map_token['traj_src'][:, -1, 1]-self.map_token['traj_src'][:, -2, 1],
                                    self.map_token['traj_src'][:, -1, 0]-self.map_token['traj_src'][:, -2, 0])
        indices = torch.linspace(0, self.map_token['traj_src'].shape[1]-1, steps=self.argmin_sample_len).long().numpy()
        self.map_token['sample_pt'] = torch.tensor(self.map_token['traj_src'][:, indices]).to(torch.float)
        self.map_token['traj_end_theta'] = torch.tensor(traj_end_theta).to(torch.float)
        self.map_token['traj_src'] = torch.tensor(self.map_token['traj_src']).to(torch.float)

    def forward(self, data: HeteroData):
        res = self.encoder(data)
        return res

    def inference(self, data: HeteroData):
        res = self.encoder.inference(data)
        return res

    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
        
    def predict_step(self, data, batch_idx, dataloader_idx=0):
        data = self.match_token_map(data)
        data = self.sample_pt_pred(data)
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        pred = self.inference(data)
        return data, pred 

    def training_step(self,
                      data,
                      batch_idx):
        data = self.match_token_map(data)
        data = self.sample_pt_pred(data)
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        pred = self(data)
        next_token_prob = pred['next_token_prob']
        next_token_idx_gt = pred['next_token_idx_gt']
        next_token_eval_mask = pred['next_token_eval_mask']
        cls_loss = self.cls_loss(next_token_prob[next_token_eval_mask], next_token_idx_gt[next_token_eval_mask])
        loss = cls_loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('cls_loss', cls_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self,
                        data,
                        batch_idx, dataloader_idx=0):
        data = self.match_token_map(data)
        data = self.sample_pt_pred(data)
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        pred = self(data)
        next_token_idx = pred['next_token_idx']
        next_token_idx_gt = pred['next_token_idx_gt']
        next_token_eval_mask = pred['next_token_eval_mask']
        next_token_prob = pred['next_token_prob']
        cls_loss = self.cls_loss(next_token_prob[next_token_eval_mask], next_token_idx_gt[next_token_eval_mask])
        loss = cls_loss
        self.TokenCls.update(pred=next_token_idx[next_token_eval_mask], target=next_token_idx_gt[next_token_eval_mask],
                        valid_mask=next_token_eval_mask[next_token_eval_mask])
        self.log('val_cls_acc', self.TokenCls, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps-1]  # * (data['agent']['category'] == 3)
        if self.inference_token:
            preds = []
            for k in range(self.num_modes):
                pred = self.inference(data)
                preds.append(pred)
            
            metrics = self.compute_metrics(preds, data, dataloader_idx)
            val_label = "val_seen" if dataloader_idx == 0 else "val_unseen"
            self.log(f'val_minADE_{val_label}', metrics['sum_minADE@3s'][-1]/metrics['count_average@3s'][-1], prog_bar=True, on_step=False, on_epoch=True, batch_size=metrics['count_average@3s'][-1])
            self.log(f'val_minFDE_{val_label}', metrics['sum_minFDE@3s'][-1]/metrics['count_final@3s'][-1], prog_bar=True, on_step=False, on_epoch=True, batch_size=metrics['count_final@3s'][-1])
            # pos_a = pred['pos_a']
            # gt = pred['gt']
            # valid_mask = data['agent']['valid_mask'][:, self.num_historical_steps:]
            # pred_traj = pred['pred_traj']
            # eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps-1]
            # if dataloader_idx == 0:
            #     self.minADE0.update(pred=pred_traj[eval_mask], target=gt[eval_mask], valid_mask=valid_mask[eval_mask])
            #     self.minFDE0.update(pred=pred_traj[eval_mask], target=gt[eval_mask], valid_mask=valid_mask[eval_mask])
            #     # print('ade: ', self.minADE.compute(), 'fde: ', self.minFDE.compute())
            #     self.log('val_minADE_val_seen', self.minADE0, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
            #     self.log('val_minFDE_val_seen', self.minFDE0, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
            # else:
            #     self.minADE1.update(pred=pred_traj[eval_mask], target=gt[eval_mask], valid_mask=valid_mask[eval_mask])
            #     self.minFDE1.update(pred=pred_traj[eval_mask], target=gt[eval_mask], valid_mask=valid_mask[eval_mask])
            #     # print('ade: ', self.minADE.compute(), 'fde: ', self.minFDE.compute())
            #     self.log('val_minADE_val_unseen', self.minADE1, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
            #     self.log('val_minFDE_val_unseen', self.minFDE1, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
            
           

    def compute_metrics(self, predictions, batch, dataloader_idx):
        metrics = self.val_seen_metrics if dataloader_idx == 0 else self.val_unseen_metrics    
            
        # K = 6
        ades = [[], [], []]
        fdes = [[], [], []]
        mrs = [[], [], []]
        K_total_preds = [0, 0, 0]
        K_total_final_preds = [0, 0, 0]
        K_pred_traj_final = []
        
        pred_traj_gt = predictions[0]['gt'][:, 1::2].permute(1, 0, 2)
        has_preds = batch['agent']['valid_mask'][:, self.num_historical_steps:][:, 1::2].permute(1, 0)
        for k in range(self.num_modes):
            pred_traj = predictions[k]['pred_traj'][:, 1::2].permute(1, 0, 2)
            K_pred_traj_final.append(pred_traj[-1])
            for tt in range(3):
                ade = cal_ade(
                    pred_traj_gt[:self.num_preds//3*(tt+1)-1],
                    pred_traj[:self.num_preds//3*(tt+1)-1],
                    has_preds[:self.num_preds//3*(tt+1)-1], mode='raw'
                )
                fde = cal_fde(
                    pred_traj_gt[:self.num_preds//3*(tt+1)-1],
                    pred_traj[:self.num_preds//3*(tt+1)-1],
                    has_preds[:self.num_preds//3*(tt+1)-1], mode='raw'
                )
                mr = cal_mr(
                    pred_traj_gt[:self.num_preds//3*(tt+1)-1],
                    pred_traj[:self.num_preds//3*(tt+1)-1],
                    has_preds[:self.num_preds//3*(tt+1)-1], mode='raw'
                )
                ades[tt].append(ade)
                fdes[tt].append(fde)
                mrs[tt].append(mr)
                K_total_preds[tt] = has_preds[:self.num_preds//3*(tt+1)-1].sum()
                K_total_final_preds[tt] = has_preds[self.num_preds//3*(tt+1)-1].sum()
        for tt in range(3):
            metrics["sum_minADE@{}s".format(tt+1)].append(torch.stack(ades[tt], 1).min(1)[0].sum().item())
            metrics["sum_minFDE@{}s".format(tt+1)].append(torch.stack(fdes[tt], 1).min(1)[0].sum().item())
            metrics["sum_minMR@{}s".format(tt+1)].append(torch.stack(mrs[tt], 1).min(1)[0].sum().item())
            metrics["count_final@{}s".format(tt+1)].append(K_total_final_preds[tt].item())
            metrics["count_average@{}s".format(tt+1)].append(K_total_preds[tt].item())

        metrics["diversity"].append(cal_diversity(torch.stack(K_pred_traj_final, 0)))
        metrics["diversity_N"].append(has_preds.size(1))
        return metrics
    
    def on_validation_start(self):
        self.val_seen_metrics = initialize_metrics()
        self.val_unseen_metrics = initialize_metrics()

    def on_validation_end(self):
        print("Validation metrics for seen dataset:")
        for tt in range(3):
            ade = np.sum(self.val_seen_metrics["sum_minADE@{}s".format(tt+1)]) / np.sum(self.val_seen_metrics["count_average@{}s".format(tt+1)])
            fde = np.sum(self.val_seen_metrics["sum_minFDE@{}s".format(tt+1)]) / np.sum(self.val_seen_metrics["count_final@{}s".format(tt+1)])
            mr = np.sum(self.val_seen_metrics["sum_minMR@{}s".format(tt+1)]) / np.sum(self.val_seen_metrics["count_final@{}s".format(tt+1)])
            
            print(f"  minADE@{tt+1}s: {ade:.4f}, minFDE@{tt+1}s: {fde:.4f}, minMR@{tt+1}s: {mr:.4f}")
        diversity = np.sum(self.val_seen_metrics["diversity"]) / np.sum(self.val_seen_metrics["diversity_N"])
        print(f"  diversity: {diversity:.4f}")
        print("Validation metrics for unseen dataset:")
        for tt in range(3):
            ade = np.sum(self.val_unseen_metrics["sum_minADE@{}s".format(tt+1)]) / np.sum(self.val_unseen_metrics["count_average@{}s".format(tt+1)])
            fde = np.sum(self.val_unseen_metrics["sum_minFDE@{}s".format(tt+1)]) / np.sum(self.val_unseen_metrics["count_final@{}s".format(tt+1)])
            mr = np.sum(self.val_unseen_metrics["sum_minMR@{}s".format(tt+1)]) / np.sum(self.val_unseen_metrics["count_final@{}s".format(tt+1)])
            
            print(f"  minADE@{tt+1}s: {ade:.4f}, minFDE@{tt+1}s: {fde:.4f}, minMR@{tt+1}s: {mr:.4f}")
        diversity = np.sum(self.val_unseen_metrics["diversity"]) / np.sum(self.val_unseen_metrics["diversity_N"])
        print(f"  diversity: {diversity:.4f}")
        print("Validation finished.")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(current_step):
            if current_step + 1 < self.warmup_steps:
                return float(current_step + 1) / float(max(1, self.warmup_steps))
            return max(
                0.0, 0.5 * (1.0 + math.cos(math.pi * (current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))))
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def load_params_from_file(self, filename, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        # logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['state_dict']

        version = checkpoint.get("version", None)
        # if version is not None:
        #     logger.info('==> Checkpoint trained from version: %s' % version)

        # logger.info(f'The number of disk ckpt keys: {len(model_state_disk)}')
        model_state = self.state_dict()
        model_state_disk_filter = {}
        for key, val in model_state_disk.items():
            if key in model_state and model_state_disk[key].shape == model_state[key].shape:
                model_state_disk_filter[key] = val
            else:
                if key not in model_state:
                    print(f'Ignore key in disk (not found in model): {key}, shape={val.shape}')
                else:
                    print(f'Ignore key in disk (shape does not match): {key}, load_shape={val.shape}, model_shape={model_state[key].shape}')

        model_state_disk = model_state_disk_filter

        missing_keys, unexpected_keys = self.load_state_dict(model_state_disk, strict=False)

        # logger.info(f'Missing keys: {missing_keys}')
        # logger.info(f'The number of missing keys: {len(missing_keys)}')
        # logger.info(f'The number of unexpected keys: {len(unexpected_keys)}')
        # logger.info('==> Done (total keys %d)' % (len(model_state)))

        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        return it, epoch
    
    def match_token_map(self, data):
        traj_pos = data['map_save']['traj_pos'].to(torch.float)
        traj_theta = data['map_save']['traj_theta'].to(torch.float)
        pl_idx_list = data['map_save']['pl_idx_list']
        token_sample_pt = self.map_token['sample_pt'].to(traj_pos.device)
        token_src = self.map_token['traj_src'].to(traj_pos.device)
        max_traj_len = self.map_token['traj_src'].shape[1]
        pl_num = traj_pos.shape[0]

        pt_token_pos = traj_pos[:, 0, :].clone()
        pt_token_orientation = traj_theta.clone()
        cos, sin = traj_theta.cos(), traj_theta.sin()
        rot_mat = traj_theta.new_zeros(pl_num, 2, 2)
        rot_mat[..., 0, 0] = cos
        rot_mat[..., 0, 1] = -sin
        rot_mat[..., 1, 0] = sin
        rot_mat[..., 1, 1] = cos
        traj_pos_local = torch.bmm((traj_pos - traj_pos[:, 0:1]), rot_mat.view(-1, 2, 2))
        distance = torch.sum((token_sample_pt[None] - traj_pos_local.unsqueeze(1))**2, dim=(-2, -1))
        pt_token_id = torch.argmin(distance, dim=1)

        if self.noise:
            topk_indices = torch.argsort(torch.sum((token_sample_pt[None] - traj_pos_local.unsqueeze(1))**2, dim=(-2, -1)), dim=1)[:, :8]
            sample_topk = torch.randint(0, topk_indices.shape[-1], size=(topk_indices.shape[0], 1), device=topk_indices.device)
            pt_token_id = torch.gather(topk_indices, 1, sample_topk).squeeze(-1)

        cos, sin = traj_theta.cos(), traj_theta.sin()
        rot_mat = traj_theta.new_zeros(pl_num, 2, 2)
        rot_mat[..., 0, 0] = cos
        rot_mat[..., 0, 1] = sin
        rot_mat[..., 1, 0] = -sin
        rot_mat[..., 1, 1] = cos
        token_src_world = torch.bmm(token_src[None, ...].repeat(pl_num, 1, 1, 1).reshape(pl_num, -1, 2),
                                    rot_mat.view(-1, 2, 2)).reshape(pl_num, token_src.shape[0], max_traj_len, 2) + traj_pos[:, None, [0], :]
        token_src_world_select = token_src_world.view(-1, 1024, 11, 2)[torch.arange(pt_token_id.view(-1).shape[0]), pt_token_id.view(-1)].view(pl_num, max_traj_len, 2)

        pl_idx_full = pl_idx_list.clone()
        token2pl = torch.stack([torch.arange(len(pl_idx_list), device=traj_pos.device), pl_idx_full.long()])
        count_nums = []
        for pl in pl_idx_full.unique():
            pt = token2pl[0, token2pl[1, :] == pl]
            left_side = (data['pt_token']['side'][pt] == 0).sum()
            right_side = (data['pt_token']['side'][pt] == 1).sum()
            center_side = (data['pt_token']['side'][pt] == 2).sum()
            count_nums.append(torch.Tensor([left_side, right_side, center_side]))
        count_nums = torch.stack(count_nums, dim=0)
        num_polyline = int(count_nums.max().item())
        traj_mask = torch.zeros((int(len(pl_idx_full.unique())), 3, num_polyline), dtype=bool)
        idx_matrix = torch.arange(traj_mask.size(2)).unsqueeze(0).unsqueeze(0)
        idx_matrix = idx_matrix.expand(traj_mask.size(0), traj_mask.size(1), -1)  #
        counts_num_expanded = count_nums.unsqueeze(-1)
        mask_update = idx_matrix < counts_num_expanded
        traj_mask[mask_update] = True

        data['pt_token']['traj_mask'] = traj_mask
        data['pt_token']['position'] = torch.cat([pt_token_pos, torch.zeros((data['pt_token']['num_nodes'], 1),
                                                                            device=traj_pos.device, dtype=torch.float)], dim=-1)
        data['pt_token']['orientation'] = pt_token_orientation
        data['pt_token']['height'] = data['pt_token']['position'][:, -1]
        data[('pt_token', 'to', 'map_polygon')] = {}
        data[('pt_token', 'to', 'map_polygon')]['edge_index'] = token2pl
        data['pt_token']['token_idx'] = pt_token_id
        return data

    def sample_pt_pred(self, data):
        traj_mask = data['pt_token']['traj_mask'] # 194,3,76
        raw_pt_index = torch.arange(1, traj_mask.shape[2]).repeat(traj_mask.shape[0], traj_mask.shape[1], 1)
        masked_pt_index = raw_pt_index.view(-1)[torch.randperm(raw_pt_index.numel())[:traj_mask.shape[0]*traj_mask.shape[1]*((traj_mask.shape[2]-1)//3)].reshape(traj_mask.shape[0], traj_mask.shape[1], (traj_mask.shape[2]-1)//3)]
        masked_pt_index = torch.sort(masked_pt_index, -1)[0]
        pt_valid_mask = traj_mask.clone()
        pt_valid_mask.scatter_(2, masked_pt_index, False)
        pt_pred_mask = traj_mask.clone()
        pt_pred_mask.scatter_(2, masked_pt_index, False)
        tmp_mask = pt_pred_mask.clone()
        tmp_mask[:, :, :] = True
        tmp_mask.scatter_(2, masked_pt_index-1, False)
        pt_pred_mask.masked_fill_(tmp_mask, False)
        pt_pred_mask = pt_pred_mask * torch.roll(traj_mask, shifts=-1, dims=2)
        pt_target_mask = torch.roll(pt_pred_mask, shifts=1, dims=2)

        data['pt_token']['pt_valid_mask'] = pt_valid_mask[traj_mask]
        data['pt_token']['pt_pred_mask'] = pt_pred_mask[traj_mask]
        data['pt_token']['pt_target_mask'] = pt_target_mask[traj_mask]

        return data

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