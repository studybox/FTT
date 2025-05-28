import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from models.fjmp.modules.fjmp_modules import *
from models.fjmp.utils.fjmp_utils import *
from models.fjmp.utils.dag_utils import * 
from models.fjmp.configs.preset_config import preset_config
from models.smart.metrics.utils import cal_ade, cal_fde, cal_mr, cal_diversity

class FJMP(pl.LightningModule):
    def __init__(self, config):
        super(FJMP, self).__init__()
        self.config = preset_config
        # self.dataset = config["dataset"]
        # self.num_train_samples = preset_config["num_train_samples"]
        # self.num_val_samples = preset_config["num_val_samples"]
        self.num_agenttypes = preset_config["num_agenttypes"]
        self.switch_lr_1 = preset_config["switch_lr_1"]
        self.switch_lr_2 = preset_config["switch_lr_2"]
        self.lr_step = preset_config["lr_step"]
        # self.mode = preset_config["mode"]
        self.input_size = preset_config["input_size"]
        self.observation_steps = preset_config["observation_steps"]
        self.prediction_steps = preset_config["prediction_steps"]
        self.num_edge_types = preset_config["num_edge_types"]
        self.h_dim = preset_config["h_dim"]
        self.num_joint_modes = preset_config["num_joint_modes"]
        self.num_proposals = preset_config["num_proposals"]
        self.learning_rate = preset_config["lr"]
        self.max_epochs = preset_config["max_epochs"]
        # self.log_path = preset_config["log_path"]
        self.batch_size = preset_config["batch_size"]
        self.decoder = preset_config["decoder"]
        self.num_heads = preset_config["num_heads"]
        self.learned_relation_header = preset_config["learned_relation_header"]
        self.resume_training = preset_config["resume_training"]
        self.proposal_coef = preset_config["proposal_coef"]
        self.rel_coef = preset_config["rel_coef"]
        self.proposal_header = preset_config["proposal_header"]
        self.two_stage_training = preset_config["two_stage_training"]
        self.training_stage = preset_config["training_stage"]
        self.ig = preset_config["ig"]
        self.focal_loss = preset_config["focal_loss"]
        self.gamma = preset_config["gamma"]
        self.weight_0 = preset_config["weight_0"]
        self.weight_1 = preset_config["weight_1"]
        self.weight_2 = preset_config["weight_2"]
        self.teacher_forcing = preset_config["teacher_forcing"]
        self.scheduled_sampling = preset_config["scheduled_sampling"]
        self.eval_training = preset_config["eval_training"]
        self.supervise_vehicles = preset_config["supervise_vehicles"]
        self.no_agenttype_encoder = preset_config["no_agenttype_encoder"]
        self.train_all = preset_config["train_all"]
        
        if self.two_stage_training and self.training_stage == 2:
            self.pretrained_relation_header = None
        
        self.build()

        self.val_seen_metrics = {}
        self.val_unseen_metrics = {}

    def build(self):
        self.feature_encoder = FJMPFeatureEncoder(self.config)
        if self.learned_relation_header:
            self.relation_header = FJMPRelationHeader(self.config)
        
        if self.proposal_header:
            self.proposal_decoder = FJMPTrajectoryProposalDecoder(self.config)
        
        if (self.two_stage_training and self.training_stage == 2) or not self.two_stage_training:
            if self.decoder == 'dagnn':
                self.trajectory_decoder = FJMPAttentionTrajectoryDecoder(self.config)
            elif self.decoder == 'lanegcn':
                self.trajectory_decoder = LaneGCNHeader(self.config)

    def forward(self, scene_idxs, graph, stage_1_graph, ig_dict, batch_idxs, batch_idxs_edges, actor_ctrs, ks=None, prop_ground_truth = 0., eval=True):
        
        if self.learned_relation_header:
            edge_logits = self.relation_header(graph)
            graph.edata["edge_logits"] = edge_logits
        else:
            # use ground-truth interaction graph
            if not self.two_stage_training:
                edge_probs = torch.nn.functional.one_hot(ig_dict["ig_labels"].long(), self.num_edge_types)
            elif self.two_stage_training and self.training_stage == 2:
                prh_logits = self.pretrained_relation_header.relation_header(stage_1_graph)
                graph.edata["edge_logits"] = prh_logits
        
        all_edges = [x.unsqueeze(1) for x in graph.edges('all')]
        all_edges = torch.cat(all_edges, 1)
        # remove half of the directed edges (effectively now an undirected graph)
        eids_remove = all_edges[torch.where(all_edges[:, 0] > all_edges[:, 1])[0], 2]
        graph.remove_edges(eids_remove)

        if self.learned_relation_header or (self.two_stage_training and self.training_stage == 2):
            edge_logits = graph.edata.pop("edge_logits")
            edge_probs = my_softmax(edge_logits, -1)

        graph.edata["edge_probs"] = edge_probs

        dag_graph = build_dag_graph(graph, self.config)
        
        if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
            dag_graph = prune_graph_johnson(dag_graph)
        
        if self.proposal_header:
            dag_graph, proposals = self.proposal_decoder(dag_graph, actor_ctrs)
        
        if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
            loc_pred = self.trajectory_decoder(dag_graph, prop_ground_truth, batch_idxs)
        
        # loc_pred: shape [N, prediction_steps, num_joint_modes, 2]
        res = {}

        if self.proposal_header:
            res["proposals"] = proposals # trajectory proposal future coordinates
        
        if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
            res["loc_pred"] = loc_pred # predicted future coordinates
        
        if self.learned_relation_header:
            res["edge_logits"] = edge_logits.float() # edge probabilities for computing BCE loss    
            res["edge_probs"] = edge_probs.float()     
        
        return res

    def training_step(self, batch, batch_idx):
        self.teacher_forcing = True
        self.config["teacher_forcing"]  = True
        if self.scheduled_sampling:
            prop_ground_truth = 1 - (self.current_epoch - 1) / (self.max_epochs - 1)   
        elif self.teacher_forcing:
            prop_ground_truth = 1.  
        else:
            prop_ground_truth = 0. 

        dd = self.process(batch)
        dgl_graph = self.init_dgl_graph(dd['batch_idxs'], dd['ctrs'], dd["agenttypes"], dd['world_locs'], dd['has_preds'])
        dgl_graph = self.feature_encoder(dgl_graph, dd['feats'][:,:self.observation_steps], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])
        if self.two_stage_training and self.training_stage == 2:
            stage_1_graph = self.build_stage_1_graph(dgl_graph, dd['feats'][:,:self.observation_steps], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])
        else:
            stage_1_graph = None
        ig_dict = {}
        ig_dict["ig_labels"] = dd["ig_labels"] 
        res = self.forward(dd["scene_idxs"], dgl_graph, stage_1_graph, ig_dict, dd['batch_idxs'], dd["batch_idxs_edges"], dd["actor_ctrs"], prop_ground_truth=prop_ground_truth, eval=False)

        loss_dict = self.get_loss(dgl_graph, dd['batch_idxs'], res, dd['agenttypes'], dd['has_preds'], dd['gt_locs'], dd['batch_size'], dd["ig_labels"], self.current_epoch)
        
        loss = loss_dict["total_loss"]
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('prop_loss', loss_dict['loss_prop_reg'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('reg_loss', loss_dict['loss_reg'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss
    
    def on_validation_start(self):
        self.val_seen_metrics = initialize_metrics()
        self.val_unseen_metrics = initialize_metrics()
        

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.teacher_forcing = False
        self.config["teacher_forcing"]  = False
        dd = self.process(batch)
        dgl_graph = self.init_dgl_graph(dd['batch_idxs'], dd['ctrs'], dd["agenttypes"], dd['world_locs'], dd['has_preds'])
        dgl_graph = self.feature_encoder(dgl_graph, dd['feats'][:,:self.observation_steps], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])
        if self.two_stage_training and self.training_stage == 2:
            stage_1_graph = self.build_stage_1_graph(dgl_graph, dd['feats'][:,:self.observation_steps], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])
        else:
            stage_1_graph = None
        ig_dict = {}
        ig_dict["ig_labels"] = dd["ig_labels"] 
        res = self.forward(dd["scene_idxs"], dgl_graph, stage_1_graph, ig_dict, dd['batch_idxs'], dd["batch_idxs_edges"], dd["actor_ctrs"], prop_ground_truth=1, eval=False)
        loss_dict = self.get_loss(dgl_graph, dd['batch_idxs'], res, dd['agenttypes'], dd['has_preds'], dd['gt_locs'], dd['batch_size'], dd["ig_labels"], self.current_epoch)
        loss = loss_dict["total_loss"]
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)

        metrics = self.compute_metrics(res["loc_pred"], dd, dataloader_idx)
        val_label = "val_seen" if dataloader_idx == 0 else "val_unseen"
        self.log(f'val_minADE_{val_label}', metrics['sum_minADE@3s'][-1]/metrics['count_average@3s'][-1], prog_bar=True, on_step=False, on_epoch=True, batch_size=metrics['count_average@3s'][-1])
        self.log(f'val_minFDE_{val_label}', metrics['sum_minFDE@3s'][-1]/metrics['count_final@3s'][-1], prog_bar=True, on_step=False, on_epoch=True, batch_size=metrics['count_final@3s'][-1])

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
        
    def process(self, data):
        num_actors = [len(x) for x in data['feats']]
        num_edges = [int(n * (n-1) / 2) for n in num_actors]

        # LaneGCN processing 
        # ctrs gets copied once for each agent in scene, whereas actor_ctrs only contains one per scene
        # same data, but different format so that it is compatible with LaneGCN L2A/A2A function     
        actor_ctrs = gpu(data["ctrs"])
        lane_graph = graph_gather(to_long(gpu(data["graph"])), self.config)
        # unique index assigned to each scene
        scene_idxs = torch.Tensor([idx for idx in data['idx']])

        graph = data["graph"]

        world_locs = [x for x in data['feat_locs']]
        world_locs = torch.cat(world_locs, 0)

        has_obs = [x for x in data['has_obss']]
        has_obs = torch.cat(has_obs, 0)

        ig_labels = [x for x in data['ig_labels_{}'.format(self.ig)]]
        ig_labels = torch.cat(ig_labels, 0)

        locs = [x for x in data['feats']]
        locs = torch.cat(locs, 0)

        vels = [x for x in data['feat_vels']]
        vels = torch.cat(vels, 0)

        psirads = [x for x in data['feat_psirads']]
        psirads = torch.cat(psirads, 0)

        gt_psirads = [x for x in data['gt_psirads']]
        gt_psirads = torch.cat(gt_psirads, 0)

        gt_vels = [x for x in data['gt_vels']]
        gt_vels = torch.cat(gt_vels, 0)

        agenttypes = [x for x in data['feat_agenttypes']]
        agenttypes = torch.cat(agenttypes, 0)[:, self.observation_steps - 1, 0]
        agenttypes = torch.nn.functional.one_hot(agenttypes.long(), self.num_agenttypes)

        # shape information is only available in INTERACTION dataset
        shapes = [x for x in data['feat_shapes']]
        shapes = torch.cat(shapes, 0)

        feats = torch.cat([locs, vels, psirads], dim=2)

        ctrs = [x for x in data['ctrs']]
        ctrs = torch.cat(ctrs, 0)

        # orig = [x.view(1, 2) for j, x in enumerate(data['orig']) for i in range(num_actors[j])]
        # orig = torch.cat(orig, 0)

        # rot = [x.view(1, 2, 2) for j, x in enumerate(data['rot']) for i in range(num_actors[j])]
        # rot = torch.cat(rot, 0)

        # theta = torch.Tensor([x for j, x in enumerate(data['theta']) for i in range(num_actors[j])])

        gt_locs = [x for x in data['gt_preds']]
        gt_locs = torch.cat(gt_locs, 0)

        has_preds = [x for x in data['has_preds']]
        has_preds = torch.cat(has_preds, 0)

        # does a ground-truth waypoint exist at the last timestep?
        has_last = has_preds[:, -1] == 1
        
        batch_idxs = []
        batch_idxs_edges = []
        actor_idcs = []
        sceneidx_to_batchidx_mapping = {}
        count_batchidx = 0
        count = 0
        for i in range(len(num_actors)):            
            batch_idxs.append(torch.ones(num_actors[i]) * count_batchidx)
            batch_idxs_edges.append(torch.ones(num_edges[i]) * count_batchidx)
            sceneidx_to_batchidx_mapping[int(scene_idxs[i].item())] = count_batchidx
            idcs = torch.arange(count, count + num_actors[i]).to(locs.device)
            actor_idcs.append(idcs)
            
            count_batchidx += 1
            count += num_actors[i]
        
        batch_idxs = torch.cat(batch_idxs).to(locs.device)
        batch_idxs_edges = torch.cat(batch_idxs_edges).to(locs.device)
        batch_size = torch.unique(batch_idxs).shape[0]

        ig_labels_metrics = [x for x in data['ig_labels_sparse']]
        ig_labels_metrics = torch.cat(ig_labels_metrics, 0)

        # 1 if agent has out-or-ingoing edge in ground-truth sparse interaction graph
        # These are the agents we use to evaluate interactive metrics
        is_connected = torch.zeros(locs.shape[0])
        count = 0
        offset = 0
        for k in range(len(num_actors)):
            N = num_actors[k]
            for i in range(N):
                for j in range(N):
                    if i >= j:
                        continue 
                    
                    # either an influencer or reactor in some DAG.
                    if ig_labels_metrics[count] > 0:                      

                        is_connected[offset + i] += 1
                        is_connected[offset + j] += 1 

                    count += 1
            offset += N

        is_connected = is_connected > 0     

        assert count == ig_labels_metrics.shape[0]

        dd = {
            'batch_size': batch_size,
            'batch_idxs': batch_idxs,
            'batch_idxs_edges': batch_idxs_edges, 
            'actor_idcs': actor_idcs,
            'actor_ctrs': actor_ctrs,
            'lane_graph': lane_graph,
            'feats': feats,
            'feat_psirads': psirads,
            'ctrs': ctrs,
            # 'orig': orig,
            # 'rot': rot,
            # 'theta': theta,
            'gt_locs': gt_locs,
            'has_preds': has_preds,
            'scene_idxs': scene_idxs,
            'sceneidx_to_batchidx_mapping': sceneidx_to_batchidx_mapping,
            'ig_labels': ig_labels,
            'gt_psirads': gt_psirads,
            'gt_vels': gt_vels,
            'agenttypes': agenttypes,
            'world_locs': world_locs,
            'has_obs': has_obs,
            'has_last': has_last,
            'graph': graph,
            'is_connected': is_connected
        }

        dd['shapes'] = shapes

        # elif self.dataset == "argoverse2":
        #     dd['is_scored'] = is_scored

        # dd = data-dictionary
        return dd
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        return optimizer
    
    def load_params_from_file(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])

    def init_dgl_graph(self, batch_idxs, ctrs, agenttypes, world_locs, has_preds):        
        n_scenarios = len(torch.unique(batch_idxs))
        graphs, labels = [], []
        for ii in range(n_scenarios):
            label = None

            # number of agents in the scene (currently > 0)
            si = ctrs[batch_idxs == ii].shape[0]
            assert si > 0

            # start with a fully-connected graph
            if si > 1:
                # off_diag = np.ones([si, si]) - np.eye(si)
                # rel_src = np.where(off_diag)[0]
                # rel_dst = np.where(off_diag)[1]
                off_diag = torch.ones(si, si, device=ctrs.device) - torch.eye(si, device=ctrs.device)
                rel_src, rel_dst = torch.where(off_diag != 0)

                graph = dgl.graph((rel_src, rel_dst))
            else:
                graph = dgl.graph(([], []), num_nodes=si)

            # separate graph for each scenario
            graph.ndata["ctrs"] = ctrs[batch_idxs == ii]
            # graph.ndata["rot"] = rot[batch_idxs == ii]
            # graph.ndata["orig"] = orig[batch_idxs == ii]
            graph.ndata["agenttypes"] = agenttypes[batch_idxs == ii].float()
            # ground truth future in SE(2)-transformed coordinates
            graph.ndata["ground_truth_futures"] = world_locs[batch_idxs == ii][:, self.observation_steps:]
            graph.ndata["has_preds"] = has_preds[batch_idxs == ii].float()
            
            graphs.append(graph)
            labels.append(label)
        
        graphs = dgl.batch(graphs)
        return graphs
    
    def build_stage_1_graph(self, graph, x, agenttypes, actor_idcs, actor_ctrs, lane_graph):
        all_edges = [x.unsqueeze(1) for x in graph.edges('uv')]
        all_edges = torch.cat(all_edges, 1)
        
        stage_1_graph = dgl.graph((all_edges[:, 0], all_edges[:, 1]), num_nodes = graph.num_nodes())
        stage_1_graph.ndata["ctrs"] = graph.ndata["ctrs"]
        stage_1_graph.ndata["rot"] = graph.ndata["rot"]
        stage_1_graph.ndata["orig"] = graph.ndata["orig"]
        stage_1_graph.ndata["agenttypes"] = graph.ndata["agenttypes"].float()

        stage_1_graph = self.pretrained_relation_header.feature_encoder(stage_1_graph, x, agenttypes, actor_idcs, actor_ctrs, lane_graph)

        return stage_1_graph

    def get_loss(self, graph, batch_idxs, res, agenttypes, has_preds, gt_locs, batch_size, ig_labels, epoch):
        
        huber_loss = nn.HuberLoss(reduction='none')
        
        if self.proposal_header:
            ### Proposal Regression Loss
            has_preds_mask = has_preds.unsqueeze(-1).unsqueeze(-1)
            has_preds_mask = has_preds_mask.expand(has_preds_mask.shape[0], has_preds_mask.shape[1], self.num_proposals, 2).bool() 

            proposals = res["proposals"]
            
            if self.supervise_vehicles:
                # only compute loss on vehicle trajectories
                vehicle_mask = agenttypes[:, 1].bool()
            else:
                # compute loss on all trajectories
                vehicle_mask = torch.ones(agenttypes[:, 1].shape).bool() 
            
            has_preds_mask = has_preds_mask[vehicle_mask]
            proposals = proposals[vehicle_mask]
            gt_locs = gt_locs[vehicle_mask]
            batch_idxs = batch_idxs[vehicle_mask]

            target = torch.stack([gt_locs] * self.num_proposals, dim=2) 

            # Regression loss
            loss_prop_reg = huber_loss(proposals, target)
            loss_prop_reg = loss_prop_reg * has_preds_mask

            b_s = torch.zeros((batch_size, self.num_proposals)).to(loss_prop_reg.device)
            count = 0
            for i, batch_num_nodes_i in enumerate(graph.batch_num_nodes()):
                batch_num_nodes_i = batch_num_nodes_i.item()
                
                batch_loss_prop_reg = loss_prop_reg[count:count+batch_num_nodes_i]    
                # divide by number of agents in the scene        
                b_s[i] = torch.sum(batch_loss_prop_reg, (0, 1, 3)) / batch_num_nodes_i

                count += batch_num_nodes_i

            # sanity check
            assert batch_size == (i + 1)

            loss_prop_reg = torch.min(b_s, dim=1)[0].mean()        
        
        if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 2):
            ### Regression Loss
            # has_preds: [N, 30]
            # res["loc_pred"]: [N, 30, 6, 2]
            has_preds_mask = has_preds.unsqueeze(-1).unsqueeze(-1)
            has_preds_mask = has_preds_mask.expand(has_preds_mask.shape[0], has_preds_mask.shape[1], self.num_joint_modes, 2).bool() 
            
            loc_pred = res["loc_pred"]
            
            if not self.proposal_header:
                if self.supervise_vehicles:
                    vehicle_mask = agenttypes[:, 1].bool()
                else:
                    vehicle_mask = torch.ones(agenttypes[:, 1].shape).bool() 
    
                gt_locs = gt_locs[vehicle_mask]
                batch_idxs = batch_idxs[vehicle_mask]
            
            has_preds_mask = has_preds_mask[vehicle_mask]
            loc_pred = loc_pred[vehicle_mask]
            
            target = torch.stack([gt_locs] * self.num_joint_modes, dim=2) 

            # Regression loss
            reg_loss = huber_loss(loc_pred, target)

            # 0 out loss for the indices that don't have a ground-truth prediction.
            reg_loss = reg_loss * has_preds_mask

            b_s = torch.zeros((batch_size, self.num_joint_modes)).to(reg_loss.device)
            count = 0
            for i, batch_num_nodes_i in enumerate(graph.batch_num_nodes()):
                batch_num_nodes_i = batch_num_nodes_i.item()
                
                batch_reg_loss = reg_loss[count:count+batch_num_nodes_i]    
                # divide by number of agents in the scene        
                b_s[i] = torch.sum(batch_reg_loss, (0, 1, 3)) / batch_num_nodes_i

                count += batch_num_nodes_i

            # sanity check
            assert batch_size == (i + 1)

            loss_reg = torch.min(b_s, dim=1)[0].mean()      

        # Relation Loss
        if self.learned_relation_header:
            if (not self.two_stage_training) or (self.two_stage_training and self.training_stage == 1):
                if self.focal_loss:
                    ce_loss = FocalLoss(weight=torch.Tensor([self.weight_0, self.weight_1, self.weight_2]) , gamma=self.gamma, reduction='mean')
                else:
                    ce_loss = nn.CrossEntropyLoss(weight=torch.Tensor([self.weight_0, self.weight_1, self.weight_2]) )

                # Now compute relation cross entropy loss
                relations_preds = res["edge_logits"]
                relations_gt = ig_labels.to(relations_preds.device).long()

                loss_rel = ce_loss(relations_preds, relations_gt)     
        
        if not self.two_stage_training:
            loss = loss_reg
            
            if self.proposal_header:
                loss = loss + self.proposal_coef * loss_prop_reg

            if self.learned_relation_header:
                loss = loss + self.rel_coef * loss_rel

            loss_dict = {"total_loss": loss,
                        "loss_reg": loss_reg
                        }

            if self.proposal_header:
                loss_dict["loss_prop_reg"] = loss_prop_reg * self.proposal_coef
            
            if self.learned_relation_header:
                loss_dict["loss_rel"] = self.rel_coef * loss_rel                   

        else:
            if self.training_stage == 1:
                loss = self.rel_coef * loss_rel
                if self.proposal_header:
                    loss = loss + loss_prop_reg * self.proposal_coef
                
                loss_dict = {"total_loss": loss,
                             "loss_rel": self.rel_coef * loss_rel} 

                if self.proposal_header:
                    loss_dict["loss_prop_reg"] = loss_prop_reg * self.proposal_coef

            else:
                loss = loss_reg
                
                if self.proposal_header:
                    loss = loss + loss_prop_reg * self.proposal_coef
                
                loss_dict = {"total_loss": loss,
                             "loss_reg": loss_reg} 
                             
                if self.proposal_header:
                    loss_dict["loss_prop_reg"] = loss_prop_reg * self.proposal_coef
        
        return loss_dict
     
    def compute_metrics(self, predictions, batch, dataloader_idx):
        metrics = self.val_seen_metrics if dataloader_idx == 0 else self.val_unseen_metrics
        
        K_pred_traj_fake_rel = predictions
        ades = [[], [], []]
        fdes = [[], [], []]
        mrs = [[], [], []]
        K_total_preds = [0, 0, 0]
        K_total_final_preds = [0, 0, 0]
        K_pred_traj_final = []

        traj_gt = batch['gt_locs'].permute(1, 0, 2)
        has_preds = batch['has_preds'].permute(1, 0)

        for k in range(self.num_joint_modes):
            pred_traj = predictions[:, :, k].permute(1, 0, 2)
            
            K_pred_traj_final.append(pred_traj[-1])
            for tt in range(3):
                ade = cal_ade(
                    traj_gt[:self.prediction_steps//3*(tt+1)-1],
                    pred_traj[:self.prediction_steps//3*(tt+1)-1],
                    has_preds[:self.prediction_steps//3*(tt+1)-1], mode='raw'
                )
                fde = cal_fde(
                    traj_gt[:self.prediction_steps//3*(tt+1)-1],
                    pred_traj[:self.prediction_steps//3*(tt+1)-1],
                    has_preds[:self.prediction_steps//3*(tt+1)-1], mode='raw'
                )
                mr = cal_mr(
                    traj_gt[:self.prediction_steps//3*(tt+1)-1],
                    pred_traj[:self.prediction_steps//3*(tt+1)-1],
                    has_preds[:self.prediction_steps//3*(tt+1)-1], mode='raw'
                )
                ades[tt].append(ade)
                fdes[tt].append(fde)
                mrs[tt].append(mr)
                K_total_preds[tt] = has_preds[:self.prediction_steps//3*(tt+1)-1].sum()
                K_total_final_preds[tt] = has_preds[self.prediction_steps//3*(tt+1)-1].sum()
        for tt in range(3):
            metrics["sum_minADE@{}s".format(tt+1)].append(torch.stack(ades[tt], 1).min(1)[0].sum().item())
            metrics["sum_minFDE@{}s".format(tt+1)].append(torch.stack(fdes[tt], 1).min(1)[0].sum().item())
            metrics["sum_minMR@{}s".format(tt+1)].append(torch.stack(mrs[tt], 1).min(1)[0].sum().item())
            metrics["count_final@{}s".format(tt+1)].append(K_total_final_preds[tt].item())
            metrics["count_average@{}s".format(tt+1)].append(K_total_preds[tt].item())
        
        metrics["diversity"].append(cal_diversity(torch.stack(K_pred_traj_final, 0)))
        metrics["diversity_N"].append(has_preds.size(1))
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