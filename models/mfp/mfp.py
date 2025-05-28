import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.ftt.utils.util import relative_to_curve, relative_to_abs, cal_ade, cal_fde, cal_mr, cal_diversity
from models.laneGCN.layers import  MapNet
from models.laneGCN.laneGCN import laneGCN_graph_gather
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from torch_scatter import scatter_max

class MFP(pl.LightningModule):
    def __init__(self, config):
        # code adopted from https://github.com/apple/ml-multiple-futures-prediction
        super(MFP, self).__init__()
        self.config = config
        self.learning_rate = config["args"].learning_rate
        self.l2_weight_decay = config["args"].l2_weight_decay
        self.learning_rate_decay = config["args"].learning_rate_decay
        self.input_embedding_size = 32
        self.encoder_size = 64
        self.hidden_fac = 2
        self.dec_fac = 2
        self.bi_direc_fac = 2
        self.dyn_embedding_size = 32
        self.use_context = True
        self.modes = config['num_mods']
        self.use_forcing = 0
        self.nbr_atten_embedding_size = 80
        self.st_enc_pos_size = 8
        self.st_enc_hist_size = self.nbr_atten_embedding_size
        self.posi_enc_dim = self.st_enc_pos_size
        self.posi_enc_ego_dim = 2
        self.decoder_size = 128
        self.bStepByStep = True
        self.bi_direc = True
        self.out_length = config['num_preds']

        self.init_rbf_state_enc(in_dim=self.encoder_size*self.hidden_fac)
        # Input embedding layer
        self.ip_emb = torch.nn.Linear(config['input_dims'][1], self.input_embedding_size)

        # Encoding RNN
        self.num_layers=2
        self.enc_lstm = torch.nn.GRU(self.input_embedding_size,self.encoder_size,    # type: ignore
                                   num_layers=self.num_layers, bidirectional=False)
        # Dynamics embeddings.
        self.dyn_emb = torch.nn.Linear(self.encoder_size*self.hidden_fac, self.dyn_embedding_size) #type: ignore

        context_feat_size = config["n_map"] if self.use_context else 0
        self.dec_lstm = []
        self.op = []
        for k in range(self.modes):
            self.dec_lstm.append(torch.nn.GRU(self.nbr_atten_embedding_size + self.dyn_embedding_size + context_feat_size+self.posi_enc_dim+self.posi_enc_ego_dim, # type: ignore
                                            self.decoder_size, num_layers=self.num_layers, bidirectional=self.bi_direc ))
            self.op.append( torch.nn.Linear(self.decoder_size*self.dec_fac, 5) ) #type: ignore

            self.op[k] = self.op[k]
            self.dec_lstm[k] = self.dec_lstm[k]

        self.dec_lstm = torch.nn.ModuleList(self.dec_lstm)
        self.op = torch.nn.ModuleList(self.op)

        self.op_modes = torch.nn.Linear(self.nbr_atten_embedding_size + self.dyn_embedding_size + context_feat_size, self.modes)

        # Nonlinear activations.
        self.leaky_relu = torch.nn.LeakyReLU(0.1) #type: ignore
        self.relu = torch.nn.ReLU() #type: ignore
        self.softmax = torch.nn.Softmax(dim=1) #type: ignore
        if self.use_context:
            self.context_net = ContextNet(config)


    def init_rbf_state_enc(self, in_dim: int ) -> None:
        """Initialize the dynamic attentional RBF encoder.
        Args:
          in_dim is the input dim of the observation.
        """
        self.sec_in_dim = in_dim
        self.extra_pos_dim = 2

        self.sec_in_pos_dim     = 2
        self.sec_key_dim        = 8
        self.sec_key_hidden_dim = 32

        self.sec_hidden_dim     = 32
        self.scale = 1.0
        self.slot_key_scale = 1.0
        self.num_slots = 8
        self.slot_keys = []

        # Network for computing the 'key'
        self.sec_key_net = torch.nn.Sequential( #type: ignore
                              torch.nn.Linear(self.sec_in_dim+self.extra_pos_dim, self.sec_key_hidden_dim),
                              torch.nn.ReLU(),
                              torch.nn.Linear(self.sec_key_hidden_dim, self.sec_key_dim)
                           )

        for ss in range(self.num_slots):
            self.slot_keys.append( torch.nn.Parameter( self.slot_key_scale*torch.randn( self.sec_key_dim, 1, dtype=torch.float32) ) ) #type: ignore
        self.slot_keys = torch.nn.ParameterList( self.slot_keys )  # type: ignore

        # Network for encoding a scene-level contextual feature.
        self.sec_hist_net = torch.nn.Sequential( #type: ignore
                              torch.nn.Linear(self.sec_in_dim*self.num_slots, self.sec_hidden_dim),
                              torch.nn.ReLU(),
                              torch.nn.Linear(self.sec_hidden_dim, self.sec_hidden_dim),
                              torch.nn.ReLU(),
                              torch.nn.Linear(self.sec_hidden_dim, self.st_enc_hist_size)
                            )

        # Encoder position of other's into a feature network, input should be normalized to ref_pos.
        self.sec_pos_net = torch.nn.Sequential( #type: ignore
                              torch.nn.Linear(self.sec_in_pos_dim*self.num_slots, self.sec_hidden_dim),
                              torch.nn.ReLU(),
                              torch.nn.Linear(self.sec_hidden_dim, self.sec_hidden_dim),
                              torch.nn.ReLU(),
                              torch.nn.Linear(self.sec_hidden_dim, self.st_enc_pos_size)
                            )


    def rbf_state_enc_get_attens(self, nbrs_enc, ref_pos, edge_index) -> List[torch.Tensor]:
        """Computing the attention over other agents.
        Args:
          nbrs_info_this is a list of list of (nbr_batch_ind, nbr_id, nbr_ctx_ind)
        Returns:
          attention weights over the neighbors.
        """
        src, dst = edge_index
        pos_enc = ref_pos[src] - ref_pos[dst]
        Key = self.sec_key_net( torch.cat( (nbrs_enc,pos_enc),dim=1) )
        attens0 = []
        for slot in self.slot_keys:
            attens0.append( torch.exp( -self.scale*(Key-torch.t(slot)).norm(dim=1)) )
        Atten = torch.stack(attens0, dim=0) # e.g. num_keys x num_agents
        attens = []
        for n in range(ref_pos.size(0)):
            attens.append( Atten[:, src==n] )
        return attens

    def rbf_state_enc_hist_fwd(self, attens, nbrs_enc, ref_pos, edge_index) -> torch.Tensor:
        """Computes dynamic state encoding.
        Computes dynica state encoding with precomputed attention tensor and the
        RNN based encoding.
        Args:
          attens is a list of [ [slots x num_neighbors]]
          nbrs_enc is num_agents by input_dim
        Returns:
          feature vector
        """
        src, dst = edge_index
        out = []
        for n in range(ref_pos.size(0)):
            nbr_feat = nbrs_enc[src==n]
            out.append(torch.mm( attens[n], nbr_feat ) )
        st_enc = torch.stack(out, dim=0).view(len(out),-1)
        return self.sec_hist_net(st_enc)

    def rbf_state_enc_pos_fwd(self, attens: List, ref_pos:torch.Tensor, fut_t: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Computes the features from dynamic attention for interactive rollouts.
        Args:
          attens is a list of [ [slots x num_neighbors]]
          ref_pos should be (num_agents by 2)
        Returns:
          feature vector
        """
        src, des = edge_index
        fut_t = fut_t[des]
        out = []
        for n in range(len(attens)):
            out.append( torch.mm(attens[n], fut_t[src==n] - ref_pos[n,:]) )
        pos_enc = torch.stack(out, dim=0).view(len(attens),-1)
        return self.sec_pos_net(pos_enc)


    def nll_loss(self, pred, data, has_preds):
        x_mean = pred[:,:,0]
        y_mean = pred[:,:,1]
        x_sigma = pred[:,:,2]
        y_sigma = pred[:,:,3]
        rho = pred[:,:,4]
        ohr = torch.pow(1-torch.pow(rho,2),-0.5) # type: ignore

        data = data.permute(1, 0, 2)
        has_preds = has_preds.permute(1, 0)

        last = has_preds.float() + 0.1 * torch.arange(self.config['num_preds']).float().to(
            has_preds.device
        ) / float(self.config['num_preds'])
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0
        has_preds = has_preds[mask]

        x = data[:,:, 0]; y = data[:,:, 1]

        results = torch.pow(ohr, 2)*(torch.pow(x_sigma, 2)*torch.pow(x-x_mean, 2) + torch.pow(y_sigma, 2)*torch.pow(y-y_mean, 2)
                    -2*rho*torch.pow(x_sigma, 1)*torch.pow(y_sigma, 1)*(x-x_mean)*(y-y_mean)) - torch.log(x_sigma*y_sigma*ohr)

        results = results[mask][has_preds]

        return torch.mean(results)

    def nll_loss_per_sample(self, pred, data, has_preds, mask):

        """NLL averages across steps and dimensions, but not samples (agents)."""
        x_mean = pred[:,:,0]
        y_mean = pred[:,:,1]
        x_sigma = pred[:,:,2]
        y_sigma = pred[:,:,3]
        rho = pred[:,:,4]
        ohr = torch.pow(1-torch.pow(rho,2),-0.5) # type: ignore

        has_preds = has_preds[mask]

        x = data[:,:, 0]; y = data[:,:, 1]
        results = torch.pow(ohr, 2)*(torch.pow(x_sigma, 2)*torch.pow(x-x_mean, 2) + torch.pow(y_sigma, 2)*torch.pow(y-y_mean, 2)
                    -2*rho*torch.pow(x_sigma, 1)*torch.pow(y_sigma, 1)*(x-x_mean)*(y-y_mean)) - torch.log(x_sigma*y_sigma*ohr)


        loss = []
        for i in range(has_preds.size(0)):
            loss.append(results[mask][i][has_preds[i]].sum() / has_preds[i].float().sum())

        #results = results[mask]*has_preds
        return torch.stack(loss, 0)
        #return torch.mean(results, dim=1)

    def nll_loss_multimodes(self, pred, data, has_preds, modes_pred, noise=0.0):
        """NLL loss multimodes for training.
            Args:
            pred is a list (with N modes) of predictions
            data is ground truth
            noise is optional
        """
        modes = len(pred)
        batch_sz, nSteps, dim = pred[0].shape

        data = data.permute(1, 0, 2)
        has_preds = has_preds.permute(1, 0)

        last = has_preds.float() + 0.1 * torch.arange(self.config['num_preds']).float().to(
            has_preds.device
        ) / float(self.config['num_preds'])
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0
        batch_sz = mask.sum()
        log_lik = np.zeros( (batch_sz, modes) )


        with torch.no_grad():
            for kk in range(modes):
                nll = self.nll_loss_per_sample(pred[kk], data, has_preds, mask)
                log_lik[:,kk] = -nll.cpu().numpy()

        priors = modes_pred[mask].detach().cpu().numpy()

        log_posterior_unnorm = log_lik + np.log(priors).reshape((-1, modes)) #[TotalObjs, net.modes]
        log_posterior_unnorm += np.random.randn( *log_posterior_unnorm.shape)*noise
        from scipy import special
        log_posterior = log_posterior_unnorm - special.logsumexp( log_posterior_unnorm, axis=1 ).reshape((batch_sz, 1))
        post_pr = np.exp(log_posterior)  #[TotalObjs, net.modes]

        post_pr = torch.tensor(post_pr).float().to(data.device)
        loss = 0.0
        for kk in range(modes):
            nll_k = self.nll_loss_per_sample(pred[kk], data, has_preds, mask)*post_pr[:,kk]
            loss += nll_k.sum()/float(batch_sz)

        kl_loss = torch.nn.KLDivLoss(reduction='batchmean') #type: ignore
        loss += kl_loss( torch.log(modes_pred[mask]), post_pr)
        return loss

    def Gaussian2d(self, x):
        x_mean  = x[:,:,0]
        y_mean  = x[:,:,1]
        sigma_x = torch.exp(x[:,:,2])
        sigma_y = torch.exp(x[:,:,3])
        rho     = torch.tanh(x[:,:,4])
        return torch.stack([x_mean, y_mean, sigma_x, sigma_y, rho], dim=2)

    def forward(self, batch):
        hist = torch.cat([batch.obs_traj_rel, batch.obs_info], -1)
        _, hist_enc = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        hist_enc = hist_enc.permute(1,0,2).contiguous()
        ego_hist_enc = self.leaky_relu(self.dyn_emb( hist_enc.view(hist_enc.shape[0], -1) ))
        nbrs_enc = hist_enc.view(hist_enc.shape[0], -1)[batch.edge_index[1]]
        ref_pos = torch.cat(batch.veh_ctrs, dim=0)
        attens = self.rbf_state_enc_get_attens(nbrs_enc, ref_pos, batch.edge_index)
        #print(len(attens), [att.size() for att in attens])
        nbr_atten_enc = self.rbf_state_enc_hist_fwd(attens, nbrs_enc, ref_pos, batch.edge_index)
        if self.use_context: #context encoding
            context_enc = self.context_net(batch)
            enc = torch.cat((nbr_atten_enc, ego_hist_enc, context_enc),1)
        else:
            enc = torch.cat((nbr_atten_enc, ego_hist_enc),1)

        ######################################################################################################
        modes_pred = None if self.modes==1 else self.softmax(self.op_modes(enc))
        if self.training:
            use_forcing = self.use_forcing
        else:
            use_forcing = 0
        fut = batch.fut_traj
        fut_preds = self.decode(enc, attens, batch.edge_index, ref_pos, fut, self.bStepByStep, use_forcing)
        if self.modes == 1:
            loss = self.nll_loss(fut_preds[0], batch.fut_traj_rel, batch.has_preds)
            return {"pred":fut_preds[0].permute(1, 0, 2)[:,:,:2],
                    "modes_pred":modes_pred,
                    "loss":loss
                    }
        else:
            loss = self.nll_loss_multimodes(fut_preds, batch.fut_traj_rel, batch.has_preds, modes_pred)
            _, row_idcs = modes_pred.max(-1)
            #col_idcs = torch.arange(num_vehs).to(fut.device)
            pred = []
            for i in range(len(row_idcs)):
                ind = row_idcs[i]
                pred.append(fut_preds[ind][i])
            pred = torch.stack(pred, dim=0)
            return {"reg":[p[:,:,:2] for p in fut_preds],
                    "pred":pred.permute(1,0,2)[:,:,:2],#fut_preds.permute(1, 0, 2)[:,:,:2],
                    "modes_pred":modes_pred,
                    "loss":loss
                    }

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
    
    def decode(self, enc: torch.Tensor, attens:List, edge_index:torch.Tensor, ref_pos:torch.Tensor,
                  fut:torch.Tensor, bStepByStep:bool, use_forcing:Any ) -> List[torch.Tensor]:
        """Decode the future trajectory using RNNs.

        Given computed feature vector, decode the future with multimodes, using
        dynamic attention and either interactive or non-interactive rollouts.
        Args:
          enc: encoded features, one per agent.
          attens: attentional weights, list of objs, each with dimenstion of [8 x 4] (e.g.)
          nbrs_info_this: information on who are the neighbors
          ref_pos: the current postion (reference position) of the agents.
          fut: future trajectory (only useful for teacher or classmate forcing)
          bStepByStep: interactive or non-interactive rollout
          use_forcing: 0: None. 1: Teacher-forcing. 2: classmate forcing.
        Returns:
          fut_pred: a list of predictions, one for each mode.
          modes_pred: prediction over latent modes.
        """
        if not bStepByStep: # Non-interactive rollouts
            enc = enc.repeat(self.out_length, 1, 1)
            pos_enc = torch.zeros( self.out_length, enc.shape[1], self.posi_enc_dim+self.posi_enc_ego_dim, device=enc.device )
            enc2 = torch.cat( (enc, pos_enc), dim=2)
            fut_preds = []
            for k in range(self.modes):
                h_dec, _ = self.dec_lstm[k](enc2)
                h_dec = h_dec.permute(1, 0, 2)
                fut_pred = self.op[k](h_dec)
                fut_pred = self.Gaussian2d(fut_pred)

                fut_preds.append(fut_pred)
            return fut_preds
        else:
            batch_sz =  enc.shape[0]

            fut_preds = []
            for k in range(self.modes):
                direc = 2 if self.bi_direc else 1
                hidden = torch.zeros(self.num_layers*direc, batch_sz, self.decoder_size).to(fut.device)
                preds: List[torch.Tensor] = []
                cur_pos = ref_pos
                for t in range(self.out_length):
                    if t == 0: # Intial timestep.
                        pred_fut_t = cur_pos
                        ego_fut_t = cur_pos
                    else:
                        if use_forcing == 0:
                            pred_fut_t = cur_pos
                            ego_fut_t = pred_fut_t
                        elif use_forcing == 1:
                            pred_fut_t = fut[t-1,:,:]
                            ego_fut_t = pred_fut_t
                        else:
                            pred_fut_t = fut[t-1,:,:]
                            ego_fut_t = cur_pos

                    if attens == None:
                        pos_enc =  torch.zeros(batch_sz, self.posi_enc_dim, device=enc.device )
                    else:
                        pos_enc = self.rbf_state_enc_pos_fwd(attens, ref_pos, pred_fut_t, edge_index)

                    enc_large = torch.cat( ( enc.view(1,enc.shape[0],enc.shape[1]),
                                           pos_enc.view(1,batch_sz, self.posi_enc_dim),
                                           ego_fut_t.view(1, batch_sz, self.posi_enc_ego_dim ) ), dim=2 )

                    out, hidden = self.dec_lstm[k]( enc_large, hidden)
                    pred = self.Gaussian2d(self.op[k](out))
                    cur_pos = cur_pos + pred[0,:,:2]
                    preds.append( pred )
                fut_pred_k = torch.cat(preds,dim=0)
                fut_preds.append(fut_pred_k.permute(1, 0, 2))

            return fut_preds

class ContextNet(nn.Module):
    def __init__(self, config):
        super(ContextNet, self).__init__()
        self.config = config
        self.map_net = MapNet(self.config)

    def forward(self, batch):
        graph = laneGCN_graph_gather(batch["graphs"])
        nodes, node_idcs, node_ctrs = self.map_net(graph)
        actor_idcs = batch.veh_batch
        actor_ctrs = batch.veh_ctrs

        batch_size = len(actor_idcs)
        hi, wi = [], []
        hi_count, wi_count = 0, 0
        for i in range(batch_size):
            dist = actor_ctrs[i].view(-1, 1, 2) - node_ctrs[i].view(1, -1, 2)
            dist = torch.sqrt((dist ** 2).sum(2))
            mask = dist <= self.config["map2actor_dist"]

            idcs = torch.nonzero(mask, as_tuple=False)
            if len(idcs) == 0:
                continue

            hi.append(idcs[:, 0] + hi_count)
            wi.append(idcs[:, 1] + wi_count)
            hi_count += len(actor_idcs[i])
            wi_count += len(node_idcs[i])
        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)
        ctx, _ = scatter_max(nodes[wi], hi, dim=0, dim_size=hi_count)
        return ctx


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