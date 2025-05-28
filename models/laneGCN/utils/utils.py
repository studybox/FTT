import torch 


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def cal_ade(pred_traj_gt, pred_traj_fake, has_preds, mode='sum'):
    ade = displacement_error(pred_traj_fake, pred_traj_gt, has_preds, mode=mode)
    return ade
def cal_fde(pred_traj_gt, pred_traj_fake, has_preds, mode='sum'):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], has_preds[-1], mode=mode)
    return fde
def cal_ede(pred_traj_gt, pred_traj_fake, has_preds):
    seq_len = pred_traj_gt.size(0)
    ede = []
    for t in range(seq_len):
        de = final_displacement_error(pred_traj_fake[t], pred_traj_gt[t], has_preds[t])
        ede.append(de)
    return ede

def cal_mr(pred_traj_gt, pred_traj_fake, has_preds, mode='sum'):
    loss = pred_traj_gt[-1][has_preds[-1]] - pred_traj_fake[-1][has_preds[-1]]
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=-1))
    if mode == 'raw':
        return loss > 2.0
    else:
        return torch.sum(loss > 2.0)

def cal_diversity(K_pred_traj_final):
    k, n, _  = K_pred_traj_final.size()
    diversity = K_pred_traj_final.view(k, 1, n, 2) - K_pred_traj_final.view(1, k, n, 2)
    diversity = diversity**2
    diversity = torch.sqrt(diversity.sum(dim=-1))
    i = torch.arange(k).view(1,k).repeat(k,1) > torch.arange(k).view(k,1).repeat(1,k)
    return torch.sum(diversity[i].mean(0)).item()

def displacement_error(pred_traj, pred_traj_gt, has_preds, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    #seq_len, _, _ = pred_traj.size()
    #loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = pred_traj[has_preds] - pred_traj_gt[has_preds]
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=-1))
    #loss = loss.sum(dim=2).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss
def final_displacement_error(pred_pos, pred_pos_gt, has_preds, mode='sum'):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt[has_preds] - pred_pos[has_preds]
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=-1))
    #loss = loss.sum(dim=1)
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)