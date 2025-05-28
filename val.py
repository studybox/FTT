from argparse import ArgumentParser
import pytorch_lightning as pl
import os.path as osp
from config import Config
from datasets import Datasets, Dataloaders, collate_fns, mix_dataset
from models import Models

def val(args):
    collate_fn = collate_fns[args.model]
    train_files, val_files_1, val_files_2 = mix_dataset(Config)
    datasets = {
        "val_unseen":Datasets[args.model](Config[args.dataset], processed_files=val_files_1),
        "val_seen":Datasets[args.model](Config[args.dataset], processed_files=val_files_2)
    }
    dataloaders = [
        Dataloaders[args.model](datasets["val_seen"], batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn),#, num_workers=args.num_workers, pin_memory=True, persistent_workers=True),
        Dataloaders[args.model](datasets["val_unseen"], batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)#, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    ]
    Config[args.dataset]["args"] = args  
    model = Models[args.model](Config[args.dataset])
    if args.load_model:
        pretrain_ckpt = osp.join(Config[args.dataset]["model_dir"], Config[args.dataset]["loaded_model"][args.model])
        model.load_params_from_file(filename=pretrain_ckpt)
    
    trainer = pl.Trainer(accelerator=args.accelerator,
                         devices=args.devices,
                         strategy='ddp', num_sanity_val_steps=0)
    trainer.validate(model, dataloaders)

if __name__ == '__main__':
    pl.seed_everything(2, workers=True)
    parser = ArgumentParser()
    parser.add_argument('--accelerator', default='gpu', type=str)
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--dataset', default="all", type=str)
    parser.add_argument('--model', default="ftt", type=str)
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--learning_rate_decay', default=0.9999, type=float)
    parser.add_argument('--l2_weight_decay', default=0.0, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_iterations', default=10000, type=int)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--batch_norm', default=1, type=bool)
    parser.add_argument('--mlp_dim', default=1024, type=int)
    parser.add_argument('--p_z_x_MLP', default=32, type=int)
    parser.add_argument('--encoder_h_dim_g', default=128, type=int)
    parser.add_argument('--decoder_h_dim_g', default=128, type=int)
    parser.add_argument('--noise_dim', default=(1,), type=tuple)
    parser.add_argument('--noise_type', default='gaussian')
    parser.add_argument('--noise_mix_type', default='ped')
    parser.add_argument('--clipping_threshold_g', default=1.0, type=float)
    parser.add_argument('--g_learning_rate', default=1e-3, type=float)
    parser.add_argument('--g_steps', default=2, type=int)
    parser.add_argument('--bottleneck_dim', default=1024, type=int)
    parser.add_argument('--d_type', default='local', type=str)
    parser.add_argument('--encoder_h_dim_d', default=64, type=int)
    parser.add_argument('--d_learning_rate', default=1e-3, type=float)
    parser.add_argument('--d_steps', default=1, type=int)
    parser.add_argument('--clipping_threshold_d', default=1.0, type=float)

    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--load_model', default=True, type=bool)
    parser.add_argument('--custom_prefix', default='', type=str)

    parser.add_argument('--use_gpu', default=1, type=bool)

    parser.add_argument('--pool_every_timestep', default=1, type=bool)
    args = parser.parse_args()

    val(args)