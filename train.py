from argparse import ArgumentParser
import pytorch_lightning as pl
import os
import os.path as osp
import torch
from config import Config
from models import Models
torch.set_float32_matmul_precision('medium')

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from datasets import Datasets, Dataloaders, collate_fns, mix_dataset

def train(args):
    collate_fn = collate_fns[args.model]
    train_files, _, val_files_seen = mix_dataset(Config)
    train_dataset = Datasets[args.model](Config[args.dataset], processed_files=train_files)
    val_dataset = Datasets[args.model](Config[args.dataset], processed_files=val_files_seen)
    
    train_dataloader = Dataloaders[args.model](train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)#, num_workers=args.num_workers )
    val_dataloader = Dataloaders[args.model](val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)#, num_workers=args.num_workers)

    Config[args.dataset]["args"] = args  
    model = Models[args.model](Config[args.dataset])
    if args.load_model:
        pretrain_ckpt = osp.join(Config[args.dataset]["model_dir"], Config[args.dataset][args.model]["loaded_model"])
        model.load_params_from_file(filename=pretrain_ckpt)

    if not os.path.exists(args.save_ckpt_path+args.model):
        os.makedirs(args.save_ckpt_path+args.model, exist_ok=True)
    model_checkpoint = ModelCheckpoint(dirpath=args.save_ckpt_path+args.model,
                                       filename="{epoch:02d}-{val_loss:.2f}",
                                       monitor='val_loss',
                                       every_n_epochs=1,
                                       save_top_k=5,
                                       mode='min')
    strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
    trainer = pl.Trainer(accelerator=args.accelerator,
                         devices=args.devices,
                         logger=None if args.debug else WandbLogger(project="ftt", name=args.model, id=args.model),
                         max_epochs=args.num_epochs,
                         callbacks=[model_checkpoint, ],
                         gradient_clip_val=0.5,
                         strategy=strategy, num_sanity_val_steps=1)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == '__main__':
    pl.seed_everything(2, workers=True)
    parser = ArgumentParser()
    parser.add_argument('--accelerator', default='gpu', type=str)
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--debug', default=True, type=bool)
    parser.add_argument('--dataset', default="all", type=str)
    parser.add_argument('--model', default="ftt", type=str)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--learning_rate_decay', default=0.9999, type=float)
    parser.add_argument('--l2_weight_decay', default=0.0, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
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
    parser.add_argument('--save_ckpt_path', type=str, default="/home/boqi/CoDriving/planning/FTT/model/")
    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--load_model', default=False, type=bool)
    parser.add_argument('--custom_prefix', default='', type=str)

    parser.add_argument('--use_gpu', default=1, type=bool)

    parser.add_argument('--pool_every_timestep', default=1, type=bool)
    args = parser.parse_args()

    train(args)