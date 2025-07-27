import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
import time
import pandas as pd
from datetime import datetime
from model.PDG2Seq_Hypergraph import PDG2Seq_Hypergraph as Network
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from lib.metrics import MAE_torch
import warnings
warnings.filterwarnings('ignore')

def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

def load_distance_matrix(dataset_name):
    """Load distance matrix for NYC-Bike dataset"""
    if dataset_name == 'NYC-Bike':
        distance_file = './data/NYC-Bike/dis_bb.csv'
        if os.path.exists(distance_file):
            distance_matrix = pd.read_csv(distance_file, header=None).values
            print(f'Loaded NYC-Bike distance matrix: {distance_matrix.shape}')
            return distance_matrix
        else:
            print(f'Distance matrix not found: {distance_file}')
    return None

#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default='NYC-Bike', type=str)
args.add_argument('--mode', default='train', type=str)
args.add_argument('--device', default='cuda:0', type=str, help='indices of GPUs')
args.add_argument('--debug', default='False', type=eval)
args.add_argument('--model', default='PDG2Seq_HyperGCN', type=str)
args.add_argument('--cuda', default=True, type=bool)
args1 = args.parse_args()

#get configuration
config_file = './config_file/{}_{}.conf'.format(args1.dataset, args1.model)
print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)

#data
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--tod', default=config['data']['tod'], type=eval)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
args.add_argument('--add_time_in_day', default=config['data']['add_time_in_day'], type=eval)
args.add_argument('--add_day_in_week', default=config['data']['add_day_in_week'], type=eval)
args.add_argument('--steps_per_day', default=config['data']['steps_per_day'], type=int)
args.add_argument('--steps_per_week', default=config['data']['steps_per_week'], type=int)

#model
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
args.add_argument('--cheb_k', default=config['model'].get('cheb_order', 2), type=int)
args.add_argument('--time_dim', default=config['model']['time_dim'], type=int)

# Hypergraph config
if 'hypergraph' in config:
    args.add_argument('--use_hypergraph', default=config['hypergraph'].get('use_hypergraph', True), type=eval)
    args.add_argument('--hyperedge_types', default=config['hypergraph'].get('hyperedge_types', 'pick_drop,geo,temporal,correlation,pattern'), type=str)
    args.add_argument('--pick_drop_threshold', default=config['hypergraph'].get('pick_drop_threshold', 0.9), type=float)
    args.add_argument('--geo_threshold', default=config['hypergraph'].get('geo_threshold', 0.1), type=float)
    args.add_argument('--temporal_threshold', default=config['hypergraph'].get('temporal_threshold', 0.8), type=float)
    args.add_argument('--correlation_threshold', default=config['hypergraph'].get('correlation_threshold', 0.7), type=float)
    args.add_argument('--pattern_threshold', default=config['hypergraph'].get('pattern_threshold', 0.8), type=float)
    args.add_argument('--pattern_window', default=config['hypergraph'].get('pattern_window', 5), type=int)
else:
    args.add_argument('--use_hypergraph', default=True, type=bool)
    args.add_argument('--hyperedge_types', default='pick_drop,geo,temporal,correlation,pattern', type=str)

#train
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--weight_decay', default=config['train']['weight_decay'], type=eval)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')

#test
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)

#log
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)

args = args.parse_args()
init_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

#load dataset
train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args, 
    normalizer=args.normalizer, tod=args.tod, dow=False, weather=False, single=True)

#load distance matrix if available
distance_matrix = load_distance_matrix(args.dataset)

#init model
model = Network(args)

# Explicitly move model to GPU before any computation
print(f"Moving model to device: {args.device}")
model = model.to(args.device)

# Verify model is on correct device
print(f"Model device check: {next(model.parameters()).device}")

# Force model to use GPU by warming up
if torch.cuda.is_available() and 'cuda' in args.device:
    print("Warming up GPU...")
    with torch.no_grad():
        dummy_input = torch.randn(1, 12, args.num_nodes, args.input_dim).to(args.device)
        dummy_target = torch.randn(1, 1, args.num_nodes, args.input_dim).to(args.device)
        _ = model(dummy_input, dummy_target)
        torch.cuda.synchronize()
    print("GPU warmup completed")

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
print_model_parameters(model, only_num=False)

#init loss function, optimizer
if args.loss_func == 'mae':
    loss = masked_mae_loss(scaler, mask_value=0.0)
elif args.loss_func == 'mse':
    loss = nn.MSELoss()
else:
    raise ValueError

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=args.weight_decay, amsgrad=False)

#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)

#config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'experiments', args.dataset, current_time)
args.log_dir = log_dir

#start training
trainer = Trainer(model, loss, optimizer, train_dataloader, val_dataloader, test_dataloader, scaler,
                  args, lr_scheduler=lr_scheduler)
if args.mode == 'train':
    trainer.train()
    trainer.test(-1, 'test')
elif args.mode == 'test':
    model.load_state_dict(torch.load('./pre-trained/' + args.dataset + '.pth'))
    print("load saved model")
    trainer.test(-1, 'test')
else:
    raise ValueError
