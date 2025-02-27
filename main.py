from utils.utils import gp_prediction, sigma_f, split_1d_tensor
from FLAlgorithms.servers.serverFedNPP import FedNPP
from utils.dataset_load import load_dataset
import numpy as np
import torch, time
import torch.nn as nn
import argparse
from embeddings import *
from hist_encoder import *
from models.wrapper import wrapper
from trainers.trainer import Trainer

# data, _ = torch.sort(torch.rand(550)*100)
# T = torch.tensor([torch.max(data)]) + 0.1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Federated Nonparametric Point Processes",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model Type FedNPP
    parser.add_argument('--Kernel_DPM_Type', type=str, default='OneLayerMLP', help = 'OneLayerMLP or None')
    parser.add_argument('--is_bayesian_paramter', type=int, default=1, help = 'Is it a baysian paramter in FedPP?')
    parser.add_argument('--aggerate_method', type=str, default="FedPP", choices = ['FedAvg', 'AggSigma', 'AggSigma2', 'FedPP'])

    # federated learning setting
    parser.add_argument('--local_only', type=int, default=0, help = 'local_only')
    parser.add_argument('--mix_type_num', type=int, default=2, help = 'the number of mixed type at one client')
    parser.add_argument('--select_period', type=int, default=1, help = 'select_period')
    parser.add_argument('--seleted_clent_data_ratio', type=float, default=1, help = 'seleted_clent_data_ratio')

    # Training
    parser.add_argument('--max_epoch', type=int, metavar='NUM', default=20,
                        help='The maximum epoch number for training.')
    parser.add_argument('--local_epoch', type=int, metavar='NUM', default=10,
                        help='The local epoch number for training at each client.')
    parser.add_argument('--clients_num', type=int, default=20,
                        help='The number of clients.')
    parser.add_argument('--time_emb', type=str, metavar='NAME', default='Trigo')
    parser.add_argument('--dataset_type', type=str, default='Synthetic', help='Synthetic or taobao')
    parser.add_argument('--dataset_only_1type', type=int, default=0)
    parser.add_argument('--embed_size', type=int, metavar='SIZE', default=32,
                        help='Hidden dimension for the model.')

    parser.add_argument('--sampling_data', type=int, default=0, help='if dataset is synthetic, decide sampling or not')
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-2)

    args = parser.parse_args()
    args = vars(args)
    args['event_type_num'] = int(2)
    save_data = load_dataset(args)
    server = FedNPP(save_data, args)
    print(server.train())
    if (args['mix_type_num'] == 1):
        server.plt_result_figure()
