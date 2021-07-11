from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import pickle
import time

import optimizers
import torch
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics
import scipy.sparse as sp



import argparse
import copy
import numpy as np
import os
# import psutil
import random
import tensorflow as tf
from time import time
from tqdm import tqdm
try:
    from tensorflow.python.ops.nn_ops import leaky_relu
except ImportError:
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import math_ops
 
 
    def leaky_relu(features, alpha=0.2, name=None):
        with ops.name_scope(name, "LeakyRelu", [features, alpha]):
            features = ops.convert_to_tensor(features, name="features")
            alpha = ops.convert_to_tensor(alpha, name="alpha")
            return math_ops.maximum(alpha * features, features)

from load_data import load_EOD_data, load_relation_data
from evaluator import evaluate

seed = 123456789
np.random.seed(seed)
tf.set_random_seed(seed)

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
import torch.optim as optim
device = 'cuda'
import pandas as pd



def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)  ##EDIT HERE make it div by trch.sum(mask)

def trr_loss_mse_rank(pred, base_price, ground_truth, mask, alpha, no_stocks):
    return_ratio = torch.div((pred- base_price), base_price)
    reg_loss = weighted_mse_loss(return_ratio, ground_truth, mask)
    all_ones = torch.ones(no_stocks,1).to(device)
    pre_pw_dif =  (torch.matmul(return_ratio, torch.transpose(all_ones, 0, 1)) 
                    - torch.matmul(all_ones, torch.transpose(return_ratio, 0, 1)))
    gt_pw_dif = (
            torch.matmul(all_ones, torch.transpose(ground_truth,0,1)) -
            torch.matmul(ground_truth, torch.transpose(all_ones, 0,1))
        )

    mask_pw = torch.matmul(mask, torch.transpose(mask, 0,1))
    rank_loss = torch.mean(
            F.relu(
                ((pre_pw_dif*gt_pw_dif)*mask_pw)))
    loss = reg_loss + alpha*rank_loss
    # print(return_ratio)
    del mask_pw, gt_pw_dif, pre_pw_dif, all_ones
    return loss, reg_loss, rank_loss, return_ratio


class ReRaLSTM:
    def __init__(self, data_path, market_name, tickers_fname, relation_name,
                 emb_fname, parameters, steps=1, epochs=50, batch_size=None, flat=False, gpu=False, in_pro=False):

        seed = 123456789
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        self.relation_name = relation_name
        # load data
        self.tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                                     dtype=str, delimiter='\t', skip_header=False)

        print('#tickers selected:', len(self.tickers))
        self.eod_data, self.mask_data, self.gt_data, self.price_data = \
            load_EOD_data(data_path, market_name, self.tickers, steps)

        
        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        self.flat = flat
        self.inner_prod = in_pro
        if batch_size is None:
            self.batch_size = len(self.tickers)  ##always,
        else:
            self.batch_size = batch_size

        self.valid_index = 756
        self.test_index = 1008
        self.trade_dates = self.mask_data.shape[1]
        self.fea_dim = 5

        self.gpu = gpu

    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.parameters['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        return self.eod_data[:, offset:offset + seq_len, :], \
               np.expand_dims(mask_batch, axis=1), \
               np.expand_dims(
                   self.price_data[:, offset + seq_len - 1], axis=1
               ), \
               np.expand_dims(
                   self.gt_data[:, offset + seq_len + self.steps - 1], axis=1
               )

    def train(self):

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if int(args.double_precision):
            torch.set_default_dtype(torch.float64)
        if int(args.cuda) >= 0:
            torch.cuda.manual_seed(args.seed)
        args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
        args.patience = args.epochs if not args.patience else  int(args.patience)
        logging.getLogger().setLevel(logging.INFO)
        if args.save:
            if not args.save_dir:
                dt = datetime.datetime.now()
                date = f"{dt.year}_{dt.month}_{dt.day}"
                models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
                save_dir = get_dir_name(models_dir)
            else:
                save_dir = args.save_dir
            logging.basicConfig(level=logging.INFO,
                                handlers=[
                                    logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                    logging.StreamHandler()
                                ])

        logging.info(f'Using: {args.device}')
        logging.info("Using seed {}.".format(args.seed))

        # Load data

        args.n_nodes = self.batch_size 
        args.feat_dim = 5
        Model = NCModel  
        args.n_classes = 1
        model = Model(args)
        model = model.to(args.device)
        optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

        if not args.lr_reduce_freq:
            args.lr_reduce_freq = args.epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.lr_reduce_freq),
            gamma=float(args.gamma)
        )
        tot_params = sum([np.prod(p.size()) for p in model.parameters()])
        logging.info(f"Total number of parameters: {tot_params}")
        
        adj = np.load('graph.npy')
        adj = sp.coo_matrix(adj, dtype=np.float32)
        adj = normalize(adj+sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)
        for i in range(self.epochs):
            t1 = time()
            np.random.shuffle(batch_offsets)
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0
            lr_scheduler.step()
            print(lr_scheduler.get_lr())
            model.train() 
            for j in tqdm(range(self.valid_index - self.parameters['seq'] -self.steps +1)):
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    batch_offsets[j])
                
                optimizer.zero_grad()
                embeddings = model.encode(torch.FloatTensor(emb_batch).to(device), adj.to(device))
                train_metrics = model.compute_metrics(embeddings,adj,torch.FloatTensor(price_batch).to(device), 
                                                                                        torch.FloatTensor(gt_batch).to(device), 
                                                                                        torch.FloatTensor(mask_batch).to(device), 
                                                                                        self.parameters['alpha'], self.batch_size)


            

                train_metrics['loss'].backward()
                optimizer.step()
            
                tra_loss += train_metrics['loss'].detach().cpu().item()
                tra_reg_loss += train_metrics['reg_loss'].detach().cpu().item()
                tra_rank_loss += train_metrics['rank_loss'].detach().cpu().item()
            
            print('Train Loss:',
                  tra_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1),
                  tra_reg_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1),
                  tra_rank_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1))
            
            with torch.no_grad():
                cur_valid_pred = np.zeros(
                    [len(self.tickers), self.test_index - self.valid_index],
                    dtype=float
                )
                cur_valid_gt = np.zeros(
                    [len(self.tickers), self.test_index - self.valid_index],
                    dtype=float
                )
                cur_valid_mask = np.zeros(
                    [len(self.tickers), self.test_index - self.valid_index],
                    dtype=float
                )
                val_loss = 0.0
                val_reg_loss = 0.0
                val_rank_loss = 0.0
                model.eval()
                for cur_offset in range(
                    self.valid_index - self.parameters['seq'] - self.steps + 1,
                    self.test_index - self.parameters['seq'] - self.steps + 1
                ):
                    emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                        cur_offset)
                
                    embeddings = model.encode(torch.FloatTensor(emb_batch).to(device), adj.to(device))
                    val_metrics = model.compute_metrics(embeddings,adj,torch.FloatTensor(price_batch).to(device), 
                                                                                        torch.FloatTensor(gt_batch).to(device), 
                                                                                        torch.FloatTensor(mask_batch).to(device), 
                                                                                        self.parameters['alpha'], self.batch_size)
                    
                    
            
                    cur_rr = val_metrics['rr'].detach().cpu().numpy().reshape((1026,1))
                    val_loss += val_metrics['loss'].detach().cpu().item()
                    val_reg_loss += val_metrics['reg_loss'].detach().cpu().item()
                    val_rank_loss += val_metrics['rank_loss'].detach().cpu().item()
                    cur_valid_pred[:, cur_offset - (self.valid_index -
                                                    self.parameters['seq'] -
                                                    self.steps + 1)] = \
                        copy.copy(cur_rr[:, 0])
                    cur_valid_gt[:, cur_offset - (self.valid_index -
                                                self.parameters['seq'] -
                                                self.steps + 1)] = \
                        copy.copy(gt_batch[:, 0])
                    cur_valid_mask[:, cur_offset - (self.valid_index -
                                                    self.parameters['seq'] -
                                                    self.steps + 1)] = \
                        copy.copy(mask_batch[:, 0])
                
                cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt,
                                        cur_valid_mask)
                print('\t Valid preformance:', cur_valid_perf)

                # test on testing set
                cur_test_pred = np.zeros(
                    [len(self.tickers), self.trade_dates - self.test_index],
                    dtype=float
                )
                cur_test_gt = np.zeros(
                    [len(self.tickers), self.trade_dates - self.test_index],
                    dtype=float
                )
                cur_test_mask = np.zeros(
                    [len(self.tickers), self.trade_dates - self.test_index],
                    dtype=float
                )
                test_loss = 0.0
                test_reg_loss = 0.0
                test_rank_loss = 0.0
                model.eval()
                for cur_offset in range(self.test_index - self.parameters['seq'] - self.steps + 1,
                                                self.trade_dates - self.parameters['seq'] - self.steps + 1):
                    emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(cur_offset)
                    
                    embeddings = model.encode(torch.FloatTensor(emb_batch).to(device), adj.to(device))
                    test_metrics = model.compute_metrics(embeddings,adj,torch.FloatTensor(price_batch).to(device), 
                                                                                        torch.FloatTensor(gt_batch).to(device), 
                                                                                        torch.FloatTensor(mask_batch).to(device), 
                                                                                        self.parameters['alpha'], self.batch_size)
                    cur_rr = test_metrics['rr'].detach().cpu().numpy().reshape((1026,1))
                    test_loss += test_metrics['loss'].detach().cpu().item()
                    test_reg_loss += test_metrics['reg_loss'].detach().cpu().item()
                    test_rank_loss += test_metrics['rank_loss'].detach().cpu().item()

                    cur_test_pred[:, cur_offset - (self.test_index -
                                                self.parameters['seq'] -
                                                self.steps + 1)] = \
                        copy.copy(cur_rr[:, 0])
                    cur_test_gt[:, cur_offset - (self.test_index -
                                                self.parameters['seq'] -
                                                self.steps + 1)] = \
                        copy.copy(gt_batch[:, 0])
                    cur_test_mask[:, cur_offset - (self.test_index -
                                                self.parameters['seq'] -
                                                self.steps + 1)] = \
                        copy.copy(mask_batch[:, 0])
                cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
                print('\t Test performance:', cur_test_perf)
       
    def update_model(self, parameters):
        for name, value in parameters.items():
            self.parameters[name] = value
        return True


if __name__ == '__main__':
    args = parser.parse_args()
    if args.t is None:
        args.t = args.m + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    args.gpu = (args.gpu == 1)

    args.inner_prod = (args.inner_prod == 1)
    lr_li = [8e-4, 5e-4, 1e-3 ,1e-4,]
    alpha_li = [10,9,8,7,6,5,4,3,2,1]
    for i in lr_li:
        for j in alpha_li:
            args.lr = i
            args.a = j
            parameters = {'seq': int(args.l), 'unit': int(args.u), 'lr': float(args.r),
                        'alpha': float(args.a)}
            print('arguments:', args)
            print('parameters:', parameters)

            RR_LSTM = ReRaLSTM(
                data_path=args.p,
                market_name=args.m,
                tickers_fname=args.t,
                relation_name=args.rel_name,
                emb_fname=args.emb_file,
                parameters=parameters,
                steps=1, epochs=1, batch_size=None, gpu=args.gpu,
                in_pro=args.inner_prod
            )

            pred_all = RR_LSTM.train()