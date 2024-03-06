#!/usr/bin/python                                                                                                                                  
#-*-coding:utf-8-*- 
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Finetune:to do some downstream task
"""

import random
from pathlib import Path
import os
from os.path import join, exists, basename
import argparse
import numpy as np
import pandas as pd
import glob
import time
import torch
import pandas as pd
import json

import paddle
import paddle.nn as nn
import pgl
import subprocess

from src.model import GeoGNNModel
from src.utils import load_json_config
from dataset import InMemoryDataset

from src.model import DownstreamModel
from src.featurizer import DownstreamTransformFn, DownstreamCollateFn
from src.utils import get_dataset, create_splitter, get_downstream_task_names, get_dataset_stat, \
        calc_rocauc_score, calc_rmse, calc_mae, exempt_parameters
import sys
sys.path.append('..')
from torch.utils.data import ConcatDataset

from plot import calculate_resutls, cm_plot, create_logger
import warnings
warnings.filterwarnings("ignore")

def train(
        args, 
        model, label_mean, label_std,
        train_dataset, collate_fn, 
        criterion, encoder_opt, head_opt):
    """
    Define the train function 
    Args:
        args,model,train_dataset,collate_fn,criterion,encoder_opt,head_opt;
    Returns:
        the average of the list loss
    """
    data_gen = train_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True,
            collate_fn=collate_fn)
    list_loss = []
    model.train()
    for atom_bond_graphs, bond_angle_graphs, labels in data_gen:
        if len(labels) < args.batch_size * 0.5:
            continue
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        scaled_labels = (labels - label_mean) / (label_std + 1e-5)
        scaled_labels = paddle.to_tensor(scaled_labels, 'float32')
        preds = model(atom_bond_graphs, bond_angle_graphs)
        loss = criterion(preds, scaled_labels)
        loss.backward()
        encoder_opt.step()
        head_opt.step()
        encoder_opt.clear_grad()
        head_opt.clear_grad()
        list_loss.append(loss.numpy())
    return np.mean(list_loss)


def no_tune_train(
        args,
        model, label_mean, label_std,
        train_dataset, collate_fn, 
        criterion, encoder_opt, head_opt):
    data_gen_list = []
    data_gen_list.append(train_dataset[0].get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True,
            collate_fn=collate_fn))
    data_gen_list.append(train_dataset[1].get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True,
            collate_fn=collate_fn))
    list_loss = []
    model.train()
    for data_gen in data_gen_list:
        for atom_bond_graphs, bond_angle_graphs, labels in data_gen:
            if len(labels) < args.batch_size * 0.5:
                continue
            atom_bond_graphs = atom_bond_graphs.tensor()
            bond_angle_graphs = bond_angle_graphs.tensor()
            scaled_labels = (labels - label_mean) / (label_std + 1e-5)
            scaled_labels = paddle.to_tensor(scaled_labels, 'float32')
            preds = model(atom_bond_graphs, bond_angle_graphs)
            loss = criterion(preds, scaled_labels)
            loss.backward()
            encoder_opt.step()
            head_opt.step()
            encoder_opt.clear_grad()
            head_opt.clear_grad()
            list_loss.append(loss.numpy())
    return np.mean(list_loss)


def evaluate(
        args, 
        model, label_mean, label_std,
        test_dataset, collate_fn, metric, epoch_id):
    """
    Define the evaluate function
    In the dataset, a proportion of labels are blank. So we use a `valid` tensor 
    to help eliminate these blank labels in both training and evaluation phase.
    """
    data_gen = test_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=False,
            collate_fn=collate_fn)
    total_pred = []
    total_label = []
    model.eval()
    for atom_bond_graphs, bond_angle_graphs, labels in data_gen:
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        labels = paddle.to_tensor(labels, 'float32')
        scaled_preds = model(atom_bond_graphs, bond_angle_graphs)
        preds = scaled_preds.numpy() * label_std + label_mean
        total_pred.append(preds)
        total_label.append(labels.numpy())
    total_pred = np.concatenate(total_pred, 0)
    total_label = np.concatenate(total_label, 0)

    os.makedirs(os.path.join(args.model_dir, f'epoch{epoch_id}'), exist_ok=True)
    np.save(os.path.join(args.model_dir, f'epoch{epoch_id}', 'test_pred.npy'), total_pred)
    np.save(os.path.join(args.model_dir, f'epoch{epoch_id}', 'test_labels.npy'), total_label)

    if metric == 'rmse':
        return calc_rmse(total_label, total_pred)
    else:
        return calc_mae(total_label, total_pred)


def get_label_stat(dataset):
    """tbd"""
    labels = np.array([data['label'] for data in dataset])
    return np.min(labels), np.max(labels), np.mean(labels)


def get_metric(dataset_name):
    """tbd"""
    if dataset_name in ['esol', 'freesolv', 'lipophilicity', 'tk10']:
        return 'rmse'
    elif dataset_name in ['qm7', 'qm8', 'qm9', 'qm9_gdb']:
        return 'mae'
    else:
        return 'rmse'
        # raise ValueError(dataset_name)


def main(args):
    """
    Call the configuration function of the model, build the model and load data, then start training.
    model_config:
        a json file  with the hyperparameters,such as dropout rate ,learning rate,num tasks and so on;
    num_tasks:
        it means the number of task that each dataset contains, it's related to the dataset;
    """
    timing = {'start_time': time.time(),'train_time': 0, 'val_time': 0, 'test_time': 0}
    logger = create_logger(args.model_dir)
    logger.info(args)

    ### config for the body
    compound_encoder_config = load_json_config(args.compound_encoder_config)
    if not args.dropout_rate is None:
        compound_encoder_config['dropout_rate'] = args.dropout_rate

    ### config for the downstream task
    task_type = 'regr'
    metric = get_metric(args. cell_line)
    task_names = get_downstream_task_names(args. cell_line, args.data_path)
    dataset_stat = get_dataset_stat(args. cell_line, args.data_path, task_names)
    label_mean = np.reshape(dataset_stat['mean'], [1, -1])
    label_std = np.reshape(dataset_stat['std'], [1, -1])

    model_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        model_config['dropout_rate'] = args.dropout_rate
    model_config['task_type'] = task_type
    model_config['num_tasks'] = len(task_names)
    logger.info('model_config:')
    logger.info(model_config)

    ### build model
    compound_encoder = GeoGNNModel(compound_encoder_config)
    model = DownstreamModel(model_config, compound_encoder)
    if metric == 'square':
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()
    encoder_params = compound_encoder.parameters()
    head_params = exempt_parameters(model.parameters(), encoder_params)
    encoder_opt = paddle.optimizer.Adam(args.encoder_lr, parameters=encoder_params)
    head_opt = paddle.optimizer.Adam(args.head_lr, parameters=head_params)
    logger.info(model)
    logger.info('Total param num: %s' % (len(model.parameters())))
    logger.info('Encoder param num: %s' % (len(encoder_params)))
    logger.info('Head param num: %s' % (len(head_params)))
    # for i, param in enumerate(model.named_parameters()):
    #     logger.info(i, param[0], param[1].name)

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info(f"number of params: {n_parameters}")    
    # import pdb
    # pdb.set_trace()

    if not args.init_model is None and not args.init_model == "":
        compound_encoder.set_state_dict(paddle.load(args.init_model))
        logger.info('Load state_dict from %s' % args.init_model)

    ### load data    
    # if args.task == 'data':
    dataset_path = os.path.join(args.cached_data_path, f'{args.split_type}_fold_{args.test_fold}.pt')
    if not os.path.exists(dataset_path):
        logger.info('Preprocessing data...')
        dataset = get_dataset(args. cell_line, args.data_path, task_names)
        transform_fn = DownstreamTransformFn()
        dataset.transform(transform_fn, num_workers=args.num_workers)
        # dataset.save_data(args.cached_data_path)
        torch.save(dataset, dataset_path)
    else:
        logger.info('Loading data...')
        dataset = torch.load(dataset_path)

    if args.split_type == 'LDMO':
        clusters = pd.read_csv(args.clustering_id_path)[['SMILES','Cluster_ID']]
        test_smiles_list = list(clusters[clusters['Cluster_ID'] == args.test_fold]['SMILES'].values)
        val_fold = 7 if args.test_fold == 1 else args.test_fold - 1
        val_smiles_list = list(clusters[clusters['Cluster_ID'] == val_fold]['SMILES'].values)
    elif args.split_type == 'scaffold':
        clusters = pd.read_csv(args.clustering_id_path)[['SMILES','Scaffold_Cluster_ID']]
        test_smiles_list = list(clusters[clusters['Scaffold_Cluster_ID'] == args.test_fold]['SMILES'].values)
        val_fold = 7 if args.test_fold == 1 else args.test_fold - 1
        val_smiles_list = list(clusters[clusters['Scaffold_Cluster_ID'] == val_fold]['SMILES'].values)

    train_idx, valid_idx, test_idx = [], [], []
    for idx, data_instance in enumerate(dataset.data_list):
        if data_instance:
            if data_instance['smiles'] in test_smiles_list:
                test_idx.extend([idx])
            elif data_instance['smiles'] in val_smiles_list:
                valid_idx.extend([idx])
            else:
                train_idx.extend([idx])
            
    train_dataset, valid_dataset, test_dataset = dataset[train_idx], dataset[valid_idx], dataset[test_idx]
    logger.info(f'Total smiles = {len(train_dataset)+len(valid_dataset)+len(test_dataset)} | '
                f'train smiles = {len(train_dataset):,} | '
                f'val smiles = {len(valid_dataset):,} | '
                f'test smiles = {len(test_dataset):,}')
    logger.info('Train min/max/mean %s/%s/%s' % get_label_stat(train_dataset))
    logger.info('Valid min/max/mean %s/%s/%s' % get_label_stat(valid_dataset))
    logger.info('Test min/max/mean %s/%s/%s' % get_label_stat(test_dataset))

    
    ### start train
    list_val_metric, list_test_metric = [], []
    collate_fn = DownstreamCollateFn(
            atom_names=compound_encoder_config['atom_names'], 
            bond_names=compound_encoder_config['bond_names'],
            bond_float_names=compound_encoder_config['bond_float_names'],
            bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
            task_type=task_type)
    for epoch_id in range(args.max_epoch):
        if args.search != 'no_tune':
            tmp_time = time.time()
            train_loss = train(
                    args, model, label_mean, label_std, 
                    train_dataset, collate_fn, 
                    criterion, encoder_opt, head_opt)
            timing['train_time'] += time.time() - tmp_time
            logger.info("epoch:%s train/loss:%s" % (epoch_id, train_loss))
            tmp_time = time.time()
            val_metric = evaluate(
                    args, model, label_mean, label_std, 
                    valid_dataset, collate_fn, metric, epoch_id)
            timing['val_time'] += time.time() - tmp_time
            tmp_time = time.time()
            test_metric = evaluate(
                    args, model, label_mean, label_std, 
                    test_dataset, collate_fn, metric, epoch_id)
            timing['test_time'] += time.time() - tmp_time

            list_val_metric.append(val_metric)
            list_test_metric.append(test_metric)
            test_metric_by_eval = list_test_metric[np.argmin(list_val_metric)]
            logger.info("epoch:%s val/%s:%s" % (epoch_id, metric, val_metric))
            logger.info("epoch:%s test/%s:%s" % (epoch_id, metric, test_metric))
            logger.info("epoch:%s test/%s_by_eval:%s" % (epoch_id, metric, test_metric_by_eval))
        else:
            tmp_time = time.time()
            train_loss = no_tune_train(
                    args, model, label_mean, label_std, 
                    [train_dataset, valid_dataset], collate_fn, 
                    criterion, encoder_opt, head_opt)
            timing['train_time'] += time.time() - tmp_time
            logger.info("epoch:%s train/loss:%s" % (epoch_id, train_loss))
            test_metric = evaluate(
                    args, model, label_mean, label_std, 
                    test_dataset, collate_fn, metric, epoch_id)
            timing['test_time'] += time.time() - tmp_time
        paddle.save(compound_encoder.state_dict(), f"{args.model_dir}/epoch{epoch_id}/compound_encoder.pdparams")
        paddle.save(model.state_dict(), '%s/epoch%d/model.pdparams' % (args.model_dir, epoch_id))

    logger.info(f'Training time: {timing["train_time"]} | Val time: {timing["val_time"]} | Test time: {timing["test_time"]}')
    outs = {
        'model_config': basename(args.model_config).replace('.json', ''),
        'metric': '',
        'dataset': args. cell_line, 
        'split_type': args.split_type, 
        'batch_size': args.batch_size,
        'dropout_rate': args.dropout_rate,
        'encoder_lr': args.encoder_lr,
        'head_lr': args.head_lr,
    }
    if args.search != 'no_tune':
        best_epoch_id = np.argmin(list_val_metric)
        logger.info(f'Best epoch at {best_epoch_id}')
        for metric, value in [
                ('test_%s' % metric, list_test_metric[best_epoch_id]),
                ('max_valid_%s' % metric, np.min(list_val_metric)),
                ('max_test_%s' % metric, np.min(list_test_metric))]:
            outs['metric'] = metric
            logger.info('\t'.join(['FINAL'] + ["%s:%s" % (k, outs[k]) for k in outs] + [str(value)]))
    else:
        best_epoch_id = epoch_id
        
    ############## Calculate results ##################
    best_dir_path = os.path.join(args.model_dir, f'bestepoch{best_epoch_id}')
    os.makedirs(best_dir_path, exist_ok=True)
    y_pred = np.load(os.path.join(args.model_dir, f'epoch{best_epoch_id}', 'test_pred.npy'))
    y_test = np.load(os.path.join(args.model_dir, f'epoch{best_epoch_id}', 'test_labels.npy'))
    subprocess.call(f'cp {os.path.join(args.model_dir, f"epoch{best_epoch_id}", "test_pred.npy")} {os.path.join(args.model_dir, f"bestepoch{best_epoch_id}", "test_pred.npy")}', shell=True)
    subprocess.call(f'cp {os.path.join(args.model_dir, f"epoch{best_epoch_id}", "test_labels.npy")} {os.path.join(args.model_dir, f"bestepoch{best_epoch_id}", "test_labels.npy")}', shell=True)

    test_results = calculate_resutls(y_test, y_pred)
    cm_plot(os.path.join(best_dir_path, 'cm.png'), test_results, 'GEM', save=True)
    del test_results['y_test']
    del test_results['y_pred']
    del test_results['cm']
    test_results['time'] = time.time() - timing['start_time']
    test_results['train_time'] = timing['train_time']
    test_results['val_time'] = timing['val_time']
    test_results['test_time'] = timing['test_time']
    # test_results['val_rmse'] = np.min(list_val_metric)
    logger.info(f'Total training time {test_results["time"]:2f}s.')
    logger.info(f'Triaining time: {timing["train_time"]} | Validation time: {timing["val_time"]} | Test time: {timing["test_time"]}')    
    with open(os.path.join(best_dir_path, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)


def set_seed(seed):
    """
    Set seed for various random number generators.

    Parameters:
        seed (int): The seed value to set.
    """
    # Set seed for random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=['train', 'data'], default='train')
    parser.add_argument("--test_fold", type=int, default=7)
    parser.add_argument("--cell_line", type=str, default='TK-10')
    parser.add_argument("--search", type=str, default='grid')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clustering_id_path", type=str, default='../../data/clustering_id_k7.csv')

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_epoch", type=int, default=100)
    # parser.add_argument("--dataset_name", choices=['esol', 'freesolv', 'lipophilicity', 'qm7', 'qm8', 'qm9', 'qm9_gdb', 'tk10'], default='tk10')
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--cached_data_path", type=str, default=None)
    parser.add_argument("--split_type", choices=['random', 'scaffold', 'random_scaffold', 'index', 'LDMO'], default='LDMO')

    parser.add_argument("--compound_encoder_config", type=str, default='model_configs/geognn_l8.json')
    parser.add_argument("--model_config", type=str,default="model_configs/down_mlp2.json")
    parser.add_argument("--init_model", type=str,default="./pretrain_models-chemrl_gem/regr.pdparams")
    parser.add_argument("--model_dir", type=str,default=None)
    parser.add_argument("--encoder_lr", type=float, default=0.001)
    parser.add_argument("--head_lr", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    args = parser.parse_args()
    
    set_seed(args.seed)
    root_path = Path(os.getcwd()).parent.parent
    args.cached_data_path=f"{root_path}/data/GEM/{args.cell_line}"
    os.makedirs(args.cached_data_path, exist_ok=True)
    # args.data_path=f"./chemrl_downstream_datasets/{args.dataset_name}"
    args.data_path=f'{root_path}/data/60_cell_lines/{args.cell_line}.csv'
    
    args.model_dir=f"{root_path}/results/GEM/{args.cell_line}/fold_{args.test_fold}/seed_{args.seed}/{args.split_type}_{args.batch_size}_{args.encoder_lr}_{args.head_lr}_{args.dropout_rate}"

    if not os.path.isfile(f'{args.model_dir}/epoch99/test_pred.npy'):
        main(args)
    else:
        print(f'{args.model_dir}/epoch99/test_pred.npy esxits, skiping...')