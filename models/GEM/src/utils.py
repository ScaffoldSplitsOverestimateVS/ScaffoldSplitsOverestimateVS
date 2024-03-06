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
utils
"""

from __future__ import print_function
import sys
import os
from os.path import exists, dirname
import numpy as np
import pickle
import json
import time
import six
if six.PY3:
    import _thread as thread
    from queue import Queue
else:
    import thread
    from Queue import Queue
from collections import OrderedDict
from datetime import datetime
from sklearn.metrics import roc_auc_score

from paddle import fluid

# from pahelix.utils.splitters import \
#     RandomSplitter, IndexSplitter, ScaffoldSplitter, RandomScaffoldSplitter
# from pahelix.datasets import *
# from pahelix.datasets.qm9_gdb_dataset import *
# from pahelix.datasets.tk10_dataset import *


def get_downstream_task_names(dataset_name, data_path):
    """
    Get task names of downstream dataset
    """
    if dataset_name == 'bace':
        task_name = get_default_bace_task_names()
    elif dataset_name == 'bbbp':
        task_name = get_default_bbbp_task_names()
    elif dataset_name == 'clintox':
        task_name = get_default_clintox_task_names() 
    elif dataset_name == 'hiv':
        task_name = get_default_hiv_task_names() 
    elif dataset_name == 'muv':
        task_name = get_default_muv_task_names() 
    elif dataset_name == 'sider':
        task_name = get_default_sider_task_names()
    elif dataset_name == 'tox21':
        task_name = get_default_tox21_task_names()
    elif dataset_name == 'toxcast':
        task_name = get_default_toxcast_task_names(data_path)
    elif dataset_name == 'esol':
        return get_default_esol_task_names()
    elif dataset_name == 'freesolv':
        return get_default_freesolv_task_names()
    elif dataset_name == 'lipophilicity':
        return get_default_lipophilicity_task_names()
    elif dataset_name == 'qm7':
        return get_default_qm7_task_names()
    elif dataset_name == 'qm8':
        return get_default_qm8_task_names()
    elif dataset_name == 'qm9':
        return get_default_qm9_task_names()  
    elif dataset_name == 'qm9_gdb':
        return get_default_qm9_gdb_task_names()  
    elif dataset_name == 'tk10':
        return ['NLOGGI50_N']
    else:
        return ['NLOGGI50_N']
        raise ValueError('%s not supported' % dataset_name)

    return task_name

def load_tk10_dataset(data_path, task_names=None):
    import pandas as pd
    from dataset import InMemoryDataset
    if task_names is None:
        task_names = ['NLOGGI50_N']

    # raw_path = os.path.join(data_path, 'raw')
    # csv_file = os.listdir(raw_path)[0]
    # input_df = pd.read_csv(os.path.join(raw_path, csv_file), sep=',')
    input_df = pd.read_csv(data_path)
    clusters = pd.read_csv('../../data/clustering_id_k7.csv')[['NSC', 'Cluster_ID']]
    input_df = clusters.merge(input_df, how='inner', on='NSC')
    # input_df = input_df[input_df['SMILES'] != 'O=[N+]([O-])c1ccc(Cl)c2c([N+](=O)[O-])cccc12'].reset_index(drop=True)
    # input_df = input_df[input_df['SMILES'] != 'O=[N+]([O-])c1cccc2nc3ccccc3c(NCCCCNc3c4ccccc4nc4cccc([N+](=O)[O-])c34)c12'].reset_index(drop=True)
    # input_df = input_df[input_df['SMILES'] != 'COC(=O)[C@H](Cc1ccccc1)NC(=O)c1ccc(CNC(=O)/C=C/c2ccc(-c3ccccc3)cc2)cc1'].reset_index(drop=True)
    # input_df = input_df[input_df['SMILES'] != 'O=C(/C=C/c1ccc(-c2ccccc2)cc1)NCc1ccc(C(=O)NCCc2ccccc2)cc1'].reset_index(drop=True)
    # input_df = input_df[input_df['SMILES'] != 'CC(C)(C(=O)N(C(=O)N1CCN(c2ccccc2)CC1)S(=O)(=O)c1ccccc1)c1ccc(Cl)cc1'].reset_index(drop=True)
    # input_df = input_df[input_df['SMILES'] != 'Brc1ccc2c(c1)[C@H]1CCc3c([nH]c4ccccc34)[C@H]1N2'].reset_index(drop=True)
    # input_df = input_df[input_df['SMILES'] != 'Cc1nc2c(o1)C(=O)c1ccccc1/C2=N/c1ccccn1'].reset_index(drop=True)
    # input_df = input_df[input_df['SMILES'] != 'CC(=O)O[C@H]1CC[C@@H]2[C@@H]3CC[C@]4(C)CC(=O)C=C[C@]4(C)[C@@H]3CC[C@]12C'].reset_index(drop=True)
    # input_df = input_df[input_df['SMILES'] != 'C=C(CN1CCN(c2cc(C(F)(F)F)cc(C(F)(F)F)c2)CC1)c1ccc(C(=O)Nc2ccccc2N)cc1'].reset_index(drop=True)
    # input_df = input_df[input_df['SMILES'] != 'C=C(CN1CCN(c2ccc(C(F)(F)F)cc2)CC1)c1ccc(C(=O)Nc2ccccc2N)cc1'].reset_index(drop=True)

    smiles_list = input_df['SMILES'].values
    labels = input_df[task_names]

    data_list = []
    for i in range(len(labels)):
        data = {
            'smiles': smiles_list[i],
            'label': labels.values[i],
        }
        data_list.append(data)
    dataset = InMemoryDataset(data_list)
    return dataset



def get_dataset(dataset_name, data_path, task_names):
    """Return dataset according to the ``dataset_name``"""
    if dataset_name == 'bace':
        dataset = load_bace_dataset(data_path, task_names)
    elif dataset_name == 'bbbp':
        dataset = load_bbbp_dataset(data_path, task_names)
    elif dataset_name == 'clintox':
        dataset = load_clintox_dataset(data_path, task_names)
    elif dataset_name == 'hiv':
        dataset = load_hiv_dataset(data_path, task_names)
    elif dataset_name == 'muv':
        dataset = load_muv_dataset(data_path, task_names)
    elif dataset_name == 'sider':
        dataset = load_sider_dataset(data_path, task_names)
    elif dataset_name == 'tox21':
        dataset = load_tox21_dataset(data_path, task_names)
    elif dataset_name == 'toxcast':
        dataset = load_toxcast_dataset(data_path, task_names)
    elif dataset_name == 'esol':
        dataset = load_esol_dataset(data_path, task_names)
    elif dataset_name == 'freesolv':
        dataset = load_freesolv_dataset(data_path, task_names)
    elif dataset_name == 'lipophilicity':
        dataset = load_lipophilicity_dataset(data_path, task_names)
    elif dataset_name == 'qm7':
        dataset = load_qm7_dataset(data_path, task_names)
    elif dataset_name == 'qm8':
        dataset = load_qm8_dataset(data_path, task_names)
    elif dataset_name == 'qm9':
        dataset = load_qm9_dataset(data_path, task_names)
    elif dataset_name == 'qm9_gdb':
        dataset = load_qm9_gdb_dataset(data_path, task_names)
    elif dataset_name == 'tk10':
        dataset = load_tk10_dataset(data_path, task_names)
    else:
        dataset = load_tk10_dataset(data_path, task_names)
        # raise ValueError('%s not supported' % dataset_name)

    return dataset


def get_tk10_stat(data_path, task_names):
    import pandas as pd
    """Return mean and std of labels"""
    label_df = pd.read_csv(data_path)
    labels = label_df[task_names].values
    return {
        'mean': np.mean(labels, 0),
        'std': np.std(labels, 0),
        'N': len(labels),
    }


def get_dataset_stat(dataset_name, data_path, task_names):
    """tbd"""
    if dataset_name == 'esol':
        return get_esol_stat(data_path, task_names)
    elif dataset_name == 'freesolv':
        return get_freesolv_stat(data_path, task_names)
    elif dataset_name == 'lipophilicity':
        return get_lipophilicity_stat(data_path, task_names)
    elif dataset_name == 'qm7':
        return get_qm7_stat(data_path, task_names)
    elif dataset_name == 'qm8':
        return get_qm8_stat(data_path, task_names)
    elif dataset_name == 'qm9':
        return get_qm9_stat(data_path, task_names)
    elif dataset_name == 'qm9_gdb':
        return get_qm9_gdb_stat(data_path, task_names)
    elif dataset_name == 'tk10':
        return get_tk10_stat(data_path, task_names)
    else:
        import pandas as pd
        """Return mean and std of labels"""
        label_df = pd.read_csv(data_path)
        labels = label_df[task_names].values
        return {
            'mean': np.mean(labels, 0),
            'std': np.std(labels, 0),
            'N': len(labels),
        }
        raise ValueError(dataset_name)


def create_splitter(split_type):
    """Return a splitter according to the ``split_type``"""
    if split_type == 'random':
        splitter = RandomSplitter()
    elif split_type == 'index':
        splitter = IndexSplitter()
    elif split_type == 'scaffold':
        splitter = ScaffoldSplitter()
    elif split_type == 'random_scaffold':
        splitter = RandomScaffoldSplitter()
    else:
        raise ValueError('%s not supported' % split_type)
    return splitter


def calc_rocauc_score(labels, preds, valid):
    """compute ROC-AUC and averaged across tasks"""
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
        preds = preds.reshape(-1, 1)

    rocauc_list = []
    for i in range(labels.shape[1]):
        c_valid = valid[:, i].astype("bool")
        c_label, c_pred = labels[c_valid, i], preds[c_valid, i]
        #AUC is only defined when there is at least one positive data.
        if len(np.unique(c_label)) == 2:
            rocauc_list.append(roc_auc_score(c_label, c_pred))

    print('Valid ratio: %s' % (np.mean(valid)))
    print('Task evaluated: %s/%s' % (len(rocauc_list), labels.shape[1]))
    if len(rocauc_list) == 0:
        raise RuntimeError("No positively labeled data available. Cannot compute ROC-AUC.")

    return sum(rocauc_list)/len(rocauc_list)


def calc_rmse(labels, preds):
    """tbd"""
    return np.sqrt(np.mean((preds - labels) ** 2))


def calc_mae(labels, preds):
    """tbd"""
    return np.mean(np.abs(preds - labels))


def exempt_parameters(src_list, ref_list):
    """Remove element from src_list that is in ref_list"""
    res = []
    for x in src_list:
        flag = True
        for y in ref_list:
            if x is y:
                flag = False
                break
        if flag:
            res.append(x)
    return res


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
| Blocks for Graph Neural Network (GNN)
| Adapted from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/model.py
"""


import paddle
import paddle.nn as nn
import pgl
import json


class GraphNorm(nn.Layer):
    """Implementation of graph normalization. Each node features is divied by sqrt(num_nodes) per graphs.
    
    Args:
        graph: the graph object from (:code:`Graph`)
        feature: A tensor with shape (num_nodes, feature_size).

    Return:
        A tensor with shape (num_nodes, hidden_size)

    References:

    [1] BENCHMARKING GRAPH NEURAL NETWORKS. https://arxiv.org/abs/2003.00982

    """

    def __init__(self):
        super(GraphNorm, self).__init__()
        self.graph_pool = pgl.nn.GraphPool(pool_type="sum")

    def forward(self, graph, feature):
        """graph norm"""
        nodes = paddle.ones(shape=[graph.num_nodes, 1], dtype="float32")
        norm = self.graph_pool(graph, nodes)
        norm = paddle.sqrt(norm)
        norm = paddle.gather(norm, graph.graph_node_id)
        return feature / norm


class MeanPool(nn.Layer):
    """
    TODO: temporary class due to pgl mean pooling
    """
    def __init__(self):
        super().__init__()
        self.graph_pool = pgl.nn.GraphPool(pool_type="sum")

    def forward(self, graph, node_feat):
        """
        mean pooling
        """
        sum_pooled = self.graph_pool(graph, node_feat)
        ones_sum_pooled = self.graph_pool(
            graph,
            paddle.ones_like(node_feat, dtype="float32"))
        pooled = sum_pooled / ones_sum_pooled
        return pooled


class GIN(nn.Layer):
    """
    Implementation of Graph Isomorphism Network (GIN) layer with edge features
    """
    def __init__(self, hidden_size):
        super(GIN, self).__init__()
        self.mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size))

    def forward(self, graph, node_feat, edge_feat):
        """
        Args:
            node_feat(tensor): node features with shape (num_nodes, feature_size).
            edge_feat(tensor): edges features with shape (num_edges, feature_size).
        """
        def _send_func(src_feat, dst_feat, edge_feat):
            x = src_feat['h'] + edge_feat['h']
            return {'h': x}

        def _recv_func(msg):
            return msg.reduce_sum(msg['h'])

        msg = graph.send(
                message_func=_send_func, 
                node_feat={'h': node_feat},
                edge_feat={'h': edge_feat})
        node_feat = graph.recv(reduce_func=_recv_func, msg=msg)
        node_feat = self.mlp(node_feat)
        return node_feat


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
Basic Encoder for compound atom/bond features.
"""
import numpy as np

import paddle
import paddle.nn as nn
import pgl

# from pahelix.utils.compound_tools import CompoundKit
# from pahelix.networks.basic_block import RBF


class AtomEmbedding(nn.Layer):
    """
    Atom Encoder
    """
    def __init__(self, atom_names, embed_dim):
        super(AtomEmbedding, self).__init__()
        self.atom_names = atom_names
        
        self.embed_list = nn.LayerList()
        for name in self.atom_names:
            embed = nn.Embedding(
                    CompoundKit.get_atom_feature_size(name) + 5,
                    embed_dim, 
                    weight_attr=nn.initializer.XavierUniform())
            self.embed_list.append(embed)

    def forward(self, node_features):
        """
        Args: 
            node_features(dict of tensor): node features.
        """
        out_embed = 0
        for i, name in enumerate(self.atom_names):
            out_embed += self.embed_list[i](node_features[name])
        return out_embed


class AtomFloatEmbedding(nn.Layer):
    """
    Atom Float Encoder
    """
    def __init__(self, atom_float_names, embed_dim, rbf_params=None):
        super(AtomFloatEmbedding, self).__init__()
        self.atom_float_names = atom_float_names
        
        if rbf_params is None:
            self.rbf_params = {
                'van_der_waals_radis': (np.arange(1, 3, 0.2), 10.0),   # (centers, gamma)
                'partial_charge': (np.arange(-1, 4, 0.25), 10.0),   # (centers, gamma)
                'mass': (np.arange(0, 2, 0.1), 10.0),   # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = nn.LayerList()
        self.rbf_list = nn.LayerList()
        for name in self.atom_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)

    def forward(self, feats):
        """
        Args: 
            feats(dict of tensor): node float features.
        """
        out_embed = 0
        for i, name in enumerate(self.atom_float_names):
            x = feats[name]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed


class BondEmbedding(nn.Layer):
    """
    Bond Encoder
    """
    def __init__(self, bond_names, embed_dim):
        super(BondEmbedding, self).__init__()
        self.bond_names = bond_names
        
        self.embed_list = nn.LayerList()
        for name in self.bond_names:
            embed = nn.Embedding(
                    CompoundKit.get_bond_feature_size(name) + 5,
                    embed_dim, 
                    weight_attr=nn.initializer.XavierUniform())
            self.embed_list.append(embed)

    def forward(self, edge_features):
        """
        Args: 
            edge_features(dict of tensor): edge features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_names):
            out_embed += self.embed_list[i](edge_features[name])
        return out_embed


class BondFloatRBF(nn.Layer):
    """
    Bond Float Encoder using Radial Basis Functions
    """
    def __init__(self, bond_float_names, embed_dim, rbf_params=None):
        super(BondFloatRBF, self).__init__()
        self.bond_float_names = bond_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_length': (np.arange(0, 2, 0.1), 10.0),   # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params
        
        self.linear_list = nn.LayerList()
        self.rbf_list = nn.LayerList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)

    def forward(self, bond_float_features):
        """
        Args: 
            bond_float_features(dict of tensor): bond float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            x = bond_float_features[name]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed


class BondAngleFloatRBF(nn.Layer):
    """
    Bond Angle Float Encoder using Radial Basis Functions
    """
    def __init__(self, bond_angle_float_names, embed_dim, rbf_params=None):
        super(BondAngleFloatRBF, self).__init__()
        self.bond_angle_float_names = bond_angle_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_angle': (np.arange(0, np.pi, 0.1), 10.0),   # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params
        
        self.linear_list = nn.LayerList()
        self.rbf_list = nn.LayerList()
        for name in self.bond_angle_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)

    def forward(self, bond_angle_float_features):
        """
        Args: 
            bond_angle_float_features(dict of tensor): bond angle float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_angle_float_names):
            x = bond_angle_float_features[name]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed

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
| Tools for compound features.
| Adapted from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/loader.py
"""
import os
from collections import OrderedDict

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem

DAY_LIGHT_FG_SMARTS_LIST = [
        # C
        "[CX4]",
        "[$([CX2](=C)=C)]",
        "[$([CX3]=[CX3])]",
        "[$([CX2]#C)]",
        # C & O
        "[CX3]=[OX1]",
        "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]",
        "[CX3](=[OX1])C",
        "[OX1]=CN",
        "[CX3](=[OX1])O",
        "[CX3](=[OX1])[F,Cl,Br,I]",
        "[CX3H1](=O)[#6]",
        "[CX3](=[OX1])[OX2][CX3](=[OX1])",
        "[NX3][CX3](=[OX1])[#6]",
        "[NX3][CX3]=[NX3+]",
        "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",
        "[NX3][CX3](=[OX1])[OX2H0]",
        "[NX3,NX4+][CX3](=[OX1])[OX2H,OX1-]",
        "[CX3](=O)[O-]",
        "[CX3](=[OX1])(O)O",
        "[CX3](=[OX1])([OX2])[OX2H,OX1H0-1]",
        "C[OX2][CX3](=[OX1])[OX2]C",
        "[CX3](=O)[OX2H1]",
        "[CX3](=O)[OX1H0-,OX2H1]",
        "[NX3][CX2]#[NX1]",
        "[#6][CX3](=O)[OX2H0][#6]",
        "[#6][CX3](=O)[#6]",
        "[OD2]([#6])[#6]",
        # H
        "[H]",
        "[!#1]",
        "[H+]",
        "[+H]",
        "[!H]",
        # N
        "[NX3;H2,H1;!$(NC=O)]",
        "[NX3][CX3]=[CX3]",
        "[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]",
        "[NX3;H2,H1;!$(NC=O)].[NX3;H2,H1;!$(NC=O)]",
        "[NX3][$(C=C),$(cc)]",
        "[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[O,N]",
        "[NX3H2,NH3X4+][CX4H]([*])[CX3](=[OX1])[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-]",
        "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-,N]",
        "[CH3X4]",
        "[CH2X4][CH2X4][CH2X4][NHX3][CH0X3](=[NH2X3+,NHX2+0])[NH2X3]",
        "[CH2X4][CX3](=[OX1])[NX3H2]",
        "[CH2X4][CX3](=[OX1])[OH0-,OH]",
        "[CH2X4][SX2H,SX1H0-]",
        "[CH2X4][CH2X4][CX3](=[OX1])[OH0-,OH]",
        "[$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H2][CX3](=[OX1])[OX2H,OX1-,N])]",
        "[CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:\
[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1",
        "[CHX4]([CH3X4])[CH2X4][CH3X4]",
        "[CH2X4][CHX4]([CH3X4])[CH3X4]",
        "[CH2X4][CH2X4][CH2X4][CH2X4][NX4+,NX3+0]",
        "[CH2X4][CH2X4][SX2][CH3X4]",
        "[CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1",
        "[$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[OX2H,OX1-,N]",
        "[CH2X4][OX2H]",
        "[NX3][CX3]=[SX1]",
        "[CHX4]([CH3X4])[OX2H]",
        "[CH2X4][cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12",
        "[CH2X4][cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1",
        "[CHX4]([CH3X4])[CH3X4]",
        "N[CX4H2][CX3](=[OX1])[O,N]",
        "N1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[O,N]",
        "[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]",
        "[$([NX1-]=[NX2+]=[NX1-]),$([NX1]#[NX2+]-[NX1-2])]",
        "[#7]",
        "[NX2]=N",
        "[NX2]=[NX2]",
        "[$([NX2]=[NX3+]([O-])[#6]),$([NX2]=[NX3+0](=[O])[#6])]",
        "[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]",
        "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]",
        "[NX3][NX3]",
        "[NX3][NX2]=[*]",
        "[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]",
        "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
        "[NX3+]=[CX3]",
        "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
        "[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])",
        "[CX3](=[OX1])[NX3H0]([NX3H0]([CX3](=[OX1]))[CX3](=[OX1]))[CX3](=[OX1])",
        "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",
        "[$([OX1]=[NX3](=[OX1])[OX1-]),$([OX1]=[NX3+]([OX1-])[OX1-])]",
        "[NX1]#[CX2]",
        "[CX1-]#[NX2+]",
        "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8].[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "[NX2]=[OX1]",
        "[$([#7+][OX1-]),$([#7v5]=[OX1]);!$([#7](~[O])~[O]);!$([#7]=[#7])]",
        # O
        "[OX2H]",
        "[#6][OX2H]",
        "[OX2H][CX3]=[OX1]",
        "[OX2H]P",
        "[OX2H][#6X3]=[#6]",
        "[OX2H][cX3]:[c]",
        "[OX2H][$(C=C),$(cc)]",
        "[$([OH]-*=[!#6])]",
        "[OX2,OX1-][OX2,OX1-]",
        # P
        "[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),\
$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-])\
,$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]",
        "[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),\
$([OX1-]),$([OX2][#6]),$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),\
$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]",
        # S
        "[S-][CX3](=S)[#6]",
        "[#6X3](=[SX1])([!N])[!N]",
        "[SX2]",
        "[#16X2H]",
        "[#16!H0]",
        "[#16X2H0]",
        "[#16X2H0][!#16]",
        "[#16X2H0][#16X2H0]",
        "[#16X2H0][!#16].[#16X2H0][!#16]",
        "[$([#16X3](=[OX1])[OX2H0]),$([#16X3+]([OX1-])[OX2H0])]",
        "[$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]",
        "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]",
        "[SX4](C)(C)(=O)=N",
        "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]",
        "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
        "[$([#16X3](=[OX1])([#6])[#6]),$([#16X3+]([OX1-])([#6])[#6])]",
        "[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]),$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6])]",
        "[$([SX4](=O)(=O)(O)O),$([SX4+2]([O-])([O-])(O)O)]",
        "[$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6]),$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2][#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2][#6])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2H,OX1H0-]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2H,OX1H0-])]",
        "[#16X2][OX2H,OX1H0-]",
        "[#16X2][OX2H0]",
        # X
        "[#6][F,Cl,Br,I]",
        "[F,Cl,Br,I]",
        "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]",
    ]


def get_gasteiger_partial_charges(mol, n_iter=12):
    """
    Calculates list of gasteiger partial charges for each atom in mol object.

    Args: 
        mol: rdkit mol object.
        n_iter(int): number of iterations. Default 12.

    Returns: 
        list of computed partial charges for each atom.
    """
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, nIter=n_iter,
                                                  throwOnParamFailure=True)
    partial_charges = [float(a.GetProp('_GasteigerCharge')) for a in
                       mol.GetAtoms()]
    return partial_charges


def create_standardized_mol_id(smiles):
    """
    Args:
        smiles: smiles sequence.

    Returns: 
        inchi.
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if not mol is None: # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles: # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return


def check_smiles_validity(smiles):
    """
    Check whether the smile can't be converted to rdkit mol object.
    """
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except Exception as e:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively.

    Args:
        mol: rdkit mol object.
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one.

    Args: 
        mol_list(list): a list of rdkit mol object.

    Returns:
        the largest mol.
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]

def rdchem_enum_to_list(values):
    """values = {0: rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, 
            1: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, 
            2: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, 
            3: rdkit.Chem.rdchem.ChiralType.CHI_OTHER}
    """
    return [values[i] for i in range(len(values))]


def safe_index(alist, elem):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return alist.index(elem)
    except ValueError:
        return len(alist) - 1


def get_atom_feature_dims(list_acquired_feature_names):
    """ tbd
    """
    return list(map(len, [CompoundKit.atom_vocab_dict[name] for name in list_acquired_feature_names]))


def get_bond_feature_dims(list_acquired_feature_names):
    """ tbd
    """
    list_bond_feat_dim = list(map(len, [CompoundKit.bond_vocab_dict[name] for name in list_acquired_feature_names]))
    # +1 for self loop edges
    return [_l + 1 for _l in list_bond_feat_dim]


class CompoundKit(object):
    """
    CompoundKit
    """
    atom_vocab_dict = {
        "atomic_num": list(range(1, 119)) + ['misc'],
        "chiral_tag": rdchem_enum_to_list(rdchem.ChiralType.values),
        "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "explicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "hybridization": rdchem_enum_to_list(rdchem.HybridizationType.values),
        "implicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "is_aromatic": [0, 1],
        "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'num_radical_e': [0, 1, 2, 3, 4, 'misc'],
        'atom_is_in_ring': [0, 1],
        'valence_out_shell': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size5': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size6': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size7': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size8': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    }
    bond_vocab_dict = {
        "bond_dir": rdchem_enum_to_list(rdchem.BondDir.values),
        "bond_type": rdchem_enum_to_list(rdchem.BondType.values),
        "is_in_ring": [0, 1],

        'bond_stereo': rdchem_enum_to_list(rdchem.BondStereo.values),
        'is_conjugated': [0, 1],
    }
    # float features
    atom_float_names = ["van_der_waals_radis", "partial_charge", 'mass']
    # bond_float_feats= ["bond_length", "bond_angle"]     # optional

    ### functional groups
    day_light_fg_smarts_list = DAY_LIGHT_FG_SMARTS_LIST
    day_light_fg_mo_list = [Chem.MolFromSmarts(smarts) for smarts in day_light_fg_smarts_list]

    morgan_fp_N = 200
    morgan2048_fp_N = 2048
    maccs_fp_N = 167

    period_table = Chem.GetPeriodicTable()

    ### atom

    @staticmethod
    def get_atom_value(atom, name):
        """get atom values"""
        if name == 'atomic_num':
            return atom.GetAtomicNum()
        elif name == 'chiral_tag':
            return atom.GetChiralTag()
        elif name == 'degree':
            return atom.GetDegree()
        elif name == 'explicit_valence':
            return atom.GetExplicitValence()
        elif name == 'formal_charge':
            return atom.GetFormalCharge()
        elif name == 'hybridization':
            return atom.GetHybridization()
        elif name == 'implicit_valence':
            return atom.GetImplicitValence()
        elif name == 'is_aromatic':
            return int(atom.GetIsAromatic())
        elif name == 'mass':
            return int(atom.GetMass())
        elif name == 'total_numHs':
            return atom.GetTotalNumHs()
        elif name == 'num_radical_e':
            return atom.GetNumRadicalElectrons()
        elif name == 'atom_is_in_ring':
            return int(atom.IsInRing())
        elif name == 'valence_out_shell':
            return CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())
        else:
            raise ValueError(name)

    @staticmethod
    def get_atom_feature_id(atom, name):
        """get atom features id"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return safe_index(CompoundKit.atom_vocab_dict[name], CompoundKit.get_atom_value(atom, name))

    @staticmethod
    def get_atom_feature_size(name):
        """get atom features size"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return len(CompoundKit.atom_vocab_dict[name])

    ### bond

    @staticmethod
    def get_bond_value(bond, name):
        """get bond values"""
        if name == 'bond_dir':
            return bond.GetBondDir()
        elif name == 'bond_type':
            return bond.GetBondType()
        elif name == 'is_in_ring':
            return int(bond.IsInRing())
        elif name == 'is_conjugated':
            return int(bond.GetIsConjugated())
        elif name == 'bond_stereo':
            return bond.GetStereo()
        else:
            raise ValueError(name)

    @staticmethod
    def get_bond_feature_id(bond, name):
        """get bond features id"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return safe_index(CompoundKit.bond_vocab_dict[name], CompoundKit.get_bond_value(bond, name))

    @staticmethod
    def get_bond_feature_size(name):
        """get bond features size"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return len(CompoundKit.bond_vocab_dict[name])

    ### fingerprint

    @staticmethod
    def get_morgan_fingerprint(mol, radius=2):
        """get morgan fingerprint"""
        nBits = CompoundKit.morgan_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]
    
    @staticmethod
    def get_morgan2048_fingerprint(mol, radius=2):
        """get morgan2048 fingerprint"""
        nBits = CompoundKit.morgan2048_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_maccs_fingerprint(mol):
        """get maccs fingerprint"""
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]

    ### functional groups

    @staticmethod
    def get_daylight_functional_group_counts(mol):
        """get daylight functional group counts"""
        fg_counts = []
        for fg_mol in CompoundKit.day_light_fg_mo_list:
            sub_structs = Chem.Mol.GetSubstructMatches(mol, fg_mol, uniquify=True)
            fg_counts.append(len(sub_structs))
        return fg_counts

    @staticmethod
    def get_ring_size(mol):
        """return (N,6) list"""
        rings = mol.GetRingInfo()
        rings_info = []
        for r in rings.AtomRings():
            rings_info.append(r)
        ring_list = []
        for atom in mol.GetAtoms():
            atom_result = []
            for ringsize in range(3, 9):
                num_of_ring_at_ringsize = 0
                for r in rings_info:
                    if len(r) == ringsize and atom.GetIdx() in r:
                        num_of_ring_at_ringsize += 1
                if num_of_ring_at_ringsize > 8:
                    num_of_ring_at_ringsize = 9
                atom_result.append(num_of_ring_at_ringsize)
            
            ring_list.append(atom_result)
        return ring_list

    @staticmethod
    def atom_to_feat_vector(atom):
        """ tbd """
        atom_names = {
            "atomic_num": safe_index(CompoundKit.atom_vocab_dict["atomic_num"], atom.GetAtomicNum()),
            "chiral_tag": safe_index(CompoundKit.atom_vocab_dict["chiral_tag"], atom.GetChiralTag()),
            "degree": safe_index(CompoundKit.atom_vocab_dict["degree"], atom.GetTotalDegree()),
            "explicit_valence": safe_index(CompoundKit.atom_vocab_dict["explicit_valence"], atom.GetExplicitValence()),
            "formal_charge": safe_index(CompoundKit.atom_vocab_dict["formal_charge"], atom.GetFormalCharge()),
            "hybridization": safe_index(CompoundKit.atom_vocab_dict["hybridization"], atom.GetHybridization()),
            "implicit_valence": safe_index(CompoundKit.atom_vocab_dict["implicit_valence"], atom.GetImplicitValence()),
            "is_aromatic": safe_index(CompoundKit.atom_vocab_dict["is_aromatic"], int(atom.GetIsAromatic())),
            "total_numHs": safe_index(CompoundKit.atom_vocab_dict["total_numHs"], atom.GetTotalNumHs()),
            'num_radical_e': safe_index(CompoundKit.atom_vocab_dict['num_radical_e'], atom.GetNumRadicalElectrons()),
            'atom_is_in_ring': safe_index(CompoundKit.atom_vocab_dict['atom_is_in_ring'], int(atom.IsInRing())),
            'valence_out_shell': safe_index(CompoundKit.atom_vocab_dict['valence_out_shell'],
                                            CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())),
            'van_der_waals_radis': CompoundKit.period_table.GetRvdw(atom.GetAtomicNum()),
            'partial_charge': CompoundKit.check_partial_charge(atom),
            'mass': atom.GetMass(),
        }
        return atom_names

    @staticmethod
    def get_atom_names(mol):
        """get atom name list
        TODO: to be remove in the future
        """
        atom_features_dicts = []
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts.append(CompoundKit.atom_to_feat_vector(atom))

        ring_list = CompoundKit.get_ring_size(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts[i]['in_num_ring_with_size3'] = safe_index(
                    CompoundKit.atom_vocab_dict['in_num_ring_with_size3'], ring_list[i][0])
            atom_features_dicts[i]['in_num_ring_with_size4'] = safe_index(
                    CompoundKit.atom_vocab_dict['in_num_ring_with_size4'], ring_list[i][1])
            atom_features_dicts[i]['in_num_ring_with_size5'] = safe_index(
                    CompoundKit.atom_vocab_dict['in_num_ring_with_size5'], ring_list[i][2])
            atom_features_dicts[i]['in_num_ring_with_size6'] = safe_index(
                    CompoundKit.atom_vocab_dict['in_num_ring_with_size6'], ring_list[i][3])
            atom_features_dicts[i]['in_num_ring_with_size7'] = safe_index(
                    CompoundKit.atom_vocab_dict['in_num_ring_with_size7'], ring_list[i][4])
            atom_features_dicts[i]['in_num_ring_with_size8'] = safe_index(
                    CompoundKit.atom_vocab_dict['in_num_ring_with_size8'], ring_list[i][5])

        return atom_features_dicts
        
    @staticmethod
    def check_partial_charge(atom):
        """tbd"""
        pc = atom.GetDoubleProp('_GasteigerCharge')
        if pc != pc:
            # unsupported atom, replace nan with 0
            pc = 0
        if pc == float('inf'):
            # max 4 for other atoms, set to 10 here if inf is get
            pc = 10
        return pc


class Compound3DKit(object):
    """the 3Dkit of Compound"""
    @staticmethod
    def get_atom_poses(mol, conf):
        """tbd"""
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses

    @staticmethod
    def get_MMFF_atom_poses(mol, numConfs=None, return_energy=False):
        """the atoms of mol will be changed in some cases."""
        try:
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
            ### MMFF generates multiple conformations
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            index = np.argmin([x[1] for x in res])
            energy = res[index][1]
            conf = new_mol.GetConformer(id=int(index))
        except:
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)
            energy = 0
            conf = new_mol.GetConformer()

        atom_poses = Compound3DKit.get_atom_poses(new_mol, conf)
        if return_energy:
            return new_mol, atom_poses, energy
        else:
            return new_mol, atom_poses

    @staticmethod
    def get_2d_atom_poses(mol):
        """get 2d atom poses"""
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return atom_poses

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        """get bond lengths"""
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))
        bond_lengths = np.array(bond_lengths, 'float32')
        return bond_lengths

    @staticmethod
    def get_superedge_angles(edges, atom_poses, dir_type='HT'):
        """get superedge angles"""
        def _get_vec(atom_poses, edge):
            return atom_poses[edge[1]] - atom_poses[edge[0]]
        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle

        E = len(edges)
        edge_indices = np.arange(E)
        super_edges = []
        bond_angles = []
        bond_angle_dirs = []
        for tar_edge_i in range(E):
            tar_edge = edges[tar_edge_i]
            if dir_type == 'HT':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]
            elif dir_type == 'HH':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[1]]
            else:
                raise ValueError(dir_type)
            for src_edge_i in src_edge_indices:
                if src_edge_i == tar_edge_i:
                    continue
                src_edge = edges[src_edge_i]
                src_vec = _get_vec(atom_poses, src_edge)
                tar_vec = _get_vec(atom_poses, tar_edge)
                super_edges.append([src_edge_i, tar_edge_i])
                angle = _get_angle(src_vec, tar_vec)
                bond_angles.append(angle)
                bond_angle_dirs.append(src_edge[1] == tar_edge[0])  # H -> H or H -> T

        if len(super_edges) == 0:
            super_edges = np.zeros([0, 2], 'int64')
            bond_angles = np.zeros([0,], 'float32')
        else:
            super_edges = np.array(super_edges, 'int64')
            bond_angles = np.array(bond_angles, 'float32')
        return super_edges, bond_angles, bond_angle_dirs



def new_smiles_to_graph_data(smiles, **kwargs):
    """
    Convert smiles to graph data.
    """
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    data = new_mol_to_graph_data(mol)
    return data


def new_mol_to_graph_data(mol):
    """
    mol_to_graph_data

    Args:
        atom_features: Atom features.
        edge_features: Edge features.
        morgan_fingerprint: Morgan fingerprint.
        functional_groups: Functional groups.
    """
    if len(mol.GetAtoms()) == 0:
        return None

    atom_id_names = list(CompoundKit.atom_vocab_dict.keys()) + CompoundKit.atom_float_names
    bond_id_names = list(CompoundKit.bond_vocab_dict.keys())

    data = {}

    ### atom features
    data = {name: [] for name in atom_id_names}

    raw_atom_feat_dicts = CompoundKit.get_atom_names(mol)
    for atom_feat in raw_atom_feat_dicts:
        for name in atom_id_names:
            data[name].append(atom_feat[name])

    ### bond and bond features
    for name in bond_id_names:
        data[name] = []
    data['edges'] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j and j->i
        data['edges'] += [(i, j), (j, i)]
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name)
            data[name] += [bond_feature_id] * 2

    #### self loop
    N = len(data[atom_id_names[0]])
    for i in range(N):
        data['edges'] += [(i, i)]
    for name in bond_id_names:
        bond_feature_id = get_bond_feature_dims([name])[0] - 1   # self loop: value = len - 1
        data[name] += [bond_feature_id] * N

    ### make ndarray and check length
    for name in list(CompoundKit.atom_vocab_dict.keys()):
        data[name] = np.array(data[name], 'int64')
    for name in CompoundKit.atom_float_names:
        data[name] = np.array(data[name], 'float32')
    for name in bond_id_names:
        data[name] = np.array(data[name], 'int64')
    data['edges'] = np.array(data['edges'], 'int64')

    ### morgan fingerprint
    data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), 'int64')
    # data['morgan2048_fp'] = np.array(CompoundKit.get_morgan2048_fingerprint(mol), 'int64')
    data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), 'int64')
    data['daylight_fg_counts'] = np.array(CompoundKit.get_daylight_functional_group_counts(mol), 'int64')
    return data


def mol_to_graph_data(mol):
    """
    mol_to_graph_data

    Args:
        atom_features: Atom features.
        edge_features: Edge features.
        morgan_fingerprint: Morgan fingerprint.
        functional_groups: Functional groups.
    """
    if len(mol.GetAtoms()) == 0:
        return None

    atom_id_names = [
        "atomic_num", "chiral_tag", "degree", "explicit_valence", 
        "formal_charge", "hybridization", "implicit_valence", 
        "is_aromatic", "total_numHs",
    ]
    bond_id_names = [
        "bond_dir", "bond_type", "is_in_ring",
    ]
    
    data = {}
    for name in atom_id_names:
        data[name] = []
    data['mass'] = []
    for name in bond_id_names:
        data[name] = []
    data['edges'] = []

    ### atom features
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return None
        for name in atom_id_names:
            data[name].append(CompoundKit.get_atom_feature_id(atom, name) + 1)  # 0: OOV
        data['mass'].append(CompoundKit.get_atom_value(atom, 'mass') * 0.01)

    ### bond features
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j and j->i
        data['edges'] += [(i, j), (j, i)]
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name) + 1   # 0: OOV
            data[name] += [bond_feature_id] * 2

    ### self loop (+2)
    N = len(data[atom_id_names[0]])
    for i in range(N):
        data['edges'] += [(i, i)]
    for name in bond_id_names:
        bond_feature_id = CompoundKit.get_bond_feature_size(name) + 2   # N + 2: self loop
        data[name] += [bond_feature_id] * N

    ### check whether edge exists
    if len(data['edges']) == 0: # mol has no bonds
        for name in bond_id_names:
            data[name] = np.zeros((0,), dtype="int64")
        data['edges'] = np.zeros((0, 2), dtype="int64")

    ### make ndarray and check length
    for name in atom_id_names:
        data[name] = np.array(data[name], 'int64')
    data['mass'] = np.array(data['mass'], 'float32')
    for name in bond_id_names:
        data[name] = np.array(data[name], 'int64')
    data['edges'] = np.array(data['edges'], 'int64')

    ### morgan fingerprint
    data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), 'int64')
    # data['morgan2048_fp'] = np.array(CompoundKit.get_morgan2048_fingerprint(mol), 'int64')
    data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), 'int64')
    data['daylight_fg_counts'] = np.array(CompoundKit.get_daylight_functional_group_counts(mol), 'int64')
    return data


def mol_to_geognn_graph_data(mol, atom_poses, dir_type):
    """
    mol: rdkit molecule
    dir_type: direction type for bond_angle grpah
    """
    if len(mol.GetAtoms()) == 0:
        return None

    data = mol_to_graph_data(mol)

    data['atom_pos'] = np.array(atom_poses, 'float32')
    data['bond_length'] = Compound3DKit.get_bond_lengths(data['edges'], data['atom_pos'])
    BondAngleGraph_edges, bond_angles, bond_angle_dirs = \
            Compound3DKit.get_superedge_angles(data['edges'], data['atom_pos'])
    data['BondAngleGraph_edges'] = BondAngleGraph_edges
    data['bond_angle'] = np.array(bond_angles, 'float32')
    return data


def mol_to_geognn_graph_data_MMFF3d(mol):
    """tbd"""
    if len(mol.GetAtoms()) <= 400:
        mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
    else:
        atom_poses = Compound3DKit.get_2d_atom_poses(mol)
    return mol_to_geognn_graph_data(mol, atom_poses, dir_type='HT')


def mol_to_geognn_graph_data_raw3d(mol):
    """tbd"""
    atom_poses = Compound3DKit.get_atom_poses(mol, mol.GetConformer())
    return mol_to_geognn_graph_data(mol, atom_poses, dir_type='HT')






# if __name__ == "__main__":
#     smiles = "OCc1ccccc1CN"
#     # smiles = r"[H]/[NH+]=C(\N)C1=CC(=O)/C(=C\C=c2ccc(=C(N)[NH3+])cc2)C=C1"
#     mol = AllChem.MolFromSmiles(smiles)
#     print(len(smiles))
#     print(mol)
#     data = mol_to_geognn_graph_data_MMFF3d(mol)

class Activation(nn.Layer):
    """
    Activation
    """
    def __init__(self, act_type, **params):
        super(Activation, self).__init__()
        if act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'leaky_relu':
            self.act = nn.LeakyReLU(**params)
        else:
            raise ValueError(act_type)
     
    def forward(self, x):
        """tbd"""
        return self.act(x)


class MLP(nn.Layer):
    """
    MLP
    """
    def __init__(self, layer_num, in_size, hidden_size, out_size, act, dropout_rate):
        super(MLP, self).__init__()

        layers = []
        for layer_id in range(layer_num):
            if layer_id == 0:
                layers.append(nn.Linear(in_size, hidden_size))
                layers.append(nn.Dropout(dropout_rate))
                layers.append(Activation(act))
            elif layer_id < layer_num - 1:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.Dropout(dropout_rate))
                layers.append(Activation(act))
            else:
                layers.append(nn.Linear(hidden_size, out_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, dim).
        """
        return self.mlp(x)


class RBF(nn.Layer):
    """
    Radial Basis Function
    """
    def __init__(self, centers, gamma, dtype='float32'):
        super(RBF, self).__init__()
        self.centers = paddle.reshape(paddle.to_tensor(centers, dtype=dtype), [1, -1])
        self.gamma = gamma
    
    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = paddle.reshape(x, [-1, 1])
        return paddle.exp(-self.gamma * paddle.square(x - self.centers))
        
    

def load_json_config(path):
    """tbd"""
    return json.load(open(path, 'r'))