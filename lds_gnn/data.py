#
#        LDS-GNN
#
#   File:     lds_gnn/data.py
#   Authors:  Luca Franceschi (luca.franceschi@iit.it)
#             Xiao He
#             Mathias Niepert (mathias.niepert@neclab.eu)
#
# NEC Laboratories Europe GmbH, Copyright (c) 2019, All rights reserved.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#        PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor.
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#


"""This module contains methods to load and manage datasets. For graph based data, it mostly resorts to gcn package"""

import numpy as np
from gcn.utils import load_data

from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer

import scipy.sparse as sp

try:
    from lds_gnn.utils import Config, upper_triangular_mask
except ImportError as e:
    from utils import Config, upper_triangular_mask


class ConfigData(Config):
    def __init__(self, **kwargs):
        self.seed = 0
        self.f1 = 'load_data_del_edges'
        self.dataset_name = 'cora'
        self.kwargs_f1 = {}
        self.f2 = 'reorganize_data_for_es'
        self.kwargs_f2 = {}
        super().__init__(**kwargs)

    def load(self):
        res = eval(self.f1)(seed=self.seed, dataset_name=self.dataset_name, **self.kwargs_f1)
        if self.f2:
            res = eval(self.f2)(res, **self.kwargs_f2, seed=self.seed)
        return res


class EdgeDelConfigData(ConfigData):
    def __init__(self, **kwargs):
        self.prob_del = 0.5
        self.enforce_connected = True
        super().__init__(**kwargs)
        self.kwargs_f1['prob_del'] = self.prob_del
        if not self.enforce_connected:
            self.kwargs_f1['enforce_connected'] = self.enforce_connected
        del self.prob_del
        del self.enforce_connected


class UCI(ConfigData):

    def __init__(self, **kwargs):
        self.n_train = None
        self.n_val = None
        self.n_es = None
        self.scale = None
        super().__init__(**kwargs)

    def load(self):
        if self.dataset_name == 'iris':
            data = datasets.load_iris()
        elif self.dataset_name == 'wine':
            data = datasets.load_wine()
        elif self.dataset_name == 'breast_cancer':
            data = datasets.load_breast_cancer()
        elif self.dataset_name == 'digits':
            data = datasets.load_digits()
        elif self.dataset_name == 'fma':
            import os
            data = np.load('%s/fma/fma.npz' % os.getcwd())
        elif self.dataset_name == '20news10':
            from sklearn.datasets import fetch_20newsgroups
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.feature_extraction.text import TfidfTransformer
            categories = ['alt.atheism',
                          'comp.sys.ibm.pc.hardware',
                          'misc.forsale',
                          'rec.autos',
                          'rec.sport.hockey',
                          'sci.crypt',
                          'sci.electronics',
                          'sci.med',
                          'sci.space',
                          'talk.politics.guns']
            data = fetch_20newsgroups(subset='all', categories=categories)
            vectorizer = CountVectorizer(stop_words='english', min_df=0.05)
            X_counts = vectorizer.fit_transform(data.data).toarray()
            transformer = TfidfTransformer(smooth_idf=False)
            features = transformer.fit_transform(X_counts).todense()
        else:
            raise AttributeError('dataset not available')

        if self.dataset_name != 'fma':
            from sklearn.preprocessing import scale
            if self.dataset_name != '20news10':
                if self.scale:
                    features = scale(data.data)
                else:
                    features = data.data
            y = data.target
        else:
            features = data['X']
            y = data['y']
        ys = LabelBinarizer().fit_transform(y)
        if ys.shape[1] == 1:
            ys = np.hstack([ys, 1 - ys])
        n = features.shape[0]
        from sklearn.model_selection import train_test_split
        train, test, y_train, y_test = train_test_split(np.arange(n), y, random_state=self.seed,
                                                        train_size=self.n_train + self.n_val + self.n_es,
                                                        test_size=n - self.n_train - self.n_val - self.n_es,
                                                        stratify=y)
        train, es, y_train, y_es = train_test_split(train, y_train, random_state=self.seed,
                                                    train_size=self.n_train + self.n_val, test_size=self.n_es,
                                                    stratify=y_train)
        train, val, y_train, y_val = train_test_split(train, y_train, random_state=self.seed,
                                                      train_size=self.n_train, test_size=self.n_val,
                                                      stratify=y_train)

        train_mask = np.zeros([n, ], dtype=bool)
        train_mask[train] = True
        val_mask = np.zeros([n, ], dtype=bool)
        val_mask[val] = True
        es_mask = np.zeros([n, ], dtype=bool)
        es_mask[es] = True
        test_mask = np.zeros([n, ], dtype=bool)
        test_mask[test] = True

        return np.zeros([n, n]), np.zeros([n, n]), features, ys, train_mask, val_mask, es_mask, test_mask


def graph_delete_connections(prob_del, seed, adj, features, y_train,
                             *other_splittables, to_dense=False,
                             enforce_connected=False):
    rnd = np.random.RandomState(seed)

    features = preprocess_features(features)

    if to_dense:
        features = features.toarray()
        adj = adj.toarray()
    del_adj = np.array(adj, dtype=np.float32)

    smpl = rnd.choice([0., 1.], p=[prob_del, 1. - prob_del], size=adj.shape) * upper_triangular_mask(
        adj.shape, as_array=True)
    smpl += smpl.transpose()

    del_adj *= smpl
    if enforce_connected:
        add_edges = 0
        for k, a in enumerate(del_adj):
            if not list(np.nonzero(a)[0]):
                prev_connected = list(np.nonzero(adj[k, :])[0])
                other_node = rnd.choice(prev_connected)
                del_adj[k, other_node] = 1
                del_adj[other_node, k] = 1
                add_edges += 1
        print('# ADDED EDGES: ', add_edges)
    return (adj, del_adj, features, y_train) + other_splittables


def load_data_del_edges(prob_del=0.4, seed=0, to_dense=True, enforce_connected=True,
                        dataset_name='cora'):
    res = graph_delete_connections(prob_del, seed, *load_data(dataset_name), to_dense=to_dense,
                                   enforce_connected=enforce_connected)
    return res


def reorganize_data_for_es(loaded_data, seed=0, es_n_data_prop=0.5):
    adj, adj_mods, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = loaded_data
    ys = y_train + y_val + y_test
    features = preprocess_features(features)
    msk1, msk2 = divide_mask(es_n_data_prop, np.sum(val_mask), seed=seed)
    mask_val = np.array(val_mask)
    mask_es = np.array(val_mask)
    mask_val[mask_val] = msk2
    mask_es[mask_es] = msk1

    return adj, adj_mods, features, ys, train_mask, mask_val, mask_es, test_mask


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(features)


def divide_mask(n1, n_tot, seed=0):
    rnd = np.random.RandomState(seed)
    p = n1 / n_tot if isinstance(n1, int) else n1
    chs = rnd.choice([True, False], size=n_tot, p=[p, 1. - p])
    return chs, ~chs
