#
#        LDS-GNN
#
#   File:     lds_gnn/utils.py
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

"""Utility methods for configurations, logging, early stopping, etc."""

import glob
import gzip
import os
import pickle
import sys
import time
from collections import defaultdict, OrderedDict

import far_ho as far
import numpy as np
import tensorflow as tf


VERSION = 1
SAVE_DIR = 'results'
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
    print('FOLDER', SAVE_DIR, 'CREATED')
SVD_MAP_FILE = '.svd_map'


class Config:
    """ Base class of a configuration instance; offers keyword initialization with easy defaults,
    pretty printing and grid search!
    """
    def __init__(self, **kwargs):
        self._version = 1
        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise AttributeError('This config does not include attribute: {}'.format(k) +
                                     '\n Available attributes with relative defaults are\n{}'.format(
                                         str(self.default_instance())))

    def __str__(self):
        _sting_kw = lambda k, v: '{}={}'.format(k, v)

        def _str_dict_pr(obj):
            return [_sting_kw(k, v) for k, v in obj.items()] if isinstance(obj, dict) else str(obj)

        return self.__class__.__name__ + '[' + '\n\t'.join(
            _sting_kw(k, _str_dict_pr(v)) for k, v in sorted(self.__dict__.items())) + ']\n'

    @classmethod
    def default_instance(cls):
        return cls()

    @classmethod
    def grid(cls, **kwargs):
        """Builds a mesh grid with given keyword arguments for this Config class.
        If the value is not a list, then it is considered fixed"""

        class MncDc:
            """This is because np.meshgrid does not always work properly..."""

            def __init__(self, a):
                self.a = a  # tuple!

            def __call__(self):
                return self.a

        sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
        for k, v in sin.items():
            copy_v = []
            for e in v:
                copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
            sin[k] = copy_v

        grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
        return [cls(**far.utils.merge_dicts(
            {k: v for k, v in kwargs.items() if not isinstance(v, list)},
            {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
        )) for vv in grd]


class Placeholders:
    def __init__(self, X, Y):
        self.label_mask = tf.placeholder(tf.int32)
        self.X = tf.constant(X, tf.float32)
        self.Y = tf.constant(Y, tf.float32)
        self.keep_prob = tf.placeholder_with_default(1., shape=())
        self.n = X.shape[0]

    def fd(self, mask, *other_fds):
        return far.utils.merge_dicts({self.label_mask: mask}, *other_fds)

    def fds(self, *masks):
        return [self.fd(m) for m in masks]


class GraphKeys(far.GraphKeys):
    STOCHASTIC_HYPER = 'stochastic_hyper'


def setup_tf(seed):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    if tf.get_default_session(): tf.get_default_session().close()
    return tf.InteractiveSession()


def new_gs():  # creator for the global step
    return tf.Variable(0, trainable=False, name='step',
                       collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])


def get_gs():
    try:
        return tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0]
    except IndexError:
        return None


def upper_triangular_initializer(const):
    def _init(shape, dtype, _=None):
        a = np.zeros(shape)
        for i in range(0, shape[0]):
            for j in range(i + 1, shape[1]):
                a[i, j] = const
        return tf.constant(a, dtype=dtype)

    return _init


def upper_tri_const(shape, minval=0., maxval=1.):
    return lambda v: tf.maximum(tf.minimum(v * upper_triangular_mask(shape), maxval), minval)


def box_const(minval=0., maxval=1.):
    return lambda v: tf.maximum(tf.minimum(v, maxval), minval)


def upper_triangular_mask(shape, as_array=False):
    a = np.zeros(shape)
    for i in range(0, shape[0]):
        for j in range(i + 1, shape[1]):
            a[i, j] = 1.
    return tf.constant(a, dtype=tf.float32) if not as_array else a


def init_svd(data_config=None, config=None):
    fn = ''
    if data_config: fn += str(data_config)
    if config: fn += str(config)
    return defaultdict(list, (('config', config), ('data_config', data_config),
                              ('name', fn)))


def restore_from_svd(svd, session=None, verbose=False):
    if session is None: session = tf.get_default_session()
    for v in tf.global_variables():
        try:
            session.run(v.assign(svd[v.name]))
            if verbose: print(v.name, 'restored')
        except KeyError:
            print('WARNING: variable', v, 'not in SVD', file=sys.stderr)
        except ValueError as e:
            print(e, file=sys.stderr)


def update_append(dct, **updates):
    for k, e in updates.items():
        dct[k].append(e)


def update_append_v2(dct: dict, upd_dct: dict):
    for k, e in upd_dct.items():
        dct[k].append(e)


def gz_read(name, results=True):
    name = '{}/{}.gz'.format(SAVE_DIR, name) if results else '{}.gz'.format(name)
    with gzip.open(name, 'rb') as f:
        return pickle.load(f)


def gz_write(content, name, results=True):
    name = '{}/{}.gz'.format(SAVE_DIR, name) if results else '{}.gz'.format(name)
    with gzip.open(name, 'wb') as f:
        pickle.dump(content, f)


def list_results(*keywords, verbose=True):
    _strip_fn = lambda nn: nn.split(os.sep)[1][:-3]

    svd_map = gz_read(SVD_MAP_FILE, results=False)
    result_list = sorted(glob.glob('{}/*.gz'.format(SAVE_DIR)),
                         key=lambda x: os.path.getmtime(x))
    result_list = map(_strip_fn, result_list)
    try:
        result_list = list(filter(lambda nn: all([kw in svd_map[nn] for kw in keywords]), result_list))
    except KeyError:
        print('Misc.list_results: something wrong happened: returning None', file=sys.stderr)
        return []
    if verbose:
        for k, v in enumerate(result_list):
            print(k, '->', svd_map[v])
    return result_list


# noinspection PyUnboundLocalVariable
def load_results(*keywords, exp_id=None, verbose=True):
    if verbose: print('loading results:')
    rs = list_results(*keywords, verbose=verbose)
    if exp_id is not None:
        rs = [rs[exp_id]]
    ldd = list(map(gz_read, rs))
    loaded = ldd if len(ldd) > 1 else ldd[0]
    return loaded


def early_stopping(patience, maxiters=1e10, on_accept=None, on_refuse=None, on_close=None, verbose=True):
    """
    Generator that implements early stopping. Use `send` method to give to update the state of the generator
    (e.g. with last validation accuracy)

    :param patience:
    :param maxiters:
    :param on_accept: function to be executed upon acceptance of the iteration
    :param on_refuse: function to be executed when the iteration is rejected (i.e. the value is lower then best)
    :param on_close: function to be exectued when early stopping activates
    :param verbose:
    :return: a step generator
    """
    val = None
    pat = patience
    t = 0
    while pat and t < maxiters:
        new_val = yield t
        if new_val is not None:
            if val is None or new_val > val:
                val = new_val
                pat = patience
                if on_accept:
                    try:
                        on_accept(t, val)
                    except TypeError:
                        try:
                            on_accept(t)
                        except TypeError:
                            on_accept()
                if verbose: print('ES t={}: Increased val accuracy: {}'.format(t, val))
            else:
                pat -= 1
                if on_refuse: on_refuse(t)
        else:
            t += 1
    yield
    if on_close: on_close(val)
    if verbose: print('ES: ending after', t, 'iterations')


def early_stopping_with_save(patience, ss, svd, maxiters=1e10, var_list=None,
                             on_accept=None, on_refuse=None, on_close=None, verbose=True):
    starting_time = -1
    gz_name = str(time.time())
    svd['file name'] = gz_name

    def _on_accept(t, val):
        nonlocal starting_time
        if starting_time == -1: starting_time = time.time()
        _var_list = tf.global_variables() if var_list is None else var_list

        svd.update((v.name, ss.run(v)) for v in _var_list)
        svd['on accept t'].append(t)
        if on_accept:
            try:
                on_accept(t, val)
            except TypeError:
                try:
                    on_accept(t)
                except TypeError:
                    on_accept()

    def _on_close(val):
        svd['es final value'] = val
        svd['version'] = VERSION
        svd['running time'] = time.time() - starting_time
        gz_write(svd, gz_name)

        try:
            fl_dict = gz_read(SVD_MAP_FILE, results=False)
        except FileNotFoundError:
            fl_dict = {}
            print('CREATING SVD MAP FILE WITH NAME:', SVD_MAP_FILE)

        fl_dict[gz_name] = svd['name']
        gz_write(fl_dict, SVD_MAP_FILE, results=False)

        if on_close:
            try:
                on_close(val)
            except TypeError:
                on_close()

    return early_stopping(patience, maxiters,
                          on_accept=_on_accept,
                          on_refuse=on_refuse,
                          on_close=_on_close,
                          verbose=verbose)
