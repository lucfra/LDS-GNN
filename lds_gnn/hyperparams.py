#
#        LDS-GNN
#
#   File:     lds_gnn/hyperparams.py
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


"""The module contains the main class responsible for hypergradient computation (estimation)
as well as utility functions to create and manage hyperparamter variables"""

import tensorflow as tf
import far_ho as far

try:
    import lds_gnn.utils
    from lds_gnn.utils import GraphKeys
except ImportError as e:
    # noinspection PyUnresolvedReferences
    from utils import GraphKeys

_STC_INITIALIZERs = {}
_STC_MAP = {}


def symm_adj_sample(probs):
    """Sampling function for symmetric Bernoulli matrices"""
    e = bernoulli_hard_sample(probs)
    return e + tf.transpose(e)


def bernoulli_hard_sample(probs):
    """Sampling function for Bernoulli"""
    return tf.floor(tf.random_uniform(probs.shape, minval=0., maxval=1.) + probs)


def get_stc_hyperparameter(name, initializer=None, shape=None, constraints=None,
                           sample_func=None, hyper_probs=None):
    """
    Get a stochastic hyperparameter. Defaults to Bernoulli hyperparameter. Mostly follows the signature of
    `tf.get_variable`

    :param name: a name for the hyperparameter
    :param initializer: an initializer (or initial value) for the parameters of the distribution
    :param shape: a shape for the stochastic hyperparameter
    :param constraints: additional (simple) constraints for the parameters of the distribution
    :param sample_func: a function that takes the distribution parameters and returns a sample
    :param hyper_probs: the variables used for the underlying probability distribution
    :return: The stochastic hyperparameter (not the distribution variables!)
    """
    if constraints is None:
        constraints = lambda _v: tf.maximum(tf.minimum(_v, 1.), 0.)
    if hyper_probs is None:  # creates the hyperparameter that is also used for sampling
        hyper_probs = tf.get_variable(
            name + '/' + GraphKeys.STOCHASTIC_HYPER, trainable=False,
            constraint=constraints,
            initializer=initializer,
            shape=shape,
            collections=[GraphKeys.GLOBAL_VARIABLES, GraphKeys.STOCHASTIC_HYPER]
        )
    if sample_func is None:
        sample_func = bernoulli_hard_sample
    hyper_sample = far.get_hyperparameter(
        name,
        initializer=sample_func(hyper_probs),
        collections=GraphKeys.STOCHASTIC_HYPER
    )
    far.utils.remove_from_collection(GraphKeys.GLOBAL_VARIABLES, hyper_sample)
    with tf.control_dependencies([tf.variables_initializer([hyper_sample])]):  # re-initialize and return the value
        _STC_INITIALIZERs[hyper_sample] = hyper_sample.read_value()

    _STC_MAP[hyper_sample] = hyper_probs

    return hyper_sample


get_bernoulli_hyperparameter = get_stc_hyperparameter


def get_probs_var(hyper):
    """Returns the distribution's parameters of stochastic hyperparameter"""
    return _STC_MAP[hyper]


def sample(hyper):
    """ Returns a `sampler` operation (in the form of an initializer, for the stochastic hyperparameter `hyper`"""
    return _STC_INITIALIZERs[hyper]


def is_stochastic_hyper(hyper):
    """Returns true if the hyperparameter is stochastic"""
    return hyper in tf.get_collection(GraphKeys.STOCHASTIC_HYPER)


def hyper_or_stochastic_hyper(hyper):
    """Returns either the underlying parameters of the probability distribution if `hyper` is stochastic,
    or `hyper`"""
    return get_probs_var(hyper) if is_stochastic_hyper(hyper) else hyper


class StcReverseHG(far.ReverseHG):
    """
    Subclass of `far.ReverseHG` that deals also with stochastic hyperparameters
    """

    def __init__(self, history=None, name='ReverseHGPlus'):
        super().__init__(history, name)
        self.samples = None

    @property
    def initialization(self):
        if self._initialization is None:
            additional_initialization = [sample(h) for h in self._hypergrad_dictionary
                                         if is_stochastic_hyper(h)]
            self.samples = additional_initialization
            # noinspection PyStatementEffect
            super(StcReverseHG, self).initialization
            self._initialization.extend(additional_initialization)
        return super(StcReverseHG, self).initialization

    def hgrads_hvars(self, hyper_list=None, aggregation_fn=None, process_fn=None):
        rs = super(StcReverseHG, self).hgrads_hvars(hyper_list, aggregation_fn, process_fn)

        def _ok_or_store_var(hg_hv_pair):
            if is_stochastic_hyper(hg_hv_pair[1]):
                return hg_hv_pair[0], get_probs_var(hg_hv_pair[1])
            return hg_hv_pair

        return [_ok_or_store_var(pair) for pair in rs]

    def run(self, T_or_generator, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, global_step=None, session=None, online=False, callback=None):
        """As in `far.ReverseHG.run`, plus, samples the stochastic hyperparameters at every iterations
        of the inner optimization dynamics"""
        cbk_multi_sample = lambda _t, _fd, _ss: _ss.run(self.samples)
        super().run(T_or_generator, inner_objective_feed_dicts, outer_objective_feed_dicts, initializer_feed_dict,
                    global_step, session, online, [cbk_multi_sample, cbk_multi_sample])

    def hypergradient(self, hyper):
        hg = self._hypergrad_dictionary[hyper]
        return hg[0] if len(hg) == 1 else hg

    @staticmethod
    def _create_hypergradient(outer_obj, hyper):
        doo_dh = tf.gradients(outer_obj, hyper)[0]
        doo_dh = far.utils.val_or_zero(doo_dh, hyper)
        if is_stochastic_hyper(hyper):
            doo_dh = far.utils.maybe_add(doo_dh, tf.gradients(outer_obj, get_probs_var(hyper))[0])
        return far.ReverseHG._create_hypergradient_from_dodh(hyper, doo_dh)

    def min_decrease_condition(self, dec=.001, patience=20, max_iters=1e6,
                               session=None, feed_dicts=None, verbose=False, obj=None, auto=True):
        """Step generator that takes into account the "decrease condition" (of the inner objective) to
        stop inner objective optimization"""
        if obj is None:
            obj = list(self._optimizer_dicts)[0].objective
        if session is None: session = tf.get_default_session()
        res_dic = {'pat': patience, 'min val': None}

        def _gen(p0=patience, val0=None):
            t = 0
            p = p0
            if val0 is None:
                prev_val = val = session.run(obj, feed_dicts)
            else:
                prev_val = val = val0
            while p > 0 and t < max_iters:
                val = session.run(obj, feed_dicts)
                if verbose > 1: print(t, 'min MD condition', prev_val, val, 'pat:', p)
                if prev_val * (1. - dec) < val:
                    p -= 1
                else:
                    p = patience
                    prev_val = val
                yield t
                t += 1
            res_dic.update({'pat': p, 'val': val, 'min val': prev_val, 'tot iter': t})
            if verbose: print(res_dic)

        if auto:
            def _the_gens():
                return _gen(res_dic['pat'], res_dic['min val']), range(int(max_iters) + 1)
        else:
            def _the_gens(p0=patience, val0=None):
                return _gen(p0, val0), range(
                    int(max_iters) + 1)

        return _the_gens, res_dic
