# -*- coding: utf-8 -*-
from numpy import ones, random, unique
from functools import reduce
from sklearn.base import BaseEstimator
from .structure import StaticPeerFixedEffectStructureMixin, CumulativePeerFixedEffectStructureMixin

class PeerFixedEffect(BaseEstimator):
    def __init__(self, effect='static', max_iteration=1000, seed=None, gamma_boundry=0.4, verbose=True):
        super().__init__()
        self.effect = effect
        self.seed = seed
        self.max_iteration = max_iteration
        self.gamma_boundry = gamma_boundry
        self.verbose = verbose
        self.n_time = None
        self.size = None
        self.gamma0 = None
        self.alpha_it0 = None
        self.alpha_it = None

    def initialized_parameter(self, size, init=None):
        if init is None:
            random.seed(self.seed)
            self.gamma0 = min(0.4, random.uniform(low=0.0, high=0.1, size=1)[0])
            self.alpha_it0 = random.random(size=(size, 1)) * 0.1
        else:
            if isinstance(init, dict) is False:
                raise ValueError('please dict')
            if reduce(lambda x, y: x & y, map(lambda x: x in ['gamma0', 'alpha_it0'], list(init.keys()))) is False:
                raise ValueError('init must have gamma0, alpha_it0 keys')
            self.gamma0 = init['gamma0']
            self.alpha_it0 = init['alpha_it0']

    def fit(self, **argv):
        raise NotImplementedError

    def predict(self):
        pass


class StaticPeerFixedEffect(PeerFixedEffect, StaticPeerFixedEffectStructureMixin):
    def fit(self, id_it, time_it, group_it, y_it, init=None):
        self.size = id_it.shape[0]
        self.n_time = len(unique(time_it))
        self.initialized_parameter(size=self.size, init=init)
        gamma0_prime = self.gamma0
        alpha_it0 = self.alpha_it0
        alpha_it = self.get_alpha_it(alpha_it0)
        for iteration in range(self.max_iteration):
            mean_alpha_jt = self.get_mean_alpha_jt(time_it=time_it, group_it=group_it, alpha_it0=alpha_it0)
            gamma0 = self.get_gamma0(y_it=y_it, alpha_it=alpha_it,
                                     mean_alpha_jt=mean_alpha_jt, rho_boundry=self.gamma_boundry)
            alpha_it0 = self.get_new_alpha_it0(n_time=self.n_time, gamma0=gamma0, id_it=id_it, time_it=time_it,
                                               group_it=group_it, y_it=y_it, alpha_it=alpha_it)
            alpha_it = self.get_alpha_it(alpha_it0=alpha_it0)
            print("{q}th iteration: estimated gamma is {gamma0:.4f}".format(q=iteration, gamma0=gamma0))
            if (abs(gamma0 - gamma0_prime) < 10 ** (-5)) & (abs(gamma0) < 0.399):
                break
            else:
                if (self.verbose)&( (iteration + 1) % 100 == 0):
                    print("{q}th iteration: estimated gamma is {gamma0:.4f}".format(q=iteration, gamma0=gamma0))
                gamma0_prime = gamma0
        print("{q}th iteration: estimated gamma is {gamma0:.4f}".format(q=iteration, gamma0=gamma0))
        self.gamma0 = gamma0_prime
        self.alpha_it0 = alpha_it0
        self.alpha_it = alpha_it


class CumulativePeerFixedEffect(PeerFixedEffect, CumulativePeerFixedEffectStructureMixin):
    def fit(self, id_it, time_it, group_it, y_it, init=None):
        self.size = id_it.shape[0]
        self.n_time = len(unique(time_it))
        self.initialized_parameter(size=self.size, init=init)
        gamma0_prime = self.gamma0
        gamma0 = gamma0_prime
        alpha_it0 = self.alpha_it0
        alpha_it = self.get_alpha_it(alpha_it0=alpha_it0, id_it=id_it, group_it=group_it, time_it=time_it, gamma0=gamma0)
        for iteration in range(self.max_iteration):
            mean_alpha_jt = self.get_mean_alpha_jt(time_it=time_it, group_it=group_it, alpha_it0=alpha_it0)
            gamma0 = self.get_gamma0(y_it=y_it, alpha_it=alpha_it,
                                     mean_alpha_jt=mean_alpha_jt, rho_boundry=self.gamma_boundry)
            alpha_it0 = self.get_new_alpha_it0(n_time=self.n_time, gamma0=gamma0, id_it=id_it, time_it=time_it,
                                               group_it=group_it, y_it=y_it, alpha_it=alpha_it)
            alpha_it = self.get_alpha_it(alpha_it0=alpha_it0, id_it=id_it, group_it=group_it, time_it=time_it, gamma0=gamma0)
            print("{q}th iteration: estimated gamma is {gamma0:.4f}".format(q=iteration, gamma0=gamma0))
            if (abs(gamma0 - gamma0_prime) < 10 ** (-5)) & (abs(gamma0) < 0.399):
                break
            else:
                if (self.verbose)&( (iteration + 1) % 100 == 0):
                    print("{q}th iteration: estimated gamma is {gamma0:.4f}".format(q=iteration, gamma0=gamma0))
                gamma0_prime = gamma0
        print("{q}th iteration: estimated gamma is {gamma0:.4f}".format(q=iteration, gamma0=gamma0))
        self.gamma0 = gamma0_prime
        self.alpha_it0 = alpha_it0
        self.alpha_it = alpha_it

