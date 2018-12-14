# -*- coding: utf-8 -*-
from numpy import ones, random, unique
from sklearn.base import BaseEstimator
from .structure import StaticPeerFixedEffectStructureMixin, CumulativePeerFixedEffectStructureMixin

class PeerFixedEffectFixedEffect(BaseEstimator):
    def __init__(self, effect='static', max_iteration=1000, seed=None, rho_boundry=0.4, verbose=True):
        super().__init__()
        self.effect = effect
        self.seed = seed
        self.max_iteration = max_iteration
        self.rho_boundry = rho_boundry
        self.verbose = verbose
        self.n_time = None
        self.size = None
        self.gamma0 = None
        self.alpha_it0 = None
        self.alpha_it = None

    def initialized_parameter(self, size):
        random.seed(self.seed)
        self.gamma0 = min(0.4, random.uniform(low=0.0, high=0.4, size=1)[0])
        self.alpha_it0 = ones(shape=(size, 1)) * random.random(size=1)[0]

    def fit(self, **argv):
        raise NotImplementedError

    def predict(self):
        pass


class StaticPeerFixedEffectFixedEffect(PeerFixedEffectFixedEffect, StaticPeerFixedEffectStructureMixin):
    def fit(self, id_it, time_it, group_it, y_it):
        self.size = id_it.shape[0]
        self.n_time = len(unique(time_it))
        self.initialized_parameter(size=self.size)
        gamma0_prime = self.gamma0
        alpha_it0 = self.alpha_it0
        alpha_it = self.get_alpha_it(alpha_it0)
        for iteration in range(self.max_iteration):
            mean_alpha_jt = self.get_mean_alpha_jt(time_it=time_it, group_it=group_it, alpha_it0=alpha_it0)
            gamma0 = self.get_gamma0(y_it=y_it, alpha_it=alpha_it,
                                     mean_alpha_jt=mean_alpha_jt, rho_boundry=self.rho_boundry)
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


class CumulativePeerFixedEffectFixedEffect(PeerFixedEffectFixedEffect, CumulativePeerFixedEffectStructureMixin):
    def fit(self, id_it, time_it, group_it, y_it):
        self.size = id_it.shape[0]
        self.n_time = len(unique(time_it))
        self.initialized_parameter(size=self.size)
        gamma0_prime = self.gamma0
        gamma0 = gamma0_prime
        alpha_it0 = self.alpha_it0
        alpha_it = self.get_alpha_it(alpha_it0=alpha_it0, id_it=id_it, group_it=group_it, time_it=time_it, gamma0=gamma0)
        for iteration in range(self.max_iteration):
            mean_alpha_jt = self.get_mean_alpha_jt(time_it=time_it, group_it=group_it, alpha_it0=alpha_it0)
            gamma0 = self.get_gamma0(y_it=y_it, alpha_it=alpha_it,
                                     mean_alpha_jt=mean_alpha_jt, rho_boundry=self.rho_boundry)
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

