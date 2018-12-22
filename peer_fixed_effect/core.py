# -*- coding: utf-8 -*-
from numpy import random, unique, multiply, array
from sklearn.base import BaseEstimator
from .structure import StaticPeerFixedEffectStructureMixin, CumulativePeerFixedEffectStructureMixin


class PeerEffectEstimator:
    def __init__(self, max_iteration=1000, seed=None, gamma_boundry=0.4, verbose=True):
        super().__init__()
        self.seed = seed
        self.max_iteration = max_iteration
        self.gamma_boundry = gamma_boundry
        self.verbose = verbose

    @property
    def parameters(self):
        return self.gamma0, self.alpha_it0, self.alpha_it

    @property
    def parameters_dict(self):
        return {'gamma0': self.gamma0, 'alpha_it0':self.alpha_it0, 'alpha_it':self.alpha_it}

    @property
    def peer_effect(self):
        return self.gamma0

    def _set_parameters(self, gamma0, alpha_it0, alpha_it):
        self.gamma0 = gamma0
        self.alpha_it0 = alpha_it0
        self.alpha_it = alpha_it

    def _initialized_parameter(self, size, init=None):
        if init is None:
            random.seed(self.seed)
            self.gamma0 = min(0.4, random.uniform(low=0.0, high=0.1, size=1)[0])
            self.alpha_it0 = random.random(size=(size, 1)) * 0.1
        else:
            if isinstance(init, dict) is False:
                raise ValueError('please dict')
            # if reduce(lambda x, y: x & y, map(lambda x: x in ['gamma0', 'alpha_it0'], list(init.keys()))) is False:
            for x in ['gamma0', 'alpha_it0']:
                if x not in init.keys():
                    raise ValueError('init must have {x} keys'.format(x=x))
            self.gamma0 = init['gamma0']
            self.alpha_it0 = init['alpha_it0']

    def fit(self, **argv):
        raise NotImplementedError('You must call abstract method.')

    def predict(self, **argv):
        raise NotImplementedError('You must call abstract method.')


class StaticPeerFixedEffectEstimator(PeerEffectEstimator, StaticPeerFixedEffectStructureMixin):
    def fit(self, reg_cls):
        gamma0, alpha_it0, alpha_it = self._fit(
            id_it=reg_cls.id_it, time_it=reg_cls.time_it, group_it=reg_cls.group_it, y_it=reg_cls.y_it,
            init = reg_cls.init,
        )
        self._set_parameters(gamma0, alpha_it0, alpha_it)

    def _fit(self,
             id_it, time_it, group_it, y_it,
             init):
        size = id_it.shape[0]
        n_time = len(unique(time_it))
        self._initialized_parameter(size=size, init=init)
        gamma0_prime = self.gamma0
        alpha_it0 = self.alpha_it0
        alpha_it = self.get_alpha_it(alpha_it0)
        for iteration in range(self.max_iteration):
            mean_alpha_jt = self.get_mean_alpha_jt(time_it=time_it, group_it=group_it, alpha_it=alpha_it0)
            gamma0 = self.get_gamma0(y_it=y_it, alpha_it=alpha_it,
                                     mean_alpha_jt=mean_alpha_jt, gamma_boundry=self.gamma_boundry)
            alpha_it0 = self.get_new_alpha_it0(n_time=n_time, gamma0=gamma0, id_it=id_it, time_it=time_it,
                                               group_it=group_it, y_it=y_it, alpha_it=alpha_it)
            alpha_it = self.get_alpha_it(alpha_it0=alpha_it0)
            if (abs(gamma0 - gamma0_prime) < 10 ** (-5)) & (abs(gamma0) < self.gamma_boundry):
                break
            else:
                if self.verbose&((iteration + 1) % 10 == 0):
                    print("{q}th iteration: estimated gamma is {gamma0:.4f}".format(q=iteration, gamma0=gamma0))
                gamma0_prime = gamma0
        return gamma0_prime, alpha_it0, alpha_it

    def predict(self, reg_cls):
        return self._predict(
            time_it=reg_cls.time_it, group_it=reg_cls.group_it,
            gamma0=reg_cls.gamma0, alpha_it0=reg_cls.alpha_it0
        )

    def _predict(self, time_it, group_it, gamma0, alpha_it0):
        alpha_it = self.get_alpha_it(alpha_it0)
        mean_alpha_jt = self.get_mean_alpha_jt(group_it, time_it, alpha_it)
        return alpha_it + gamma0 * mean_alpha_jt


class CumulativePeerFixedEffectEstimator(PeerEffectEstimator, CumulativePeerFixedEffectStructureMixin):
    def fit(self, reg_cls):
        gamma0, alpha_it0, alpha_it = self._fit(
            id_it=reg_cls.id_it, time_it=reg_cls.time_it, group_it=reg_cls.group_it, y_it=reg_cls.y_it,
            init = reg_cls.init,
        )
        self._set_parameters(gamma0, alpha_it0, alpha_it)

    def _fit(self, id_it, time_it, group_it, y_it, init=None):
        size = id_it.shape[0]
        n_time = len(unique(time_it))
        self._initialized_parameter(size=size, init=init)
        gamma0_prime = self.gamma0
        gamma0 = gamma0_prime
        alpha_it0 = self.alpha_it0
        alpha_it = self.get_alpha_it(alpha_it0=alpha_it0, id_it=id_it, group_it=group_it, time_it=time_it, gamma0=gamma0)
        for iteration in range(self.max_iteration):
            # TODO: 個々の時系列注意！（fixedeffect使うのか、蓄積データ使うのか。）
            mean_alpha_jt = self.get_mean_alpha_jt(time_it=time_it, group_it=group_it, alpha_it=alpha_it)
            gamma0 = self.get_gamma0(y_it=y_it, alpha_it=alpha_it,
                                     mean_alpha_jt=mean_alpha_jt, gamma_boundry=self.gamma_boundry)
            alpha_it0 = self.get_new_alpha_it0(n_time=n_time, gamma0=gamma0, id_it=id_it, time_it=time_it,
                                               group_it=group_it, y_it=y_it, alpha_it=alpha_it)
            alpha_it = self.get_alpha_it(alpha_it0=alpha_it0, id_it=id_it, group_it=group_it, time_it=time_it, gamma0=gamma0)
            if (abs(gamma0 - gamma0_prime) < 10 ** (-5)) & (abs(gamma0) < self.gamma_boundry):
                break
            else:
                if self.verbose&( (iteration + 1) % 10 == 0):
                    print("{q}th iteration: estimated gamma is {gamma0:.4f}".format(q=iteration, gamma0=gamma0))
                gamma0_prime = gamma0
        return gamma0_prime, alpha_it0, alpha_it


class PeerFixedEffectRegression(BaseEstimator):
    estimator_mapper = {
        'static': StaticPeerFixedEffectEstimator,
        'cumulative': CumulativePeerFixedEffectEstimator
    }

    def __init__(self, effect='static', max_iteration=1000, seed=None, gamma_boundry=0.4, verbose=True):
        super().__init__()
        self.seed = seed
        self.max_iteration = max_iteration
        self.gamma_boundry = gamma_boundry
        self.verbose = verbose
        self.id_it = None
        self.time_it = None
        self.y_it = None
        self.delta_it = None
        self.init = None
        self.estimator = self.estimator_mapper[effect](
            max_iteration=max_iteration, seed=seed, gamma_boundry=gamma_boundry, verbose=verbose)
        self.peer_effect_sample = None

    @property
    def peer_effect_parametor(self):
        if self.peer_effect_sample is None:
            return self.estimator.peer_effect
        if self.peer_effect_sample is not None:
            return array(self.peer_effect_sample).mean(), array(self.peer_effect_sample).std()

    def set_attribute(self, **params):
        for key, value in params.items():
            self.__setattr__(key, value)

    def fit(self, id_it, time_it, group_it, y_it, delta_it=None, init=None, bootstrap=False):
        self.set_attribute(id_it=id_it, time_it=time_it, group_it=group_it, y_it=y_it, delta_it=delta_it, init=init)
        self.estimator.fit(reg_cls=self)
        self.set_attribute(**self.estimator.parameters_dict)
        if bootstrap is True:
            self.set_attribute(init=self.estimator.parameters_dict)
            peer_effect_value = []
            for i in range(100):
                resid_it = y_it - self.predict()
                y_it_updated = y_it + multiply(resid_it, random.standard_normal(size=resid_it.shape))
                self.set_attribute(y_it=y_it_updated)
                self.estimator.fit(reg_cls=self)
                peer_effect_value.append(self.estimator.peer_effect)
                self.set_attribute(**self.estimator.parameters_dict)
            self.peer_effect_sample = peer_effect_value

    def predict(self):
        return self.estimator.predict(reg_cls=self)
