import numpy as np
from pandas import DataFrame
from numpy import random
from numpy import kron, c_
import json

class MixinRandomSample:
    def sample(self, dist, size, **argv) -> np.array:
        if dist == 'uniform':
            return self.numpy_random_function(random.uniform, size=size, **argv)
        if dist == 'standard':
            return self.numpy_random_function(random.standard_normal, size=size)
        if dist == 'normal':
            return self.numpy_random_function(random.normal, size=size, **argv)
        else:
            raise LookupError("""
            {0} isn\'t supported. Our support function is  'uniform', 'standard', 'normal'.
            """.format(dist))

    @staticmethod
    def numpy_random_function(func, size, **argv):
        return func(size=size, **argv)


class Sample(MixinRandomSample):
    def __init__(self, n_individual=10000, n_group=10, n_time=4, n_characteristics=3, seed=None,
                 x_it_argv=None, fixed_effect_it_argv=None, resid_it_argv=None):
        self.n_individual = n_individual
        self.n_group = n_group
        self.n_time = n_time
        self.n_characteristics = n_characteristics
        self.x_it_argv = {} if x_it_argv is None else x_it_argv
        self.fixed_effect_it_argv = {} if fixed_effect_it_argv is None else fixed_effect_it_argv
        self.resid_it_argv = {} if resid_it_argv is None else resid_it_argv
        if seed is not None:
            random.seed(seed)
        self._initialized_model_parametor()

    def _initialized_model_parametor(self, dist='uniform', **argv):
        self.beta1 = random.uniform(size=(self.n_characteristics, 1))
        self.beta2 = random.uniform(size=(1, 1))
        rho0_tmp = random.uniform(low=0.0, high=0.4, size=1)[0]
        if dist != 'uniform':
            # ここのパラメータを自動化するニーズは特にないと思うので一旦後回し
            rho0_tmp = self.sample(size=1, dist=dist, **argv)[0]
        self.rho0 = np.array([[min(0.4, rho0_tmp)]])

    def build(self):
        self._build()

    def _build(self):
        id_it = self.id_it(self.n_individual, self.n_time)
        group_it = self.group_it(self.n_individual, self.n_group, self.n_time)
        time_it = self.time_it(self.n_individual, self.n_time)
        x_it = self.x_it(self.n_individual, self.n_characteristics, self.n_time, **self.x_it_argv)
        fixed_effect_it = self.fixed_effect_it(self.n_individual, self.n_time, **self.fixed_effect_it_argv)
        alpha_it0 = self.alpha_it0(x_it, self.beta1, fixed_effect_it, self.beta2)
        alpha_jt0 = self.alpha_jt0(group_it, time_it, alpha_it0, rho0=self.rho0)
        resid_it = self.resid_it(self.n_individual, self.n_time, **self.resid_it_argv)
        yit = self.y_it(alpha_it0, alpha_jt0, resid_it)
        self.data = DataFrame(
            c_[id_it, group_it, time_it, fixed_effect_it, alpha_it0, alpha_jt0, resid_it, yit, x_it],
            columns=['ids', 'group', 'time', 'fixed_effect_it', 'alpha_it0', 'alpha_jt0', 'resid_it', 'yit'] + ['xit' + str(i) for i in range(x_it.shape[1])]
        )

    def id_it(self, n_individual, n_time):
        return kron(np.arange(0, n_individual).reshape((n_individual, 1)), np.ones(shape=(n_time, 1)))

    def time_it(self, n_individual, n_time):
        return kron(np.ones(shape=(n_individual, 1)), np.arange(0, n_time).reshape((n_time, 1)))

    def group_it(self, n_individual, n_group, n_time):
        return random.randint(low=0, high=n_group, size=(n_individual * n_time, 1))

    def x_it(self, n_individual, n_characteristics, n_time, dist='standard', **argv):
        x_i = self.sample(dist=dist, size=(n_individual, n_characteristics), **argv)
        return kron(x_i, np.ones(shape=(n_time, 1)))

    def fixed_effect_it(self, n_individual, n_time, dist='standard', **argv):
        fixed_effect_i = self.sample(size=(n_individual, 1), dist=dist, **argv)
        return kron(fixed_effect_i, np.ones(shape=(n_time, 1)))

    def resid_it(self, n_individual, n_time, dist='standard', **argv):
        return self.sample(size=(n_individual * n_time, 1), dist=dist, **argv) * 0.01

    def alpha_it0(self, x_it, beta1, fixed_effect_it, beta2):
        return x_it.dot(beta1) + fixed_effect_it.dot(beta2)

    def alpha_jt0(self, group_it, time_it, aplha_it0, rho0):
        df = DataFrame(c_[group_it, time_it, aplha_it0], columns=['group', 'time', 'alphait'])
        df['mean_alphajt'] = df.groupby(['group', 'time'])['alphait'].transform(
            lambda x: (x.sum() - x) / (x.count() - 1))
        return rho0 * df[['mean_alphajt']].values

    def y_it(self, aplha_it0, aplha_jt0, resid_it):
        return aplha_it0 + aplha_jt0 + resid_it
