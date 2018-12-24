import numpy as np
from pandas import DataFrame
from numpy import random, kron, c_
from functools import reduce
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


class SampleBase(MixinRandomSample):
    def __init__(self, n_individual=10000, n_group=10, n_time=4, n_characteristics=3, seed=None,
                 x_it_argv=None, fixed_effect_it_argv=None, resid_it_argv=None,
                 course_or_not=False, n_course=3):
        self.n_individual = n_individual
        self.n_group = n_group
        self.n_time = n_time
        self.n_characteristics = n_characteristics
        self.x_it_argv = {} if x_it_argv is None else x_it_argv
        self.fixed_effect_it_argv = {} if fixed_effect_it_argv is None else fixed_effect_it_argv
        self.resid_it_argv = {} if resid_it_argv is None else resid_it_argv
        self.course_or_not = course_or_not
        self.n_course = n_course
        self.data = None
        if seed is not None:
            random.seed(seed)
        self._initialized_model_parametor()

    def _initialized_model_parametor(self, dist='uniform', **argv):
        self.beta1 = random.uniform(size=(self.n_characteristics, 1))
        self.beta2 = random.uniform(size=1)[0]
        rho0_tmp = random.uniform(low=0.0, high=0.4, size=1)[0]
        if dist != 'uniform':
            # ここのパラメータを自動化するニーズは特にないと思うので一旦後回し
            rho0_tmp = self.sample(size=1, dist=dist, **argv)[0]
        self.rho0 = min(0.4, rho0_tmp)

    def dump_parametor(self):
        return json.dumps(
            obj={'beta1': self.beta1.tolist(), 'beta2': self.beta2.tolist(), 'rho0': self.rho0.tolist()},
            indent=True
        )

    def build(self):
        self.data = self._build()

    def _build(self, **argv):
        raise NotImplementedError('Please override')

    def id_it(self, n_individual, n_time):
        return kron(np.arange(0, n_individual).reshape((n_individual, 1)), np.ones(shape=(n_time, 1)))

    def time_it(self, n_individual, n_time):
        return kron(np.ones(shape=(n_individual, 1)), np.arange(0, n_time).reshape((n_time, 1)))

    def group_it(self, n_individual, n_group, n_time):
        return random.randint(low=0, high=n_group, size=(n_individual * n_time, 1))

    def x_it(self, n_individual, n_characteristics, n_time, dist='standard', **argv):
        x_i0 = self.sample(dist=dist, size=(n_individual, n_characteristics), **argv)
        return kron(x_i0, np.ones(shape=(n_time, 1)))

    def fixed_effect_it(self, n_individual, n_time, dist='standard', **argv):
        fixed_effect_i0 = self.sample(size=(n_individual, 1), dist=dist, **argv)
        return kron(fixed_effect_i0, np.ones(shape=(n_time, 1)))

    def course_it(self, time_it, group_it, n_course):
        course_effect = (
            DataFrame(
                c_[time_it, group_it],
                columns=['time', 'group'])
            .assign(
                course=1)
            .groupby(['time', 'group'])
            [['course']]
            .transform(
                lambda x: x * random.randint(low=0, high=n_course, size=1)[0])
        )
        return course_effect

    def course_effect_it(self, time_it, course_it, dist='standard', **argv):
        array = (
            DataFrame(c_[time_it, course_it], columns=['time', 'course'])
            .assign(val = 1)
            .groupby(['time', 'course'])
            [['val']]
            .transform(
                lambda x: x * self.sample(size=1, dist=dist, **argv)[0]
            )
            .values
        )
        return array

    def alpha_it0(self, x_it0, beta1, fixed_effect_it0, beta2):
        return x_it0.dot(beta1) + fixed_effect_it0.dot(beta2)

    def alpha_it(self, **argv):
        raise NotImplementedError('Please override')

    def alpha_jt(self, group_it, time_it, alpha_it0, rho0):
        df = DataFrame(c_[group_it, time_it, alpha_it0], columns=['group', 'time', 'alphait'])
        df['mean_alphajt'] = df.groupby(['group', 'time'])['alphait'].transform(
            lambda x: (x.sum() - x) / (x.count() - 1))
        return rho0 * df[['mean_alphajt']].values

    def resid_it(self, n_individual, n_time, dist='standard', **argv):
        return self.sample(size=(n_individual * n_time, 1), dist=dist, **argv) * 0.01

    # def y_it(self, aplha_it, aplha_jt, resid_it):
    #     return aplha_it + aplha_jt + resid_it

    def y_it(self, *argv):
        return reduce(lambda x, y: x + y, argv)


class StaticSample(SampleBase):
    def _build(self):
        id_it = self.id_it(self.n_individual, self.n_time)
        group_it = self.group_it(self.n_individual, self.n_group, self.n_time)
        time_it = self.time_it(self.n_individual, self.n_time)
        x_it = self.x_it(self.n_individual, self.n_characteristics, self.n_time, **self.x_it_argv)
        fixed_effect_it = self.fixed_effect_it(self.n_individual, self.n_time, **self.fixed_effect_it_argv)
        alpha_it0 = self.alpha_it0(x_it, self.beta1, fixed_effect_it, self.beta2)
        alpha_it = self.alpha_it(alpha_it0)
        alpha_jt = self.alpha_jt(group_it, time_it, alpha_it, rho0=self.rho0)
        resid_it = self.resid_it(self.n_individual, self.n_time, **self.resid_it_argv)
        if self.course_or_not is False:
            yit = self.y_it(alpha_it, alpha_jt, resid_it)
            return DataFrame(
                c_[id_it, group_it, time_it, fixed_effect_it, alpha_it0, alpha_it, alpha_jt, resid_it, yit, x_it],
                columns=['ids', 'group', 'time', 'fixed_effect_it', 'alpha_it0', 'alpha_it', 'alpha_jt', 'resid_it',
                         'yit'] + ['xit' + str(i) for i in range(x_it.shape[1])]
            )
        else:
            course_it = self.course_it(time_it=time_it, group_it=group_it, n_course=self.n_course)
            course_effect_it = self.course_effect_it(time_it, course_it)
            yit = self.y_it(alpha_it, alpha_jt, course_effect_it, resid_it)
            return DataFrame(
                c_[id_it, group_it, time_it, fixed_effect_it, alpha_it0, alpha_it,
                   alpha_jt, course_it, course_effect_it, resid_it, yit, x_it],
                columns=['ids', 'group', 'time', 'fixed_effect_it', 'alpha_it0', 'alpha_it', 'alpha_jt',
                         'course_it', 'course_effect_it', 'resid_it', 'yit']
                        + ['xit' + str(i) for i in range(x_it.shape[1])]
            )

    def alpha_it(self, alpha_it0):
        return alpha_it0


class CumulativeSample(SampleBase):
    def _build(self):
        id_it = self.id_it(self.n_individual, self.n_time)
        group_it = self.group_it(self.n_individual, self.n_group, self.n_time)
        time_it = self.time_it(self.n_individual, self.n_time)
        x_it = self.x_it(self.n_individual, self.n_characteristics, self.n_time, **self.x_it_argv)
        fixed_effect_it = self.fixed_effect_it(self.n_individual, self.n_time, **self.fixed_effect_it_argv)
        alpha_it0 = self.alpha_it0(x_it, self.beta1, fixed_effect_it, self.beta2)
        alpha_it = self.alpha_it(alpha_it0, id_it, group_it, time_it, self.rho0)
        alpha_jt = self.alpha_jt(group_it, time_it, alpha_it, rho0=self.rho0)
        resid_it = self.resid_it(self.n_individual, self.n_time, **self.resid_it_argv)
        if self.course_or_not is False:
            yit = self.y_it(alpha_it, alpha_jt, resid_it)
            return DataFrame(
                c_[id_it, group_it, time_it, fixed_effect_it, alpha_it0, alpha_it, alpha_jt, resid_it, yit, x_it],
                columns=['ids', 'group', 'time', 'fixed_effect_it', 'alpha_it0', 'alpha_it', 'alpha_jt', 'resid_it',
                         'yit'] + ['xit' + str(i) for i in range(x_it.shape[1])]
            )
        else:
            course_it = self.course_it(time_it=time_it, group_it=group_it, n_course=self.n_course)
            course_effect_it = self.course_effect_it(course_it)
            yit = self.y_it(alpha_it, alpha_jt, course_effect_it, resid_it)
            return DataFrame(
                c_[id_it, group_it, time_it, fixed_effect_it, alpha_it0, alpha_it,
                   alpha_jt, course_it, course_effect_it, resid_it, yit, x_it],
                columns=['ids', 'group', 'time', 'fixed_effect_it', 'alpha_it0', 'alpha_it', 'alpha_jt',
                         'course_it', 'course_effect_it', 'resid_it', 'yit']
                        + ['xit' + str(i) for i in range(x_it.shape[1])]
            )

    def alpha_it(self, alpha_it0, id_it, group_it, time_it, gamma0):
        def set_initial_alphait(df, start_time):
            df.loc[df['time'] == start_time, 'alpha_it'] = df['alpha_it0']
            return df

        def mean_alphajt_1(df_specific_time):
            return df_specific_time.groupby(['group_t_1'])['alphait_1'].transform(
                lambda x: (x.sum() - x) / (x.count() - 1))

        def set_alpha_it(df, set_time_list):
            # id方向とtime方向に順序がaccendingになっている必要がある(ここでチェックはしない)
            for time in set_time_list:
                df['alphait_1'] = df.groupby(['ids'])['alpha_it'].shift(1)
                df['group_t_1'] = df.groupby(['ids'])['group'].shift(1)
                slicing = df['time'] == time
                df.loc[slicing, 'mean_alphajt_1'] = mean_alphajt_1(df[slicing])
                df.loc[slicing, 'alpha_it'] = df.loc[slicing, 'alphait_1'] + gamma0 * df.loc[slicing, 'mean_alphajt_1']
            return df

        time_list = sorted(np.unique(time_it).tolist())
        df = (
            DataFrame(c_[id_it, group_it, time_it, alpha_it0], columns=['ids', 'group', 'time', 'alpha_it0'])
            .sort_values(['ids', 'time'])
            .pipe(set_initial_alphait, start_time=time_list[0])
            .pipe(set_alpha_it, set_time_list=time_list[1:])
            .sort_index()
        )
        alpha_it = df[['alpha_it']].values
        return alpha_it