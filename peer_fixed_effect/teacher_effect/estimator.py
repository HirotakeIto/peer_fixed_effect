"""
Notation:
 i: individual student
 j: teacher
 g: grade
 t: time
"""
import pandas as pd
from peer_fixed_effect.teacher_effect.structure import simple
from importlib import reload
from numpy import random, unique, multiply, array
reload(simple)

class BaseEstimator(simple.SimpleTeacherEffectMixin, simple.SimpleIndividualEffectMixin, simple.SimplePersistenceMixin):
    def __init__(self, max_iteration=1000, seed=None, gamma_boundry=0.4, verbose=True, tolerance=10 ** (-4), **argv):
        # sort前提
        # self.df = pd.read_excel('tests/testdata/kinsler.xlsx', sheets='data')
        id_col = 'ids'
        tid_col = 'tid'
        grade_col = 'grade'
        time_col = 'time'
        y_col = 'y'
        eft_i_col = 'effect_i'
        eft_it_col = 'effect_it'
        eft_jt_col = 'effect_tid_t'
        max_grade_col = 'max_grade'
        min_grade_col = 'min_grade'
        sigma = 0.1
        self.seed = seed
        self.max_iteration = max_iteration
        self.gamma_boundry = gamma_boundry
        self.verbose = verbose
        self.tolerance = tolerance
        self.id_col = id_col
        self.tid_col = tid_col
        self.grade_col = 'grade'
        self.time_col = time_col
        self.y_col = y_col
        self.eft_i_col = eft_i_col
        self.eft_it_col = eft_it_col
        self.eft_jt_col = eft_jt_col
        self.max_grade_col = 'max_grade'
        self.min_grade_col = 'min_grade'
        self.sigma = 0.1
        df = (
            pd.read_csv('./test.csv')
            # pd.read_csv('./test_saitama.csv')
                .assign(
                min_grade=lambda dfx: dfx.groupby(self.id_col)[self.grade_col].transform('min'),
                max_grade=lambda dfx: dfx.groupby(self.id_col)[self.grade_col].transform('max')
            )
        )
        self.df = df
        self.initialization()

    @property
    def parameters(self):
        return self.sigma, self.eft_i_col, self.eft_jt_col

    @property
    def parameters_dict(self):
        return {'gamma0': self.sigma, 'eft_i_col':self.eft_i_col, 'eft_jt_col':self.eft_jt_col}

    @property
    def peer_effect(self):
        return self.sigma

    def initialization(self, random_init=False):
        for _ in range(10):
            self.initialize_individual_effect_ijgt(self)
            self.initialize_teacher_effect_ijgt(self)

    def fit(self):
        sigma_prime = 0.5
        for iteration in range(10000):
            sigma = self.estimated_sigma(self, ftol=10**(-3), sigma_init=sigma_prime)
            if abs(sigma_prime - sigma) < self.tolerance:
                print(sigma)
                break
            else:
                self.sigma = sigma
                sigma_prime = sigma
            # print(self.sigma)
            self.df[self.eft_it_col] = self.estimate_individual_effect_ijgt(self)
            self.df[self.eft_jt_col] = self.estimate_teacher_effect_ijgt(self)
            if self.verbose & (iteration % 10 == 3):
                print("{q}th iteration: estimated gamma is {sigma:.4f}".format(q=iteration, sigma=self.sigma))

#

#
be = BaseEstimator()
be.fit()

# df = pd.read_csv('./tests/data_tmp.csv')
# df_tmp = (
#     df
#     [['zmath_level', 'school_id', 'grade', 'year', 'mst_id']]
#     .reset_index(drop=True)
#     .rename(
#         columns = {'zmath_level': 'y', 'school_id': 'tid', 'mst_id': 'ids',
#           'year': 'time'}
#     )
# )
# df_tmp.to_csv('test_saitama.csv', index=False)


# # id_col = 'ids'
# tid_col = 'tid'
# grade_col = 'grade'
# time_col = 'time'
# y_col = 'y'
# eft_i_col = 'effect_i'
# eft_it_col = 'effect_it'
# eft_jt_col = 'effect_tid_t'
# df = (
#     pd.read_csv('./test.csv')
#     .assign(
#         min_grade = lambda dfx: dfx.groupby(id_col)[grade_col].transform('min'),
#         max_grade =  lambda dfx: dfx.groupby(id_col)[grade_col].transform('max')
#     )
# )
# df[eft_it_col] = random.uniform(0, 1, size=df.shape[0])
# df[eft_jt_col] = random.uniform(0, 1, size=df.shape[0])
# df[eft_jt_col] = df.groupby([tid_col, grade_col])[y_col].transform('mean')
# df[eft_it_col] = df[y_col] - df[eft_jt_col]
# df[eft_it_col] = df.groupby([id_col])[eft_it_col].transform('mean')
# sigma = 0
# for _ in range(10):
#     df[eft_jt_col] = df[y_col] - df[eft_it_col]
#     df[eft_jt_col] = df.groupby([tid_col, grade_col])[eft_jt_col].transform('mean')
#     df[eft_it_col] = df[y_col] - df[eft_jt_col]
#     df[eft_it_col] = df.groupby([id_col])[eft_it_col].transform('mean')


# from sklearn.preprocessing import OneHotEncoder
# from sklearn.linear_model import LinearRegression
# enc = OneHotEncoder(handle_unknown='ignore')
# lr = LinearRegression(fit_intercept=False)
# x_onehot = enc.fit_transform(df[[tid_col, grade_col]].values).toarray()
# lr.fit(y=df[y_col].values, X=x_onehot)
# lr.coef_
# x_onehot = enc.fit_transform(df[[id_col]].values).toarray()
# lr.fit(y=df[y_col].values, X=x_onehot)
# lr.coef_



# # # %%memit
# import pandas as pd
# df = pd.read_csv('./test.csv')
# aa = df.groupby('ids')
# df['wei'] = 1
# df['ass'] = df['wei'] * df['tid']
# aa['ass'].sum()
# #
# #
# #
# #
# #
# # aa.groups
# # df.groupby(aa.groups)['tid'].sum()
# # aa = df.groupby(['ids', 'tid'])
# aa.pipe(lambda grp: grp.apply(lambda dfx: dfx.index))