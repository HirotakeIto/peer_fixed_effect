
import pandas as pd
from peer_fixed_effect.teacher_effect.model import simple
from importlib import reload
from sklearn.base import BaseEstimator
reload(simple)

class SimpleRegression(BaseEstimator):
    model = simple.SimpleModel

    def __init__(self, max_iteration=1000, seed=None, verbose=True, tolerance=10 ** (-8), random_init=False):
        self.seed = seed
        self.max_iteration = max_iteration
        self.verbose = verbose
        self.tolerance = tolerance
        self.random_init = random_init
        self.parameters_dict_bootstraped = []

    def fit(self, df, id_col, tid_col, grade_col, y_col, eft_it_col, eft_jt_col, bootstrap=None, **argv):
        if bootstrap is None:
            self.mo = self.model(
                df, id_col, tid_col, grade_col, y_col, eft_it_col, eft_jt_col,
                max_iteration=self.max_iteration, seed=self.seed,
                verbose=self.verbose, tolerance=self.tolerance, random_init=self.random_init,
                **argv
            )
            self.mo.fit()
        else:
            for _ in range(bootstrap):
                self.mo = self.model(
                    df, id_col, tid_col, grade_col, y_col, eft_it_col, eft_jt_col,
                    max_iteration=self.max_iteration, seed=self.seed,
                    verbose=self.verbose, tolerance=self.tolerance, random_init=self.random_init
                )
                self.mo.fit()
                self.parameters_dict_bootstraped.append(self.mo.parameters_dict)



class SimpleFixedRegression(BaseEstimator):
    model = simple.SimpleFixedModel

    def __init__(self, max_iteration=1000, seed=None, verbose=True, tolerance=10 ** (-8), random_init=False):
        self.seed = seed
        self.max_iteration = max_iteration
        self.verbose = verbose
        self.tolerance = tolerance
        self.random_init = random_init
        self.parameters_dict_bootstraped = []

    def fit(self, df, id_col, tid_col, grade_col, y_col, eft_it_col, eft_jt_col, bootstrap=None, **argv):
        if bootstrap is None:
            self.mo = self.model(
                df, id_col, tid_col, grade_col, y_col, eft_it_col, eft_jt_col,
                max_iteration=self.max_iteration, seed=self.seed,
                verbose=self.verbose, tolerance=self.tolerance, random_init=self.random_init,
                **argv
            )
            self.mo.fit()
        else:
            for _ in range(bootstrap):
                self.mo = self.model(
                    df, id_col, tid_col, grade_col, y_col, eft_it_col, eft_jt_col,
                    max_iteration=self.max_iteration, seed=self.seed,
                    verbose=self.verbose, tolerance=self.tolerance, random_init=self.random_init
                )
                self.mo.fit()
                self.parameters_dict_bootstraped.append(self.mo.parameters_dict)



# df = pd.read_csv('./test.csv')
# df = pd.read_csv('./test_saitama.csv')
# sr = SimpleRegression(random_init = False, tolerance=10**(-2))
# sr.fit(
#     df=df.dropna().reset_index(drop=True),
#     id_col = 'ids',
#     tid_col = 'tid',
#     grade_col = 'grade',
#     y_col = 'y',
#     eft_it_col = 'effect_it',
#     eft_jt_col = 'effect_tid_t',
#     bootstrap=None
# )

# df = pd.read_csv('./test2.csv')
# sr = SimpleFixedRegression(random_init = False, tolerance=10**(-2))
# sr.fit(
#     df=df.dropna().reset_index(drop=True),
#     id_col = 'ids',
#     tid_col = 'tid',
#     grade_col = 'grade',
#     y_col = 'y',
#     eft_it_col = 'effect_it',
#     eft_jt_col = 'effect_tid_t',
#     bootstrap=None
# )
# sr.mo.persistence
# sr.parameters_dict_bootstraped

#
# df = (
#     pd.read_csv('memo/data/toda_kyouin.csv')
#     .dropna(subset=['mst_id', 'grade_prime', 'year_prime', target])
#     .dropna(subset=['teacher_id'])
#     .assign(
#         iddd=lambda dfx: dfx.year_prime.astype(int).astype(str) + dfx.grade_prime.astype(int).astype(str)
#     )
#     .assign(
#         n_t=lambda dfx: dfx.groupby(['teacher_id'])['iddd'].transform('nunique')
#     )
# )
# df.groupby('teacher_id')['iddd'].nunique().value_counts()
#
# import random, string
# def give_sample_id(dfx):
#     ids_choice = pd.Series(
#         pd.np.random.choice(dfx['mst_id'].unique(), size=len(dfx['mst_id'].unique()))
#     ).value_counts()
#     dfxx = pd.DataFrame()
#     for i in range(10000):
#         ids_extract = ids_choice[ids_choice > i].index
#         if ids_extract.shape[0] == 0:
#             break
#         df_extract = (
#             dfx.loc[dfx.mst_id.isin(ids_extract), :]
#             .assign(
#                 mst_id = lambda dfx: dfx['mst_id'] + ''.join(random.sample(string.ascii_lowercase, k=i))
#             )
#         )
#         dfxx = dfxx.append(df_extract)
#     return dfxx
#
# import pandas as pd
# target = 'zkokugo_level'
# df = (
#     pd.read_csv('memo/data/toda_kyouin.csv')
#     # .pipe(give_missing_teacher_id)
#     .pipe(lambda dfx: dfx.loc[dfx['grade_prime'] <= 6, :])
#     .pipe(lambda dfx: dfx.loc[dfx['grade_prime'] >= 3, :])
#     .dropna(subset=['mst_id', 'grade_prime', 'year_prime', target])
#     .dropna(subset=['teacher_id'])
#     .assign(
#         iddd=lambda dfx: dfx.year_prime.astype(int).astype(str) + dfx.grade_prime.astype(int).astype(str)
#     )
#     .sort_values(['mst_id', 'grade_prime'])
#     .assign(
#         n_t = lambda dfx: dfx.groupby(['teacher_id'])['iddd'].transform('nunique')
#     )
#     # .pipe(lambda dfx: dfx.loc[dfx['n_t'] >= 2, :])
#     .pipe(give_sample_id)
#     # .assign(
#     #     n=lambda dfx: dfx.groupby(['mst_id'])['mst_id'].transform('size')
#     # )
#     # .pipe(lambda dfx: dfx.loc[dfx['n'] >1, :])
#     [['mst_id', 'year_prime', 'grade_prime', 'teacher_id', target]]
#     .reset_index(drop=True)
# )
# print(
#     df.groupby(['year_prime', 'grade_prime'])[['zselfefficacy', 'zselfcontrol', 'zdilligence', 'zkokugo_level']].mean())
# print(df.groupby(['teacher_id'])['iddd'].nunique().mean())
# tids = df.groupby(['teacher_id'])['iddd'].nunique()
# tids_included = tids[tids > 1].index
# # df = df.loc[df['teacher_id'].isin(tids_included)]

# sr = SimpleRegression(random_init=False, max_iteration=5000, tolerance=10**(-3))
# sr.fit(
#     df=df,
#     id_col='mst_id',
#     tid_col='teacher_id',
#     grade_col='grade_prime',
#     y_col=target,
#     eft_it_col='effect_it',
#     eft_jt_col='effect_tid_t',
#     bootstrap=None,
#     init_sigma=None,
# )
# sr.mo.parameters_dict
# sr.mo.parameters_dict['eft_jt_col'].describe()
# sr.mo.parameters_dict['eft_it_col'].describe()

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
#

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