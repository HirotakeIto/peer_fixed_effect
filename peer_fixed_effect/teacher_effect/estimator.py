
import pandas as pd
from peer_fixed_effect.teacher_effect.model import simple
from importlib import reload
from sklearn.base import BaseEstimator
reload(simple)

class SimpleRegression(BaseEstimator):
    model = simple.SimpleModel

    def __init__(self, max_iteration=1000, seed=None, verbose=True, tolerance=10 ** (-4)):
        self.seed = seed
        self.max_iteration = max_iteration
        self.verbose = verbose
        self.tolerance = tolerance
        self.parameters_dict_bootstraped = []

    def fit(self, df, id_col, tid_col, grade_col, y_col, eft_it_col, eft_jt_col, bootstrap=None):
        if bootstrap is None:
            self.mo = self.model(
                df, id_col, tid_col, grade_col, y_col, eft_it_col, eft_jt_col,
                max_iteration=self.max_iteration, seed=self.seed, verbose=self.verbose, tolerance=self.tolerance
            )
            self.mo.fit()
        else:
            for _ in range(bootstrap):
                self.mo = self.model(
                    df, id_col, tid_col, grade_col, y_col, eft_it_col, eft_jt_col,
                    max_iteration=self.max_iteration, seed=self.seed, verbose=self.verbose, tolerance=self.tolerance
                )
                self.mo.fit()
                self.parameters_dict_bootstraped.append(self.mo.parameters_dict)


# df = pd.read_csv('./test.csv')
# sr = SimpleRegression()
# sr.fit(
#     df=df,
#     id_col = 'ids',
#     tid_col = 'tid',
#     grade_col = 'grade',
#     y_col = 'y',
#     eft_it_col = 'effect_it',
#     eft_jt_col = 'effect_tid_t',
#     bootstrap=2
# )
# sr.parameters_dict_bootstraped

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