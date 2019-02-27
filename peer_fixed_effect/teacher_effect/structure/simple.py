"""
Notation:
 i: individual student
 j: teacher
 g: grade
 t: time
"""

import pandas as pd
# from scipy.optimize import least_squares
from scipy.optimize import minimize
from numpy import random
from .mixin import TeacherEffectMixin, IndividualEffectMixin, PersistenceMixin

__all__ = [
    'SimpleTeacherEffectMixin', 'SimpleIndividualEffectMixin', 'SimplePersistenceMixin'
]

class SimpleTeacherEffectMixin(TeacherEffectMixin):
    def get_teacher_effect_ijgt_discounted_cumsum_except_now(
            self, df: pd.DataFrame, sigma, id_col, grade_col, eft_jt_col, max_grade_col, **argv):
        # df_gby_id_sorted = df.sort_values([grade_col], ascending=True).groupby(id_col)
        # max_grade = df_gby_id_sorted[grade_col].transform('max')
        # df['discount'] = (sigma ** (max_grade - df[grade_col]))
        # df['eft_jt_n_discount'] = df[eft_jt_col] * df['discount']
        # df_gby_id_sorted = df.sort_values([grade_col], ascending=True).groupby(id_col)
        if sigma == 0:
            return 0
        df['dis_des'] = sigma ** (df[max_grade_col] - df[grade_col])
        df['eft_jt_dis_des'] = df[eft_jt_col] * df['dis_des']
        df_gby_id_sorted = df.sort_values([grade_col], ascending=True).groupby(id_col)
        value = ((df_gby_id_sorted['eft_jt_dis_des'].cumsum().sort_index() - df['eft_jt_dis_des'])/df['dis_des']).fillna(0) # これモデル的に唯のcumsumでも良いのでは？
        df.drop(['dis_des', 'eft_jt_n_discount'], inplace=True, axis=1)
        return value

    def get_teacher_effect_ijgt_discounted_cumsum(
            self, df: pd.DataFrame, sigma, id_col, grade_col, eft_jt_col, max_grade_col, **argv):
        # df_gby_id_sorted = df.sort_values([grade_col], ascending=True).groupby(id_col)
        # max_grade = df_gby_id_sorted[grade_col].transform('max').sort_index()
        # df['discount'] = (sigma ** (max_grade - df[grade_col]))
        # df['eft_jt_n_discount'] = df[eft_jt_col] * df['discount']
        # import pdb;pdb.set_trace()
        if sigma == 0:
            return df[eft_jt_col]
        df['dis_des'] = sigma ** (df[max_grade_col] - df[grade_col])
        df['eft_jt_dis_des'] = df[eft_jt_col] * df['dis_des']
        df_gby_id_sorted = df.sort_values([grade_col], ascending=True).groupby(id_col)
        value = ((df_gby_id_sorted['eft_jt_dis_des'].cumsum().sort_index()) / df['dis_des']).fillna(0)
        return value

    def get_teacher_effect_ijgt(self, df: pd.DataFrame, effect_jt, **argv):
        return df[effect_jt]

    def initialize_teacher_effect_ijgt(self, cls, random_init=False, **argv):
        if random_init is True:
            cls.df[cls.eft_jt_col] = random.uniform(0, 1, size=cls.df.shpae[0])
        elif (cls.eft_it_col not in cls.df.columns) | (cls.eft_jt_col not in cls.df.columns):
            cls.df[cls.eft_jt_col] = cls.df.groupby([cls.tid_col, cls.grade_col])[cls.y_col].transform('mean')
        else:
            cls.df[cls.eft_jt_col] = cls.df[cls.y_col] - cls.df[cls.eft_it_col]
            cls.df[cls.eft_jt_col] = cls.df.groupby([cls.tid_col, cls.grade_col])[cls.eft_jt_col].transform('mean')

    def estimate_teacher_effect_ijgt(self, cls, **argv):
        return self._estimate_teacher_effect_jg(
            df=cls.df,
            sigma=cls.sigma,
            id_col=cls.id_col,
            tid_col=cls.tid_col,
            grade_col=cls.grade_col,
            y_col=cls.y_col,
            eft_it_col=cls.eft_it_col,
            eft_jt_col=cls.eft_jt_col,
            min_grade_col=cls.min_grade_col,
            max_grade_col=cls.max_grade_col,
            **argv
        )

    def _estimate_teacher_effect_jg(
            self, df: pd.DataFrame, sigma, id_col, tid_col, grade_col, y_col, eft_it_col, eft_jt_col, min_grade_col, max_grade_col, **argv):
        df['dis_asc'] = sigma ** (df[grade_col] - df[min_grade_col])
        df['dis_des'] = sigma ** (df[max_grade_col] - df[grade_col])
        df['dis_asc2'] = sigma ** (2 * (df[grade_col] - df[min_grade_col]))
        df['eft_jt_dis_des'] = df[eft_jt_col] * df['dis_des']
        df['eft_jt_dis_asc'] = df[eft_jt_col] * df['dis_asc']
        df_gby_id_sorted = df.sort_values([grade_col], ascending=True).groupby(id_col, sort=False)
        df_gby_id_sortedr = df.sort_values([grade_col], ascending=False).groupby(id_col, sort=False)
        # df_gby_idtid_sorted = df.sort_values([grade_col], ascending=True).groupby([id_col, tid_col], sort=False)
        df['partC'] = ((df_gby_id_sortedr['dis_asc2'].cumsum().sort_index() - df['dis_asc2']) / df['dis_asc2']).fillna(
            0)
        df['partc'] = 1 + df['partC']
        # PartA
        # df['parta'] = df[y_col] - df[eft_it_col] - (df_gby_idtid_sorted[eft_jt_col].cumsum().sort_index() - df[eft_jt_col]) #  間違ってるよ。。。
        df['parta'] = df[y_col] - df[eft_it_col] - (
                    (df_gby_id_sorted['eft_jt_dis_des'].cumsum().sort_index() - df['eft_jt_dis_des']) / df[
                'dis_des']).fillna(0)
        # PartB
        # df['partb2']  = df[y_col] - df[eft_it_col] -df[eft_jt_col] - ((df_gby_idtid_sorted['eft_jt_dis_des'].cumsum().sort_index() - df['eft_jt_dis_des'])/df['dis_des']).fillna(0) #  間違ってるよ。。。
        df['partb2'] = df[y_col] - df[eft_it_col] - df[eft_jt_col] - (
                    (df_gby_id_sorted['eft_jt_dis_des'].cumsum().sort_index() - df['eft_jt_dis_des']) / df[
                'dis_des']).fillna(0)
        df['partb2_dis_asc'] = df['partb2'] * df['dis_asc']
        df_gby_id_sortedr = df.sort_values([grade_col], ascending=False).groupby(id_col)
        df['partb'] = ((df_gby_id_sortedr['partb2_dis_asc'].cumsum().sort_index() - df['partb2_dis_asc']) / df[
            'dis_asc']).fillna(0) + df[eft_jt_col] * df['partC']
        # tid_updated
        df['value'] = (df['parta'] + df['partb'])
        return df.groupby([grade_col, tid_col])['value'].transform('sum') / df.groupby([grade_col, tid_col])[
            'partc'].transform('sum')


class SimpleIndividualEffectMixin(IndividualEffectMixin):
    def get_individual_effect_ijgt(self, df: pd.DataFrame, eft_it_col, **argv):
        return df[eft_it_col]

    def initialize_individual_effect_ijgt(self, cls, random_init=False, **argv):
        if random_init is True:
            cls.df[cls.eft_it_col] = random.uniform(0, 1, size=cls.df.shpae[0])
        elif (cls.eft_it_col not in cls.df.columns) | (cls.eft_jt_col not in cls.df.columns):
            cls.df[cls.eft_it_col] = cls.df.groupby([cls.id_col])[cls.y_col].transform('mean')
        else:
            cls.df[cls.eft_it_col] = cls.df[cls.y_col] - cls.df[cls.eft_jt_col]
            cls.df[cls.eft_it_col] = cls.df.groupby([cls.id_col])[cls.eft_it_col].transform('mean')

    def estimate_individual_effect_ijgt(self, cls, **argv):
        return self._estimate_individual_effect_i(
            df=cls.df,
            sigma=cls.sigma,
            id_col=cls.id_col,
            grade_col=cls.grade_col,
            y_col=cls.y_col,
            eft_jt_col=cls.eft_jt_col,
            max_grade_col=cls.max_grade_col,
            **argv
        )

    def _estimate_individual_effect_i(
            self, df: pd.DataFrame, sigma, id_col, grade_col, y_col, eft_jt_col, max_grade_col, **argv):
        # df['dis_asc'] = sigma ** (df[grade_col] - df[min_grade_col])
        # df['dis_des'] = sigma ** (df[max_grade_col] - df[grade_col])
        # df['dis_asc2'] = sigma ** (2 * (df[grade_col] - df[min_grade_col]))
        # df['eft_jt_dis_des'] = df[eft_jt_col] * df['dis_des']
        # df['eft_jt_dis_asc'] = df[eft_jt_col] * df['dis_asc']
        # df_gby_id_sorted = df.sort_values([grade_col], ascending=True).groupby(id_col)
        df['wei'] = \
            df[y_col]\
            - (
                SimpleTeacherEffectMixin().get_teacher_effect_ijgt_discounted_cumsum(
                    df,
                    sigma=sigma,
                    id_col=id_col,
                    grade_col=grade_col,
                    eft_jt_col=eft_jt_col,
                    max_grade_col=max_grade_col
                )
            )
        return df.groupby(id_col)['wei'].transform('mean')


class SimplePersistenceMixin(PersistenceMixin):
    def get_residual_given_sigma(
            self,
            sigma, df: pd.DataFrame,
            id_col, grade_col, y_col, eft_it_col, eft_jt_col, max_grade_col, **argv):
        df['discouted_cumsum'] = SimpleTeacherEffectMixin().get_teacher_effect_ijgt_discounted_cumsum(
            df,
            sigma=sigma,
            id_col=id_col,
            grade_col=grade_col,
            eft_jt_col=eft_jt_col,
            max_grade_col=max_grade_col
        )
        df['resid2'] = (df[y_col] - df[eft_it_col] - df['discouted_cumsum']) ** 2
        return df['resid2'].sum()

    def estimated_sigma(self, cls, **argv):
        return self._estimated_sigma(
            df=cls.df,
            id_col=cls.id_col,
            grade_col=cls.grade_col,
            y_col=cls.y_col,
            eft_it_col=cls.eft_it_col,
            eft_jt_col=cls.eft_jt_col,
            max_grade_col=cls.max_grade_col,
            **argv
        )

    def _estimated_sigma(
            self,
            df: pd.DataFrame, id_col, grade_col, y_col, eft_it_col, eft_jt_col, max_grade_col,
            sigma_init=0.5, ftol=10**(-3), **argv):
        # res_1 = least_squares(
        #     self.get_residual_given_sigma,
        #     sigma_init,
        #     kwargs={
        #         'df': df, 'id_col': id_col, 'grade_col': grade_col,
        #         'y_col': y_col, 'eft_it_col': eft_it_col, 'eft_jt_col': eft_jt_col, 'max_grade_col': max_grade_col},
        #     bounds=([0], [1]),
        #     ftol=ftol
        # )
        res_1 = minimize(
            fun=self.get_residual_given_sigma,
            x0=pd.np.array(sigma_init),
            args=(df, id_col, grade_col,  y_col,  eft_it_col, eft_jt_col,  max_grade_col),
            bounds=([[0, 1]]),
            tol=ftol
        )
        return res_1.x[0]


# class SimpleInitializeMixin(InitializeMixin):
#     def initialization(self, **argv):
#         pass