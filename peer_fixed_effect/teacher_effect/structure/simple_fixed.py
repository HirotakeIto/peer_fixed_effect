from .simple import SimpleTeacherEffectMixin
from numpy import random
import pandas as pd

class SimpleTeacherFixedEffectMixin(SimpleTeacherEffectMixin):
    def initialize_teacher_effect_ijgt(self, cls, random_init=False, **argv):
        if random_init is True:
            cls.df[cls.eft_jt_col] = random.normal(0, 1, size=cls.df.shape[0])
            cls.df[cls.eft_jt_col] = cls.df.groupby([cls.tid_col])[cls.eft_jt_col].transform('mean')
        elif (cls.eft_it_col not in cls.df.columns) | (cls.eft_jt_col not in cls.df.columns):
            cls.df[cls.eft_jt_col] = cls.df.groupby([cls.tid_col])[cls.y_col].transform('mean')
        else:
            cls.df[cls.eft_jt_col] = cls.df[cls.y_col] - cls.df[cls.eft_it_col]
            cls.df[cls.eft_jt_col] = cls.df.groupby([cls.tid_col])[cls.eft_jt_col].transform('mean')


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
        """
        間違っていると思うんだけど、、、なぜかワークした
        """
        df['dis_des'] = sigma ** (df[max_grade_col] - df[grade_col])
        df['eft_jt_dis_des'] = df[eft_jt_col] * df['dis_des']
        df_gby_id_sorted = df.sort_values([grade_col], ascending=True).groupby(id_col)
        df_gby_idtid_sorted = df.sort_values([grade_col], ascending=True).groupby([id_col, tid_col], sort=False)
        df['dis_des_id_tid'] = (df_gby_idtid_sorted['dis_des'].cumsum().sort_index() / df['dis_des']).fillna(0)
        df['dis_des_eft_jt_id_tid'] = (
                    df_gby_idtid_sorted['eft_jt_dis_des'].cumsum().sort_index() / df['dis_des']).fillna(0)
        df['dis_des_id_tid2'] = df['dis_des_id_tid'] ** 2
        df['resid_id_t'] = df[y_col] - df[eft_it_col] - (
                    (df_gby_id_sorted['eft_jt_dis_des'].cumsum().sort_index()) / df['dis_des']).fillna(0) + df[
                               'dis_des_eft_jt_id_tid']
        df['weighted_resid_id_t'] = df['dis_des_id_tid'] * df['resid_id_t']
        df['mother'] = df.groupby([id_col, tid_col])['dis_des_id_tid2'].transform('sum')
        df['mother'] = df.groupby([tid_col])['mother'].transform('sum')
        df['child'] = df.groupby([id_col, tid_col])['weighted_resid_id_t'].transform('sum')
        df['child'] = df.groupby([tid_col])['child'].transform('sum')
        return df['child'] / df['mother']
