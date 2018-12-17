import numpy as np
from pandas import DataFrame
from numpy import c_
from sklearn.linear_model import LinearRegression


class PeerFixedEffectStructureMixin:
    def __init__(self):
        self.LR = LinearRegression()

    @staticmethod
    def get_mean_alpha_jt(time_it: np.array, group_it: np.array, alpha_it0: np.array):
        return (
            DataFrame(c_[time_it, group_it, alpha_it0], columns=['time', 'group', 'alphait0'])
            .groupby(['time', 'group'])
            [['alphait0']]
            .transform(lambda x: (x.sum() - x) / (x.count() - 1))
            .values
        )

    def get_gamma0(self, y_it: np.array, alpha_it: np.array, mean_alpha_jt: np.array, rho_boundry=0.4):
        """
        LinearRegressionのオブジェクト生成にかかる時間を節約する
        """
        return self._get_gamma0(
            LR=self.LR, y_it=y_it, alpha_it=alpha_it, mean_alpha_jt=mean_alpha_jt, rho_boundry=rho_boundry)

    @staticmethod
    def _get_gamma0(LR, y_it, alpha_it, mean_alpha_jt, rho_boundry=0.4):
        LR.fit(y = y_it - alpha_it, X=mean_alpha_jt)
        return max(min(LR.coef_[0, 0], rho_boundry), -rho_boundry)

    @staticmethod
    def get_new_alpha_it0(
            n_time, gamma0,
            id_it: np.array, time_it: np.array, group_it: np.array, y_it: np.array, alpha_it: np.array
    ):
        """
        Return  alpha_it0 vector which represents fixed-effect of individual i.
        This vector is (n * t, 1) matrix (n: number of individuals, t: number of time).
        :param n_time:
        :param gamma0:
        :param id_it:
        :param time_it:
        :param group_it:
        :param y_it:
        :param alpha_it:
        :return:
        """
        # read set
        df = DataFrame(
            c_[id_it, time_it, group_it, y_it, alpha_it],
            columns=['id', 'time', 'group', 'yit', 'alphait0']
        )
        # calc
        grouped = df.groupby(['time', 'group'])
        df['m_tn'] = grouped['group'].transform(lambda x: x.shape[0] - 1)
        df['s_tn'] = grouped['alphait0'].transform('sum')
        df['sum_yit'] = grouped['yit'].transform('sum')
        df['gamma_over_m'] = gamma0 / df['m_tn']
        df['part1'] = df['yit']
        df['part2'] = df['gamma_over_m'] * (df['s_tn'] - df['alphait0'])
        df['part3'] = df['gamma_over_m'] * (df['sum_yit'] - df['yit'])
        df['part4'] = df['part2']
        df['part5'] = \
            (df['gamma_over_m'] * df['gamma_over_m']) \
            * \
            (df['m_tn'] * (df['s_tn'] - df['alphait0']) - df['s_tn'] + df['alphait0'])
        df['A_it'] = df['part1'] - df['part2'] + df['part3'] - df['part4'] - df['part5']
        df['child'] = df.groupby('id')['A_it'].transform('sum')
        df['mother'] = n_time + gamma0 * gamma0 / df.groupby('id')['m_tn'].transform('sum')
        df['new_alpha_it0'] = df['child'] / df['mother']
        return df[['new_alpha_it0']].values

    @staticmethod
    def get_alpha_it(**argv):
        raise NotImplementedError('Please use overided class')


class StaticPeerFixedEffectStructureMixin(PeerFixedEffectStructureMixin):
    @staticmethod
    def get_alpha_it(alpha_it0: np.array):
        return alpha_it0


class CumulativePeerFixedEffectStructureMixin(PeerFixedEffectStructureMixin):
    @staticmethod
    def get_alpha_it(alpha_it0: np.array, id_it: np.array, group_it: np.array, time_it: np.array, gamma0: float):
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
