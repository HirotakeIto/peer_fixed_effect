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
        df = DataFrame(c_[id_it, group_it, time_it, alpha_it0],
                      columns=['id', 'group', 'time', 'alpha_it0'])
        time_list = sorted(df['time'].unique().tolist())

        def mean_alphajt0(df):
            df['mean_alphajt_1'] = df.groupby(['group', 'time'])['alphait_1'].transform(
                lambda x: (x.sum() - x) / (x.count() - 1))
            return df

        df.loc[df['time'] == time_list[0], 'alpha_it'] = df['alpha_it0']
        for time in time_list[1:]:
            df['alphait_1'] = (
                df
                    .sort_values(['id', 'time'])
                    .groupby(['id'])
                ['alpha_it']
                    .shift(1)
            )
            df = df.pipe(mean_alphajt0)
            df = df.sort_index()
            df.loc[df['time'] == time, 'alpha_it'] = df['alpha_it0'] + gamma0 * df['mean_alphajt_1']
        alpha_it = df[['alpha_it']].values
        return alpha_it

        # def mean_alphajt0(df_tmp):
        #     df_tmp['mean_alphajt0'] = df_tmp.groupby(['group', 'time'])['alphait0'].transform(
        #         lambda x: (x.sum() - x) / (x.count() - 1))
        #     return df_tmp
        #
        # df = (
        #     DataFrame(c_[id_it, group_it, time_it, alpha_it0],
        #               columns=['id', 'group', 'time', 'alphait0'])
        #     .pipe(mean_alphajt0)
        #     .sort_values(['id', 'time'])
        #     .assign(
        #         cumsum_mean_alphajt0=lambda dfx: dfx.groupby(['id'])['mean_alphajt0'].cumsum())
        #     .assign(
        #         shift_cumsum_mean_alphajt0=lambda dfx: dfx.groupby(['id'])['cumsum_mean_alphajt0'].shift(1))
        #     .fillna(0)
        #     .sort_index()
        #     .assign(
        #         alpha_it=lambda dfx: dfx['alphait0'] + gamma0 * dfx['shift_cumsum_mean_alphajt0']
        #     )
        # )
        # alpha_it = df[['alpha_it']].values
        # return alpha_it

        # df = DataFrame(c_[id_it, group_it, time_it, alpha_it0], columns=['id', 'group', 'time', 'alphait0'])
        # df['mean_alphajt0'] = df.groupby(['group', 'time'])['alphait0'].transform(
        #     lambda x: (x.sum() - x) / (x.count() - 1))
        # df['cumsum_mean_alphajt0'] = (
        #     df
        #     .sort_values(['id', 'time'])
        #     .groupby(['id'])
        #     ['mean_alphajt0']
        #     .cumsum()
        #     .sort_index()
        # )
        # df['alpha_it'] = df['alphait0'] + gamma0 * df['cumsum_mean_alphajt0']
        # alpha_it = df[['alpha_it']].values
        # return alpha_it
