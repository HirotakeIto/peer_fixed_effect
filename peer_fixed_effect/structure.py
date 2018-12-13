from typing import Union
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class Structure:

    def __init__(self, x, y, group, ids, times, seed=None):
        # data's parametor
        self.gamma = None
        self.num_time = None
        # data's columns
        self.id_col = 'ids'
        self.group_col = 'group'
        self.time_col = 'time'
        self.y_col = 'yit'
        self.x_col_format = 'xit{i}'
        self.x_col = [self.x_col_format.format(i=i) for i in range(x.shape[1])]
        # fixed effect columns
        self.alphai0_qth_col = 'alphai0_qth'
        self.alphait_qth_col = 'alphait_qth'
        self.alphai0_q_minus_1th_col = 'alphai0_q_minus_1th'
        self.mean_alphajt_ith_col = 'mean_alphajt_ith'
        # setting
        self.df = self.create_data(x=x, y=y, group=group, ids=ids, times=times)
        self.set_initial_value()
        self.calc_data_value()
        # util
        self.LR = LinearRegression(fit_intercept=True)
        # other setting
        if seed:
            np.random.seed(seed)

    def create_data(
            self,
            x: np.array,
            y: np.array,
            group: np.array,
            ids: np.array,
            times: np.array):
        columns = [self.id_col, self.group_col, self.time_col, self.y_col] + self.x_col
        df = pd.DataFrame(np.c_[ids, group, times, y, x], columns=columns)
        return df

    def set_initial_value(self):
        self.df[self.alphai0_qth_col] = np.random.random(size=1)[0]
        self.df[self.alphait_qth_col] = self.df[self.alphai0_qth_col]

    def set_loop(self):
        self.df[self.alphai0_q_minus_1th_col] = self.df[self.alphai0_qth_col]

    def calc_data_value(self):
        self.num_time = len(self.df[self.time_col].unique())

    def set_alphait(self):
        self.df[self.alphait_qth_col] = self.calc_alphait()

    def calc_alphait(self):
        return self.df[self.alphai0_qth_col]

    def set_mean_alphajt(self):
        self.df[self.mean_alphajt_ith_col] = self.calc_mean_alphajt()

    def calc_mean_alphajt(self):
        return \
            self.df \
            .groupby([self.group_col, self.time_col])[self.alphai0_q_minus_1th_col] \
            .transform(lambda x: (x.sum() - x) / (x.count() - 1))

    def set_gamma(self):
        self.LR.fit(
            y=self.df[self.y_col] - self.df[self.alphai0_q_minus_1th_col],
            X=self.df[[self.mean_alphajt_ith_col]])
        self.gamma = max(min(self.LR.coef_[0], 0.399), -0.399)

    def set_alphai0_qth(self):
        self.df[self.alphai0_qth_col] = self.calc_alphai0_qth()

    def calc_alphai0_qth(self):
        # read set
        df = self.df
        gamma_ith = self.gamma
        num_time = self.num_time
        # calc
        grouped = df.groupby([self.time_col, self.group_col])
        df['m_tn'] = grouped[self.group_col].transform(lambda x: x.shape[0] - 1)
        df['s_tn'] = grouped[self.alphai0_q_minus_1th_col].transform('sum')
        df['sum_yit'] = grouped[self.y_col].transform('sum')
        df['gamma_over_m'] = gamma_ith / df['m_tn']
        df['part1'] = df[self.y_col]
        df['part2'] = df['gamma_over_m'] * (df['s_tn'] - df[self.alphai0_q_minus_1th_col])
        df['part3'] = df['gamma_over_m'] * (df['sum_yit'] - df[self.y_col])
        df['part4'] = df['part2']
        df['part5'] = \
            (df['gamma_over_m'] * df['gamma_over_m']) \
            * \
            (df['m_tn'] * (df['s_tn'] - df[self.alphai0_q_minus_1th_col]) - df['s_tn'] + df[self.alphai0_q_minus_1th_col])
        df['A_it'] = df['part1'] - df['part2'] + df['part3'] - df['part4'] - df['part5']
        df['child'] = df.groupby(self.id_col)['A_it'].transform('sum')
        df['mother'] = num_time + gamma_ith * gamma_ith / df.groupby(self.id_col)['m_tn'].transform('sum')
        return df['child'] / df['mother']


def debug():
    import pandas as pd
    df = pd.read_csv('memo/data/sample1/sample1.csv')
    df = pd.read_csv('memo/data/sample.csv')
    dt = Structure(
        x=df[['xit0', 'xit1', 'xit2']].values,
        y=df['yit'].values,
        group=df['group'].values,
        ids=df['ids'].values,
        times=df['time'].values,
    )
    dt.set_initial_value()
    dt.set_loop()
    for _ in range(10):
        dt.set_alphait()
        dt.set_mean_alphajt()
        dt.set_gamma()
        dt.set_alphai0_qth()
        print(dt.gamma)
        # print(dt.df[dt.alphait_qth_col].head(5))
        dt.set_loop()

