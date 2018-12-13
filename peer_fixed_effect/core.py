# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from .structure import Structure


class PeerFixedEffectRegression(BaseEstimator):
    data_cls = Structure

    def __init__(self, effect='static', max_iteration=1000, seed=None):
        super().__init__()
        self.effect = effect
        self.max_iteration = max_iteration
        self.gammma = None
        self.alpha = None
        self.ids = None
        self.time = None

    def fit(self, x, y, group, ids, times):
        """

        :param x:
        :param y:
        :param ids:
        :param times:
        :return:
        """
        dt = self.data_cls(x=x, y=y, group=group, ids=ids, times=times)
        dt.set_initial_value()
        dt.set_loop()
        gamma_1 = 9999999
        for _ in range(1000):
            dt.set_alphait()
            dt.set_mean_alphajt()
            dt.set_gamma()
            dt.set_alphai0_qth()
            # print(dt.df[dt.alphait_qth_col].head(5))
            dt.set_loop()
            print("{q}th iteration: estimated gamma is {gamma:.4f}".format(q=_, gamma=dt.gamma))
            if abs(gamma_1 - dt.gamma) < 10 ** (-5):
                break
            else:
                gamma_1 = dt.gamma


    def predict(self):
        pass
