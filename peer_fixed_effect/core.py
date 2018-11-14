# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from . import helpers
import pandas as pd
import numpy as np
from typing import Union


class PeerFixedEffectRegression(BaseEstimator):
    gammma = None
    alpha = None
    ids = None
    time = None

    def __init__(self, effect='static', max_iteration=1000, seed=None):
        super().__init__()
        self.effect = effect
        self.max_iteration = max_iteration
        if seed:
            np.random.seed(seed)

    def fit(self,
            x: Union(np.array, pd.DataFrame),
            y: Union(np.array, pd.DataFrame),
            group: Union(np.array, pd.DataFrame),
            ids: Union(np.array, pd.DataFrame),
            times: Union(np.array, pd.DataFrame)):
        """

        :param x:
        :param y:
        :param ids:
        :param times:
        :return:
        """
        pass

    def predict(self):
        pass
