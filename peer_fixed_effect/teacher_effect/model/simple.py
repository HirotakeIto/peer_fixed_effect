"""
Notation:
 i: individual student
 j: teacher
 g: grade
 t: time
"""

from .base import BaseModel
from ..structure.simple import SimplePersistenceMixin, SimpleTeacherEffectMixin, SimpleIndividualEffectMixin
from ..structure.simple_fixed import SimpleTeacherFixedEffectMixin


class SimpleModel(BaseModel, SimpleIndividualEffectMixin, SimpleTeacherEffectMixin, SimplePersistenceMixin):
    @property
    def parameters(self):
        return self.sigma, self.df[self.eft_it_col], self.df[self.eft_jt_col]

    @property
    def parameters_dict(self):
        return {'sigma': self.sigma, 'eft_it_col':self.df[self.eft_it_col], 'eft_jt_col':self.df[self.eft_jt_col]}

    @property
    def persistence(self):
        return self.sigma

    def __init__(
            self, df, id_col, tid_col, grade_col, y_col, eft_it_col, eft_jt_col,
            max_iteration=1000, seed=None, verbose=True, tolerance=10 ** (-4), random_init=False,
            **argv):
        # id_col = 'ids'
        # tid_col = 'tid'
        # grade_col = 'grade'
        # time_col = 'time'
        # y_col = 'y'
        # eft_i_col = 'effect_i'
        # eft_it_col = 'effect_it'
        # eft_jt_col = 'effect_tid_t'
        # max_grade_col = 'max_grade'
        # min_grade_col = 'min_grade'
        # sigma = 0.1
        self.seed = seed
        self.max_iteration = max_iteration
        self.verbose = verbose
        self.tolerance = tolerance
        self.id_col = id_col
        self.tid_col = tid_col
        self.grade_col = grade_col
        # self.time_col = time_col
        self.y_col = y_col
        # self.eft_i_col = eft_i_col
        self.eft_it_col = eft_it_col
        self.eft_jt_col = eft_jt_col
        self.max_grade_col = 'max_grade'
        self.min_grade_col = 'min_grade'
        self.random_init = random_init
        self.df = (
            df
            .assign(
                min_grade=lambda dfx: dfx.groupby(self.id_col)[self.grade_col].transform('min'),
                max_grade=lambda dfx: dfx.groupby(self.id_col)[self.grade_col].transform('max')
            )
        )

    def initialization(self, random_init=False):
        self.sigma = 0.5
        for _ in range(5):
            self.initialize_individual_effect_ijgt(self, random_init)
            self.initialize_teacher_effect_ijgt(self, random_init)

    def iteration(self, **argv):
        self.sigma = self.estimated_sigma(self, ftol=10 ** (-3), sigma_init=self.sigma)
        self.df[self.eft_it_col] = self.estimate_individual_effect_ijgt(self)
        self.df[self.eft_jt_col] = self.estimate_teacher_effect_ijgt(self)

    def fit(self):
        self.initialization(self.random_init)
        # import pdb;pdb.set_trace()
        for iteration in range(self.max_iteration):
            sigma_prime = self.sigma
            self.iteration()
            print(self.sigma)
            if abs(sigma_prime - self.sigma) < self.tolerance:
                print(self.sigma)
                if iteration > 10:
                    break
            if self.verbose & (iteration % 10 == 3):
                print("{q}th iteration: estimated gamma is {sigma:.4f}".format(q=iteration, sigma=self.sigma))


class SimpleFixedModel(BaseModel, SimpleIndividualEffectMixin, SimpleTeacherFixedEffectMixin, SimplePersistenceMixin):
    @property
    def parameters(self):
        return self.sigma, self.df[self.eft_it_col], self.df[self.eft_jt_col]

    @property
    def parameters_dict(self):
        return {'sigma': self.sigma, 'eft_it_col':self.df[self.eft_it_col], 'eft_jt_col':self.df[self.eft_jt_col]}

    @property
    def persistence(self):
        return self.sigma

    def __init__(
            self, df, id_col, tid_col, grade_col, y_col, eft_it_col, eft_jt_col,
            max_iteration=1000, seed=None, verbose=True, tolerance=10 ** (-4), random_init=False,
            **argv):
        # id_col = 'ids'
        # tid_col = 'tid'
        # grade_col = 'grade'
        # time_col = 'time'
        # y_col = 'y'
        # eft_i_col = 'effect_i'
        # eft_it_col = 'effect_it'
        # eft_jt_col = 'effect_tid_t'
        # max_grade_col = 'max_grade'
        # min_grade_col = 'min_grade'
        # sigma = 0.1
        self.seed = seed
        self.max_iteration = max_iteration
        self.verbose = verbose
        self.tolerance = tolerance
        self.id_col = id_col
        self.tid_col = tid_col
        self.grade_col = grade_col
        # self.time_col = time_col
        self.y_col = y_col
        # self.eft_i_col = eft_i_col
        self.eft_it_col = eft_it_col
        self.eft_jt_col = eft_jt_col
        self.max_grade_col = 'max_grade'
        self.min_grade_col = 'min_grade'
        self.random_init = random_init
        self.df = (
            df
            .assign(
                min_grade=lambda dfx: dfx.groupby(self.id_col)[self.grade_col].transform('min'),
                max_grade=lambda dfx: dfx.groupby(self.id_col)[self.grade_col].transform('max')
            )
        )

    def initialization(self, random_init=False):
        self.sigma = 0.5
        for _ in range(5):
            self.initialize_individual_effect_ijgt(self, random_init)
            self.initialize_teacher_effect_ijgt(self, random_init)

    def iteration(self, **argv):
        self.sigma = self.estimated_sigma(self, ftol=10 ** (-3), sigma_init=self.sigma)
        self.df[self.eft_it_col] = self.estimate_individual_effect_ijgt(self)
        self.df[self.eft_jt_col] = self.estimate_teacher_effect_ijgt(self)

    def fit(self):
        self.initialization(self.random_init)
        # import pdb;pdb.set_trace()
        for iteration in range(self.max_iteration):
            sigma_prime = self.sigma
            self.iteration()
            print(self.sigma)
            if abs(sigma_prime - self.sigma) < self.tolerance:
                print(self.sigma)
                if iteration > 10:
                    break
            if self.verbose & (iteration % 10 == 3):
                print("{q}th iteration: estimated gamma is {sigma:.4f}".format(q=iteration, sigma=self.sigma))

