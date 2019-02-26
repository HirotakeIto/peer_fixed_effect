class TeacherEffectMixin:
    def get_teacher_effect_ijgt_discounted_cumsum_except_now(self, **argv):
        raise NotImplementedError

    def get_teacher_effect_ijgt_discounted_cumsum(self, **argv):
        raise NotImplementedError

    def get_teacher_effect_ijgt(self, **argv):
        raise NotImplementedError

    def initialize_teacher_effect_ijgt(self, **argv):
        raise NotImplementedError

    def estimate_teacher_effect_ijgt(self, **argv):
        raise NotImplementedError


class IndividualEffectMixin:
    def get_individual_effect_ijgt(self, **argv):
        pass

    def initialize_individual_effect_ijgt(self, **argv):
        raise NotImplementedError

    def estimate_individual_effect_ijgt(self, **argv):
        raise NotImplementedError


class PersistenceMixin:
    def estimated_sigma(self, **argv):
        raise NotImplementedError


# class InitializationMixin:
#     def initialization(self, **argv):
#         raise NotImplementedError
