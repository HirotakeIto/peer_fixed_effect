class BaseModel:
    @property
    def parameters(self):
        raise NotImplementedError

    @property
    def parameters_dict(self):
        raise NotImplementedError

    @property
    def persistence(self):
        raise NotImplementedError

    def fit(self, **argv):
        raise NotImplementedError

    def initialization(self, **argv):
        raise NotImplementedError

    def iteration(self, **argv):
        raise NotImplementedError