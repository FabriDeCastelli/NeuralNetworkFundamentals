import numpy as np


class Regularizer:
    def __init__(self):
        pass

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()

    def set_lambda(self, **_):
        raise NotImplementedError()

    def __repr__(self):
        return self.__class__.__name__

    def to_dict(self):
        raise NotImplementedError()


class L1(Regularizer):
    def __init__(self, l1=0.01):
        self.l1 = l1
        super().__init__()

    def set_lambda(self, l1):
        self.l1 = l1

    def forward(self, x):
        return self.l1 * np.sum(np.abs(x))

    def backward(self, x):
        return self.l1 * np.sign(x)

    def to_dict(self):
        return {
            "l1": self.l1
        }


class L2(Regularizer):
    def __init__(self, l2=0.01):
        self.l2 = l2
        super().__init__()

    def set_lambda(self, l2):
        self.l2 = l2

    def forward(self, x):
        return self.l2 * np.sum(np.square(x))

    def backward(self, x):
        return self.l2 * x

    def to_dict(self):
        return {
            "l2": self.l2
        }


class L1L2(Regularizer):
    def __init__(self, l1=0.01, l2=0.01):
        self.l1 = l1
        self.l2 = l2
        super().__init__()

    def set_lambda(self, l1, l2):
        self.l1 = l1
        self.l2 = l2

    def forward(self, x):
        return self.l1 * np.sum(np.abs(x)) + self.l2 * np.sum(np.square(x))

    def backward(self, x):
        return self.l1 * np.sign(x) + self.l2 * x

    def to_dict(self):
        return {
            "l1": self.l1,
            "l2": self.l2
        }


regularizer_dict = {
    "l1": L1(),
    "l2": L2(),
    "l1l2": L1L2(),
}
