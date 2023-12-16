class Regularizer:
    def __init__(self):
        pass


class L1(Regularizer):
    def __init__(self, l1):
        self.l1 = l1
        super().__init__()


class L2(Regularizer):
    def __init__(self, l2):
        self.l2 = l2
        super().__init__()
