from layers.layer import Layer


class Model:

    def __init__(self):
        self.layers = []

    def add(self, layer: Layer):
        self.layers.append(layer)

    def compile(self):
        # ----- Implement ------

        pass

    def fit(self):
        # ----- Implement ------

        pass

    def predict(self):
        # ----- Implement ------

        pass

    def evaluate(self):
        # ----- Implement ------

        pass

    def summary(self):
        # ----- Implement ------

        pass
