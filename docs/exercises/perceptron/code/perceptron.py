import numpy as np

ETA = 0.01
MAX_EPOCHS = 100

def step(z):
    # degrau: 1 quando z >= 0, senao 0
    if z >= 0:
        return 1

    return 0

def initial_weights(rng):
    return rng.normal(0, 0.01, size=2), 0.0

class Perceptron:

    def __init__(self, w, b, eta=ETA):
        self.w = w
        self.b = b
        self.eta = eta

        self.epochs = 0
        self.history = []
        self.updates_history = []

        self.pocket_w = w
        self.pocket_b = b
        self.pocket_accuracy = 0.0
        self.pocket_epoch = 0
        self.pocket_history = []

    def predict(self, x):
        return step(np.dot(self.w, x) + self.b)

    def predict_all(self, X):
        return (X @ self.w + self.b >= 0).astype(int)

    def accuracy(self, X, y):
        return np.mean(self.predict_all(X) == y)

    def update(self, x, label):
        # y - y_hat vale 0 quando acerta, +1 ou -1 nos dois tipos de erro
        error = label - self.predict(x)

        if error == 0:
            return False

        self.w = self.w + self.eta * error * x
        self.b = self.b + self.eta * error
        return True

    def keep_best(self, X, y):
        # pocket: guarda a melhor acuracia ja vista e os pesos daquele momento
        current = self.accuracy(X, y)

        if current > self.pocket_accuracy:
            self.pocket_w = self.w
            self.pocket_b = self.b
            self.pocket_accuracy = current
            self.pocket_epoch = self.epochs

    def run_epoch(self, X, y):
        updates = 0

        for i in range(len(X)):
            if self.update(X[i], y[i]):
                updates = updates + 1
                self.keep_best(X, y)

        return updates

    def train(self, X, y):
        for epoch in range(1, MAX_EPOCHS + 1):
            self.epochs = epoch
            updates = self.run_epoch(X, y)

            self.updates_history.append(updates)
            self.history.append(self.accuracy(X, y))
            self.pocket_history.append(self.pocket_accuracy)

            if updates == 0:
                break

    def direction(self):
        return self.w / np.linalg.norm(self.w)

    def wrong(self, X, y):
        return self.predict_all(X) != y

    def wrong_pocket(self, X, y):
        return (X @ self.pocket_w + self.pocket_b >= 0).astype(int) != y
