import numpy as np

def sigmoid(u):
    return 1 / (1 + np.exp(-u))

def softmax(u):
    # tira o maior de cada linha antes do exponencial, senao estoura
    shifted = u - u.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)

    return exponentials / exponentials.sum(axis=1, keepdims=True)

def binary_cross_entropy(Y, P):
    return -np.mean(Y * np.log(P + 1e-12) + (1 - Y) * np.log(1 - P + 1e-12))

def categorical_cross_entropy(Y, P):
    return -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))

def one_hot(y, classes):
    return np.eye(classes)[y]

class MLP:

    def __init__(self, sizes, rng, eta, output, loss):
        self.sizes = sizes
        self.eta = eta
        self.output = output
        self.loss = loss
        self.losses = []

        self.W = []
        self.b = []

        for i in range(len(sizes) - 1):
            self.W.append(rng.normal(0, 1 / np.sqrt(sizes[i]), (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

    def forward(self, X):
        activations = [X]

        for i in range(len(self.W)):
            u = activations[-1] @ self.W[i] + self.b[i]

            if i < len(self.W) - 1:
                activations.append(np.tanh(u))
            else:
                activations.append(self.output(u))

        return activations

    def backward(self, activations, Y):
        # sigmoid com entropia binaria e softmax com entropia categorica dao o mesmo delta
        delta = (activations[-1] - Y) / len(Y)

        grad_W = [None] * len(self.W)
        grad_b = [None] * len(self.W)

        for i in reversed(range(len(self.W))):
            grad_W[i] = activations[i].T @ delta
            grad_b[i] = delta.sum(axis=0)

            if i > 0:
                # derivada do tanh: 1 - tanh(u)^2, e activations[i] ja e o tanh
                delta = (delta @ self.W[i].T) * (1 - activations[i] ** 2)

        return grad_W, grad_b

    def update(self, grad_W, grad_b):
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - self.eta * grad_W[i]
            self.b[i] = self.b[i] - self.eta * grad_b[i]

    def train(self, X, Y, epochs):
        for _ in range(epochs):
            activations = self.forward(X)
            self.losses.append(self.loss(Y, activations[-1]))

            grad_W, grad_b = self.backward(activations, Y)
            self.update(grad_W, grad_b)

    def predict(self, X):
        probabilities = self.forward(X)[-1]

        if probabilities.shape[1] == 1:
            return (probabilities[:, 0] > 0.5).astype(int)

        return probabilities.argmax(axis=1)

    def count_parameters(self):
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)

def accuracy(model, X, y):
    return np.mean(model.predict(X) == y)
