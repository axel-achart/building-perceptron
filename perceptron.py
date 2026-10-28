import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def activation(self, value):
        return 1 if value >= 0 else 0

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])

        for _ in range(self.epochs):
            errors = 0

            for inputs, target in zip(X, y):
                value = np.dot(inputs, self.weights) + self.bias
                prediction = self.activation(value)
                error = target - prediction

                if error != 0:
                    self.weights += self.learning_rate * error * inputs
                    self.bias += self.learning_rate * error
                    errors += 1

            if errors == 0:
                break

    def predict(self, X):
        predictions = []

        for inputs in X:
            value = np.dot(inputs, self.weights) + self.bias
            predictions.append(self.activation(value))

        return np.array(predictions)
