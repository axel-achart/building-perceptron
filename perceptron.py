# Building the class of Perceptron
# Construction de la classe Perceptron

import numpy as np


class Perceptron:
    # Store hyperparameters, weights and bias are created in fit()
    # Stocke les hyperparamètres, les poids et le biais sont créés dans fit()
    def __init__(self, learning_rate=0.1, n_epochs=50):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    # Step activation: 1 if the weighted sum is positive or zero, else 0
    # Fonction d'activation en escalier : 1 si la somme pondérée est positive ou nulle, sinon 0
    def step(self, weighted_sum):
        return np.where(weighted_sum >= 0, 1, 0)

    # Weighted sum of the inputs plus the bias
    # Somme pondérée des entrées plus le biais
    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    # Train the perceptron with the perceptron learning rule
    # Entraîne le perceptron avec la règle d'apprentissage du perceptron
    def fit(self, X, y):
        n_features = X.shape[1]

        # Weights and bias initialized to zero, one weight per feature
        # Poids et biais initialisés à zéro, un poids par feature
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_epochs):
            for xi, target in zip(X, y):
                y_pred = self.step(self.net_input(xi))
                error = target - y_pred

                # Update rule: w = w + lr * (y - y_pred) * x and b = b + lr * (y - y_pred)
                # Règle de mise à jour : w = w + lr * (y - y_pred) * x et b = b + lr * (y - y_pred)
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error
        return self

    # Predict the class (0 or 1) for one or several samples
    # Prédit la classe (0 ou 1) pour un ou plusieurs échantillons
    def predict(self, X):
        return self.step(self.net_input(X))


# Generate random fake data: two point clouds, one per class
# Génère des données factices aléatoires : deux nuages de points, un par classe
def generate_data(n_samples=200):
    half = n_samples // 2
    class_0 = np.random.randn(half, 2) + [-2, -2]
    class_1 = np.random.randn(half, 2) + [2, 2]
    X = np.vstack((class_0, class_1))

    # Labels: 0 for the first cloud, 1 for the second
    # Étiquettes : 0 pour le premier nuage, 1 pour le second
    y = np.array([0] * half + [1] * half)
    return X, y


# Shuffle the data and split it into a train set and a test set
# Mélange les données et les sépare en un jeu d'entraînement et un jeu de test
def train_test_split(X, y, test_size=0.2):
    indices = np.random.permutation(len(X))
    n_test = int(len(X) * test_size)
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# Share of correct predictions
# Proportion de prédictions correctes
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)



if __name__ == "__main__":
    X, y = generate_data()                                          # Génération de données aléatoires
    X_train, X_test, y_train, y_test = train_test_split(X, y)       # Séparation jeu de test et d'entraînement

    model = Perceptron(learning_rate=0.1, n_epochs=50)              # Déclaration classe dans une variable avec certains paramètres
    model.fit(X_train, y_train)                                     # Lancement Entraînement

    y_pred = model.predict(X_test)                                  # Lancement Prédiction (donc application final)
    print("weights : " + str(model.weights))
    print("bias : " + str(model.bias))
    print("accuracy : " + str(accuracy(y_test, y_pred)))
