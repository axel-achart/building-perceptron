"""
perceptron.py
=============

Implémentation from scratch (NumPy uniquement) du Perceptron de Frank Rosenblatt (1958),
en programmation orientée objet.

Le Perceptron est un classifieur binaire linéaire :

    z     = w1*x1 + w2*x2 + ... + wn*xn + b   (somme pondérée + biais)
    y_hat = 1 si z >= 0, sinon 0              (fonction d'activation : marche de Heaviside)

Règle d'apprentissage (règle du Perceptron de Rosenblatt), appliquée exemple par exemple :

    erreur = y - y_hat
    w      = w + eta * erreur * x
    b      = b + eta * erreur

Exemple d'utilisation
---------------------
>>> import numpy as np
>>> from perceptron import Perceptron
>>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
>>> y = np.array([0, 0, 0, 1])                 # porte logique AND
>>> model = Perceptron(learning_rate=0.1, n_epochs=20, random_state=42)
>>> model.fit(X, y).predict(X)
array([0, 0, 0, 1])
"""

from __future__ import annotations

import numpy as np


class Perceptron:
    """Perceptron simple (un seul neurone) pour la classification binaire.

    Paramètres
    ----------
    learning_rate : float, défaut 0.01
        Taux d'apprentissage (eta), compris dans ]0, 1]. Contrôle l'amplitude
        de chaque correction des poids.
    n_epochs : int, défaut 50
        Nombre maximal de passages complets (époques) sur le jeu d'entraînement.
    shuffle : bool, défaut True
        Mélange les exemples à chaque époque pour éviter les cycles liés à l'ordre des données.
    early_stopping : bool, défaut True
        Arrête l'entraînement dès qu'une époque se termine sans aucune erreur
        (le modèle a convergé : les données sont séparées).
    random_state : int ou None, défaut None
        Graine aléatoire pour rendre l'initialisation et le mélange reproductibles.

    Attributs (disponibles après `fit`)
    -----------------------------------
    weights_ : np.ndarray, forme (n_features,)
        Poids appris w.
    bias_ : float
        Biais appris b.
    errors_ : list[int]
        Nombre d'exemples mal classés (= nombre de mises à jour) à chaque époque.
    n_epochs_run_ : int
        Nombre d'époques réellement effectuées.
    converged_ : bool
        True si une époque sans erreur a été atteinte.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 50,
        shuffle: bool = True,
        early_stopping: bool = True,
        random_state: int | None = None,
    ) -> None:
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate doit être compris dans ]0, 1].")
        if not isinstance(n_epochs, int) or n_epochs < 1:
            raise ValueError("n_epochs doit être un entier >= 1.")

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.early_stopping = early_stopping
        self.random_state = random_state

    # Briques de calcul
  
    def net_input(self, X: np.ndarray) -> np.ndarray:
        """Somme pondérée des entrées : z = X·w + b."""
        return np.dot(X, self.weights_) + self.bias_

    @staticmethod
    def activation(z: np.ndarray) -> np.ndarray:
        """Fonction de Heaviside (marche) : 1 si z >= 0, sinon 0."""
        return np.where(z >= 0.0, 1, 0)

    # Entraînement
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """Entraîne le Perceptron avec la règle d'apprentissage de Rosenblatt.

        Paramètres
        ----------
        X : array-like, forme (n_samples, n_features)
            Variables explicatives.
        y : array-like, forme (n_samples,)
            Étiquettes binaires, codées 0 et 1.

        Retour
        ------
        self : l'objet entraîné (permet d'enchaîner `fit(...).predict(...)`).
        """
        X, y = self._validate(X, y)
        rng = np.random.default_rng(self.random_state)

        # Initialisation : petits poids aléatoires (proches de 0) et biais nul
        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias_ = 0.0
        self.errors_: list[int] = []
        self.converged_ = False

        for _ in range(self.n_epochs):
            indices = rng.permutation(len(y)) if self.shuffle else np.arange(len(y))
            n_errors = 0

            for i in indices:
                xi, target = X[i], y[i]
                y_hat = self.activation(self.net_input(xi))  # 1) prédiction
                error = target - y_hat                        # 2) erreur : -1, 0 ou +1
                if error != 0:                                # 3) correction si erreur
                    update = self.learning_rate * error
                    self.weights_ += update * xi
                    self.bias_ += update
                    n_errors += 1

            self.errors_.append(n_errors)

            if n_errors == 0:
                self.converged_ = True
                if self.early_stopping:
                    break

        self.n_epochs_run_ = len(self.errors_)
        return self

    # Prédiction et évaluation
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retourne la classe prédite (0 ou 1) pour chaque exemple."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        return self.activation(self.net_input(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy : proportion de prédictions correctes."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def decision_boundary(self) -> tuple[float, float]:
        """Pour 2 variables : renvoie (pente, ordonnée à l'origine) de la droite w1*x1 + w2*x2 + b = 0."""
        self._check_is_fitted()
        if self.weights_.shape[0] != 2:
            raise ValueError("La frontière n'est représentable comme une droite qu'avec 2 variables.")
        w1, w2 = self.weights_
        if w2 == 0:
            raise ZeroDivisionError("w2 = 0 : la frontière est verticale (x1 = -b / w1).")
        return -w1 / w2, -self.bias_ / w2

    # Utilitaires internes
    @staticmethod
    def _validate(X, y) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.ndim != 2:
            raise ValueError("X doit être un tableau 2D (n_samples, n_features).")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X et y doivent avoir le même nombre d'exemples.")
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("y doit être binaire et codé en 0 / 1.")
        return X, y.astype(int)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "weights_"):
            raise RuntimeError("Le modèle n'est pas entraîné : appelez d'abord fit(X, y).")

    def __repr__(self) -> str:
        return (
            f"Perceptron(learning_rate={self.learning_rate}, n_epochs={self.n_epochs}, "
            f"shuffle={self.shuffle}, early_stopping={self.early_stopping}, "
            f"random_state={self.random_state})"
        )


if __name__ == "__main__":
    # Petit test rapide : la porte logique AND est linéairement séparable
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    model = Perceptron(learning_rate=0.1, n_epochs=20, random_state=42).fit(X_and, y_and)
    print(model)
    print("Prédictions AND :", model.predict(X_and))
    print("Erreurs par époque :", model.errors_)
    print("Convergé :", model.converged_)
