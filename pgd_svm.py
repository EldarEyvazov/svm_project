# pgd_svm.py
# This file contains the implementation of the Projected Gradient Descent method for SVM.

import numpy as np
import time
from scipy.sparse import issparse, diags, hstack, vstack
from scipy.spatial import distance

class PGD:
    def __init__(self, C=0.1, epsilon=1e-6, s=0.1, max_iterations=1000, step_method="armijo"):
        self.C = C
        self.epsilon = epsilon
        self.s = s
        self.max_iterations = max_iterations
        self.step_method = step_method
        self.A = None
        self.history = None
        self.weights = None
        self.bias = None

    def _initialize_solution(self, n):
        x_0 = np.zeros(n)
        x_0[0] = 1.0
        return x_0

    def _add_bias_to_features(self, features):
        num_samples = features.shape[0]
        if issparse(features):
            bias_column = np.ones((num_samples, 1))
            return hstack([features, bias_column])
        else:
            return np.concatenate((features, np.ones((num_samples, 1))), axis=1)

    def _build_constraint_matrix(self, features, labels):
        features_with_bias = self._add_bias_to_features(features)
        labels = labels.reshape(-1, 1)

        if issparse(features_with_bias):
            weighted_features = features_with_bias.multiply(labels)
        else:
            weighted_features = features_with_bias * labels

        identity_matrix = (1 / np.sqrt(self.C)) * np.eye(features_with_bias.shape[0])

        if issparse(weighted_features):
            constraint_matrix = vstack([weighted_features.T, identity_matrix])
        else:
            constraint_matrix = np.concatenate((weighted_features.T, identity_matrix), axis=0)
        
        return constraint_matrix

    def _objective_function(self, x):
        return np.linalg.norm(self.A @ x, ord=2) ** 2

    def _gradient(self, x):
        A_transpose_dot_x = self.A @ x
        return 2 * (self.A.T @ A_transpose_dot_x)

    def _calculate_weights_and_bias(self, x, feature_count):
        weight_bias_vector = self.A @ x
        self.weights = weight_bias_vector[:feature_count]
        self.bias = weight_bias_vector[feature_count]

    def predict(self, features):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        predictions = features @ self.weights + self.bias
        return np.where(predictions > 0, 1, -1)

    def compute_accuracy(self, labels, predictions):
        return np.sum(predictions.flatten() == labels.flatten()) / len(labels)

    def _armijo_line_search(self, x, dk, initial_gamma=1, beta=0.8, sigma=0.1):
        gamma = initial_gamma
        while True:
            lhs = self._objective_function(x + gamma * dk)
            rhs = self._objective_function(x) + sigma * gamma * np.dot(self._gradient(x).T, dk)
            if lhs <= rhs:
                break
            gamma *= beta
        return gamma

    def _project_onto_unit_simplex(self, y):
        n = y.shape[0]
        u = np.sort(y)[::-1]
        cum_sum = np.cumsum(u) - 1
        indices = np.arange(1, n + 1)
        k = np.max(np.where((cum_sum) / indices < u)[0])
        tau = (np.sum(u[:k + 1]) - 1) / (k + 1)
        return np.maximum(y - tau, 0)

    def fit(self, X_train, y_train):
        self.A = self._build_constraint_matrix(X_train, y_train)
        features_count = X_train.shape[1]
        n = self.A.shape[1]
        x = self._initialize_solution(n)

        self.history = {
            "iteration": [0],
            "cpu_time": [0],
            "objective_value": [self._objective_function(x)],
            "gradient_mapping": []
        }
        start_time = time.time()

        for k in range(1, self.max_iterations + 1):
            x_hat = self._project_onto_unit_simplex(x - self.s * self._gradient(x))
            g = distance.euclidean(x, x_hat)

            objective_value = self._objective_function(x)
            current_time = time.time() - start_time

            self.history["iteration"].append(k)
            self.history["cpu_time"].append(current_time)
            self.history["objective_value"].append(objective_value)
            self.history["gradient_mapping"].append(g)

            if g <= self.epsilon:
                print(f"Stopping criteria reached at iteration {k}")
                break

            dk = x_hat - x

            if self.step_method == "armijo":
                gamma = self._armijo_line_search(x, dk)
            else:
                raise ValueError("Invalid step_method.")

            x = x + gamma * dk

        self._calculate_weights_and_bias(x, features_count)
        return self
