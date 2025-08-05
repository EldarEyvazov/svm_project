# frank_wolfe_svm.py
# This file contains the implementation of the Frank-Wolfe Optimizer for SVM.

import numpy as np
import time
from scipy.sparse import hstack, issparse, vstack, diags

class FrankWolfeOptimizer:
    def __init__(self, C=0.01, initial_solution=None, epsilon=1e-5, max_iterations=1e5):
        self.C = C
        self.initial_solution = initial_solution
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.constraint_matrix = None
        self.objective_value = None
        self.history = None
        self.weights = None
        self.bias = None

    def _initialize_solution(self, dimension):
        x_0 = np.zeros((dimension, 1))
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

    def _objective(self, solution=None):
        if solution is None:
            solution = self.initial_solution
        return (np.linalg.norm(self.constraint_matrix @ solution, ord=2) ** 2).item()

    def _compute_gradient(self):
        A_transpose_dot_x = self.constraint_matrix @ self.initial_solution
        return 2 * (self.constraint_matrix.T @ A_transpose_dot_x)

    def _linear_minimization_oracle(self, gradient):
        min_index = gradient.argmin()
        s = np.zeros_like(gradient)
        s[min_index] = 1
        return s

    def _frank_wolfe_step(self):
        gradient = self._compute_gradient()
        linear_minimizer = self._linear_minimization_oracle(gradient)
        fw_direction = linear_minimizer - self.initial_solution
        dual_gap = -np.dot(gradient.T, fw_direction).item()
        return gradient, linear_minimizer, fw_direction, dual_gap

    def _armijo_line_search(self, fw_direction, dual_gap, max_step_size):
        previous_solution = self.initial_solution
        previous_objective_value = self.objective_value
        alpha, beta = 0.1, 0.8
        step_size = max_step_size
        iterations = 0

        while True:
            new_solution = previous_solution + step_size * fw_direction
            new_objective_value = self._objective(new_solution)
            if new_objective_value <= previous_objective_value + alpha * step_size * (-dual_gap):
                break
            step_size *= beta
            iterations += 1
            if iterations >= 1000:
                break
        
        return new_solution, new_objective_value

    def _calculate_weights_and_bias(self, feature_count):
        weight_bias_vector = self.constraint_matrix @ self.initial_solution
        self.weights = weight_bias_vector[:feature_count]
        self.bias = weight_bias_vector[feature_count]

    def predict(self, features):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        predictions = features @ self.weights + self.bias
        return np.where(predictions > 0, 1, -1)

    def compute_accuracy(self, labels, predictions):
        return np.sum(predictions.flatten() == labels.flatten()) / len(labels)

    def fit(self, features, labels, step_size_strategy="armijo"):
        self.constraint_matrix = self._build_constraint_matrix(features, labels)
        solution_dim = self.constraint_matrix.shape[1]
        feature_count = features.shape[1]

        if self.initial_solution is None:
            self.initial_solution = self._initialize_solution(solution_dim)

        self.objective_value = self._objective()
        self.history = {"iteration": [0], "cpu_time": [0], "objective_value": [self.objective_value], "dual_gap": []}
        start_time = time.time()

        for iteration in range(int(self.max_iterations)):
            grad, s, fw_direction, dual_gap = self._frank_wolfe_step()
            self.history["dual_gap"].append(dual_gap)

            if dual_gap <= self.epsilon:
                print(f"Convergence achieved at iteration {iteration + 1}")
                break

            if step_size_strategy == "armijo":
                self.initial_solution, self.objective_value = self._armijo_line_search(fw_direction, dual_gap, 1.0)
            elif step_size_strategy == "diminishing":
                step_size = 2 / (iteration + 2)
                self.initial_solution += step_size * fw_direction
                self.objective_value = self._objective(self.initial_solution)
            else:
                raise ValueError(f"Unknown step size strategy: {step_size_strategy}")

            elapsed_time = time.time() - start_time
            self.history["iteration"].append(iteration + 1)
            self.history["objective_value"].append(self.objective_value)
            self.history["cpu_time"].append(elapsed_time)

        self._calculate_weights_and_bias(feature_count)
        return self
