import numpy as np


def normalize(features, axis):
    mins = (np.min(features, axis=axis)).reshape(-1, 1)
    maxs = (np.max(features, axis=axis)).reshape(-1, 1)
    denominator = maxs - mins
    denominator[denominator == 0] = 1
    means = (np.divide((features - mins), denominator))
    return means


def standardize(features, axis):
    mean = np.mean(features, axis=axis).reshape(-1, 1)
    std = np.std(features, axis=axis).reshape(-1, 1)
    std[std == 0] = 1  # preventing divide by 0
    return ((features.T - mean) / std).T


def initialize_weights(degree):
    return np.array([0.1] * degree).reshape(-1, 1).T


def get_cost(output, predictions):
    return np.sqrt(np.sum(np.square(output - predictions)))


def add_constant_row(original_array, size_of_added_row):
    constant_row = np.array([1] * size_of_added_row).reshape(-1, 1).T
    desired_ar = np.row_stack((constant_row, original_array))
    return desired_ar


class Regression:
    def __init__(self, features, output):
        """
        :param features: features should be given (nx,m) numpy array
        :param output: output should be given (1,m) numpy array
        """
        self.weights = None
        self.cost_history = None
        #  checking type
        if type(features) == np.ndarray and type(output) == np.ndarray:
            #  checking sizes
            if len(features.shape) == 1:
                features = features.reshape(-1, 1).T
            if len(output.shape) == 1:
                output = output.reshape(-1, 1).T
            if output.shape[0] != 1:
                raise Exception("Output array should be given in (1,m) numpy array form.")
            m = output.shape[1]
            if features.shape[1] != m:
                raise Exception("Feature array should be given in (nx,m) numpy array form.")
            self.features = features
            self.output = output
        else:
            raise Exception("Features and output variables should be numpy array.")

    def predict_output(self):  # (nx,m) and (nx, 1)
        return np.dot(self.weights.T, self.features)

    def predict_output_with_given_features(self, features):
        standardized_f = standardize(features, axis=1)
        constant_row_added_f = add_constant_row(standardized_f, features.shape[1])
        return np.dot(self.weights.T, constant_row_added_f).T

    def feature_derivative(self, errors):  # (nx,m) (1,m)
        return 2 * np.dot(self.features, errors.T)

    def least_square_method(self):
        A = np.linalg.pinv(np.dot(self.features, self.features.T))
        M = np.dot(A, self.features)
        print(np.dot(M, self.output.T))
        self.weights = np.dot(M, self.output.T)

    def gradient_descent_method(self, learning_rate, num_iter):
        self.cost_history = [0] * num_iter
        iter = 0
        nx = self.features.shape[0]
        initial_weights = initialize_weights(nx).T
        self.weights = np.array(initial_weights)  # make sure it's a numpy array
        m = self.output.shape[1]
        while iter < num_iter:
            predictions = self.predict_output()
            errors = predictions - self.output
            derivatives = self.feature_derivative(errors) / m
            self.weights = self.weights - learning_rate * derivatives
            cost = get_cost(self.output, predictions)
            self.cost_history[iter] = cost
            iter += 1

    def arrange_features(self):
        self.features = standardize(self.features, 1)
        self.features = add_constant_row(self.features, self.output.shape[1])
        print(self.features)

    def train_linear_model(self, method="GD", learning_rate=1e-3, num_iter=500):
        self.arrange_features()
        if method == "GD":
            self.gradient_descent_method(learning_rate, num_iter)
        elif method == "LS":
            self.least_square_method()
