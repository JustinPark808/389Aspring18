from math import exp
from numpy import sign


class LogReg:
    """
    Class to represent a logistic regression model.
    """

    def __init__(self, l_rate, epochs, n_features):
        """
        Create a new model with certain parameters.

        :param l_rate: Initial learning rate for model.
        :param epoch: Number of epochs to train for.
        :param n_features: Number of features.
        """
        self.l_rate = l_rate
        self.epochs = epochs
        self.coef = [0.0] * n_features
        self.bias = 0.0

    def sigmoid(self, score, threshold=20.0):
        """
        Prevent overflow of exp by capping activation at 20.

        :param score: A real valued number to convert into a number between 0 and 1
        """
        if abs(score) > threshold:
            score = threshold * sign(score)
        activation = exp(score)
        return activation / (1.0 + activation)

    def predict(self, features):
        """
        Given an example's features and the coefficients, predicts the class probability.

        :param features: List of real valued features for a single training example.

        :return: Returns the predicted probability (between 0 and 1).
        """
        score = 0
        for coef, feature in zip(self.coef, features):
            score += coef * feature

        return self.sigmoid(score)

    def sg_update(self, features, label, decay, epoch):
        """
        Computes the update to the weights based on a predicted example.

        :param features: Features to train on.
        :param label: Corresponding label for features.
        """
        # Get predicted value and error
        yhat = self.predict(features)
        error = label - yhat

        # Update bias
        self.bias += self.l_rate * error * yhat * (1 - yhat)

        # Update each coefficient
        for i, feature in enumerate(features):
            self.coef[i] += self.l_rate * error * feature * yhat * (1 - yhat)

        # Update the learning rate
        self.l_rate *= 1 / (1 + decay * epoch)

        return

    def train(self, X, y, decay):
        """
        Computes logistic regression coefficients using stochastic gradient descent.

        :param X: Features to train on.
        :param y: Corresponding label for each set of features.

        :return: Returns a list of model weight coefficients where coef[0] is the bias.
        """
        for epoch in range(self.epochs):
            for features, label in zip(X, y):
                self.sg_update(features, label, decay, epoch)

        return self.bias, self.coef
