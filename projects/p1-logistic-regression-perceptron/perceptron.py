class Perceptron:
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

    def predict(self, features):
        """
        Given an example's features and the coefficients, predicts the class.

        :param features: List of real valued features for a single training example.

        :return: Returns the predicted class (either 0 or 1).
        """
        score = 0
        for coef, feature in zip(self.coef, features):
            score += coef * feature

        return 0 if score < 0 else 1

    def sg_update(self, features, label, decay, epoch):
        """
        Computes the update to the weights based on a predicted example.

        :param features: Features to train on.
        :param label: Corresponding label for features.
        """
        yhat = self.predict(features)
        error = label - yhat

        # Update bias
        self.bias += self.l_rate * error

        # Update each coefficient
        for i, feature in enumerate(features):
            self.coef[i] += self.l_rate * error * feature

        # Update learning rate
        self.l_rate *= 1 / (1 + decay * epoch)

        return

    def train(self, X, y, decay):
        """
        Trains the model on training data.

        :param X: Features to train on.
        :param y: Corresponding label for each set of features.
        """
        for epoch in range(self.epochs):
            for features, label in zip(X, y):
                self.sg_update(features, label, decay, epoch)
        return self.bias, self.coef
