import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, n_iter=1000):   # default values for lr and n_iter
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    # for training (on dataset X with corresponding truth values y)
    def fit(self, X, y):
        # X dimension n_samples (rows), n_features (columns - coordinates for example)
        n_samples, n_features = X.shape

        # initialize weights and bias to be zero
        self.weights = np.zeros(n_features)
        self.bias = 0

        # loss = y_hat - y 
        # MSE = 1/n ∑ (y_hat - y)^2 = 1/n ∑ ((wx+b) - y)^2
        # derivatives: dMSE/dw = 2/n ∑ x((wx+b)-y)  dMSE/db = 2/n ∑ ((wx+b)-y)

        # apply gradient descent
        for _ in range(self.n_iter):
            y_hat = np.dot(X, self.weights) + self.bias # prediction

            dw = (2/n_samples) * np.dot(X.T, (y_hat - y))
            db = (2/n_samples) * np.sum((y_hat - y))

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    # predict values for new dataset X using our fitted model
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
