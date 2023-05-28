import numpy as np

class LogisticRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        #initialize our parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_hat = self.__sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T, (y_hat - y))
            db = (1/n_samples) * np.sum(y_hat - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
        linear_mod = np.dot(X, self.weights) + self.bias
        y_hat = self.__sigmoid(linear_mod) # get the probability value

        # classification step - if y_hat > 0.5 then its class 1
        y_pred_class = [1 if i > 0.5 else 0 for i in y_hat]
        return y_pred_class

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    