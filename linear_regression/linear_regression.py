import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, n_iter=1000):   # default values for lr and n_iter
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    # for training (on dataset X with corresponding truth values y)
    def fit(self, X, y):
        pass
        for i in range(X):
            # get the prediction
            y_hat = np.dot(np.transpose(self.weights), self.X[0])
            # compare the prediction to the actutal value
            loss = 0.5*(y_hat - y[i])**2

    # predict values for new dataset X using our fitted model
    def predict(self, X):
        pass
