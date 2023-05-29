import numpy as np

class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y) # find unique elements in the array (to get each of the classes)
        n_classes = len(self._classes)

        # init mean, variance, and prior probabiliites
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64) # for each class we need means for each feature
        self._variance = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[c==y] # getting the samples with this specific class
            self._mean[c, :] = X_c.mean(axis=0) # filling the row for this specific class
            self._variance[c, :] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)
    
    def predict(self, X):
        y_hat = [self._predict(sample) for sample in X]
        return y_har


    def _predict(self, X):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        
        numer = np.exp(- (x - mean)**2 / (2*var) )
        denom = np.sqrt(2* np.pi * var)
        
        return numer/denom