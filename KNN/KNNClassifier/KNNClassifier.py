import numpy as np
import pandas as pd

class MyKNNClf():

    def __init__(self, k = 3, metric = 'euclidean', weight = 'uniform'):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.X, self.y = None, None
        self.train_size = None


    def __repr__(self):
        return f'MyKNNClf class: k={self.k}'


    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X, self.y = X.copy().reset_index(drop=True), y.copy().reset_index(drop=True)
        self.train_size = self.X.shape


    def metric_calc(self, x1, x2, i):
        if self.metric == 'euclidean':
            return np.sum((x2 - x1.values[i])**2, axis=1)**(1/2)
        if self.metric == 'chebyshev':
            return np.max(np.abs(x2 - x1.values[i]), axis=1)
        if self.metric == 'manhattan':
            return np.sum(np.abs(x2 - x1.values[i]), axis=1)
        if self.metric == 'cosine':
            divisible = np.sum(x2 * x1.values[i], axis=1)
            divider = (np.sum(x2**2, axis=1)**(1/2)) * (np.sum(x1.values[i]**2)**(1/2))
            return 1 - (divisible/divider)


    def predict_proba(self, X: pd.DataFrame):
        X = X.copy().reset_index(drop=True)
        res = []
        for i in range(X.shape[0]):
            metrick_list = self.metric_calc(X, self.X, i)
            var = pd.Series(metrick_list).sort_values().index[:self.k]
            classes = [self.y[i] for i in var]

            if self.weight == 'uniform':
                res.append(np.mean(classes))

            if self.weight == 'rank':
                divisible = 0
                divider = 0
                for i in range(1, len(classes)+1):
                    if classes[i-1] == 1:
                        divisible += 1/i
                    divider += 1/i
                res.append(divisible/divider)

            if self.weight == 'distance':
                divisible = 0
                divider = 0
                for i in range(1, len(classes)+1):
                    if classes[i-1] == 1:
                        divisible += 1/(metrick_list[var[i-1]] + 1e-15)
                    divider += 1/(metrick_list[var[i-1]] + 1e-15)
                res.append(divisible/divider)

        return pd.Series(res)


    def predict(self, X):
        return self.predict_proba(X).apply(lambda x: 1 if x >= 0.5 else 0)