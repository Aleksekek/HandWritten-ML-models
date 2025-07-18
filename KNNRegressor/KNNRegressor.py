import numpy as np
import pandas as pd

class MyKNNReg():

    def __init__(self, k = 3, metric = 'euclidean', weight = 'uniform'):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.X, self.y = None, None
        self.train_size = None


    def __repr__(self):
        return f'MyKNNReg class: k={self.k}'


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
            return 1- (divisible/divider)


    def predict(self, X: pd.DataFrame):
        X=X.copy().reset_index(drop=True)
        res = []
        for i in range(X.shape[0]):
            metrick = self.metric_calc(X, self.X, i)
            indexes = pd.Series(metrick).sort_values().index[:self.k]

            if self.weight == 'uniform':
                res.append(np.mean([self.y[num] for num in indexes]))

            if self.weight == 'rank':
                weights = []
                for i in range(1, len(indexes)+1):
                    weight = (1/i)/(np.sum([1/i for i in range(1, len(indexes)+1)]))
                    weights.append(weight)
                res.append(np.sum([self.y[indexes[i]]*weights[i] for i in range(self.k)]))

            if self.weight == 'distance':
                weights = []
                for i in indexes:
                    weight = (1/(metrick[i]+1e-15))/(np.sum([1/(metrick[i]+1e-15) for i in indexes]))
                    weights.append(weight)
                res.append(np.sum([self.y[indexes[i]]*weights[i] for i in range(self.k)]))

        return pd.Series(res)