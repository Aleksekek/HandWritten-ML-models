import numpy as np
import pandas as pd
import random

class MySVM():

    def __init__(self, n_iter: int = 10, learning_rate: float = 0.001, C: float = 1, sgd_sample: float | int = None, random_state: int = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.C = C
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.w = None
        self.b = None


    def __repr__(self) -> str:
        return f'MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}, C={self.C}, sgd_sample={self.sgd_sample}, random_state={self.random_state}'


    def fit(self, X: pd.DataFrame, y: pd.DataFrame, verbose: bool | int = False):
        X, y = X.copy(), y.copy()
        if sorted(y.unique()) != [-1, 1]:
            y = y.apply(lambda x: -1 if x == sorted(y.unique())[0] else 1)
        self.w = np.ones(X.shape[1])
        self.b = 1
        random.seed(self.random_state)

        if verbose:
            print(f'start | {self._loss(X, y)}')
        
        for iter in range(1, self.n_iter+1):

            if self.sgd_sample:
                if 0<self.sgd_sample<1:
                    sgd = round(self.sgd_sample * X.shape[0])
                else:
                    sgd = self.sgd_sample
                sample_rows_idx = random.sample(range(X.shape[0]), sgd)
                X_batch, y_batch = X.iloc[sample_rows_idx], y.iloc[sample_rows_idx]
            else:
                X_batch, y_batch = X, y

            for xi in X_batch.iterrows():

                row = xi[0]
                arr = np.array(xi[1])

                if y_batch[row]*(np.dot(arr, self.w) + self.b)>=1:
                    delta_w = 2*self.w
                    delta_b = 0
                else:
                    delta_w = 2*self.w - self.C*y_batch[row]*arr
                    delta_b = -self.C*y_batch[row]
            
                self.w -= self.learning_rate*delta_w
                self.b -= self.learning_rate*delta_b

            if verbose and iter%verbose==0:
                print(f'{iter} | {self._loss(X, y)}') 


    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = X.copy()
        y_pred = np.dot(X, self.w) + self.b
        y_pred = pd.Series(y_pred)
        y_pred = y_pred.apply(lambda x: 0 if x<=0 else 1)
        return y_pred


    def get_coef(self) -> tuple:
        return (self.w, self.b)


    def _hinge_loss(self, X: pd.DataFrame, y: pd.Series) -> float:
        X, y = X.copy(), y.copy()
        y_pred = np.dot(X, self.w) + self.b
        loss = 1 - (y * y_pred)
        return np.mean(loss.apply(lambda x: max(0, x)))


    def _margin_error(self) -> float:
        return min(self.w**2)


    def _loss(self, X: pd.DataFrame, y: pd.Series):
        return self._margin_error() + self.C*self._hinge_loss(X, y)