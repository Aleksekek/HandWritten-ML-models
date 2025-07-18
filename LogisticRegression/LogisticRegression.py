import numpy as np
import pandas as pd

import random

class MyLogReg():

    def __init__(self, n_iter = 100, learning_rate = 0.01, metric = None, reg = None, l1_coef = 0, l2_coef = 0, sgd_sample = None, random_state = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        random.seed(random_state)
        self.weights = None


    def __repr__(self) -> str:
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False):
        X=X.copy()
        X.reset_index(inplace=True, drop=True)
        X.insert(loc=0, column='x0', value=1)
        self.weights = np.ones(X.shape[1])

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

            pred = self.sigmoid(np.dot(X_batch, self.weights))
            grad = np.dot((pred - y_batch), X_batch)/len(y_batch)

            if self.reg:
                if self.reg == 'l1':
                    grad += self.l1_coef * np.sign(self.weights)
                if self.reg == 'l2':
                    grad += self.l2_coef * 2 * self.weights
                if self.reg == 'elasticnet':
                    grad += self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights

            if callable(self.learning_rate):
                lr = self.learning_rate(iter)
            else:
                lr = self.learning_rate

            self.weights -= grad*lr

            if verbose:

                if self.reg:
                    if self.reg == 'l1':
                        reg = self.l1_coef * np.sum(np.abs(self.weights))
                    if self.reg == 'l2':
                        reg = self.l2_coef * np.sum((self.weights)**2)
                    if self.reg == 'elasticnet':
                        reg = self.l1_coef * np.sum(np.abs(self.weights))
                        reg += self.l2_coef * np.sum((self.weights)**2)
                else:
                    reg = 0
                
                if iter == 1:
                    pred = self.sigmoid(np.dot(X, np.ones(X.shape[1])))
                    loss = -np.mean(y*np.log(pred) + (1-y)*np.log(1-pred)) + reg
                    res = f'start | loss: {loss}'
                    print_flag = True

                if iter%verbose==0:
                    pred = self.sigmoid(np.dot(X, self.weights))
                    loss = -np.mean(y*np.log(pred+1e-15) + (1-y)*np.log(1-pred+1e-15)) + reg
                    res = f'{iter} | loss: {loss}'
                    print_flag = True

                if self.metric and print_flag:
                    metric_val = self.metric_calc(pred, y)
                    res += f' | {self.metric}: {metric_val}'

                if callable(self.learning_rate) and print_flag :
                    res += f' | lr: {lr}'

                if print_flag:
                    print(res)
                    print_flag = False

    
    def predict_proba(self, X:pd.DataFrame):
        X = X.copy()
        X.insert(loc=0, column='x0', value=1)
        return pd.Series(self.sigmoid(np.dot(X, self.weights)))
    

    def predict(self, X:pd.DataFrame):
        pred = self.predict_proba(X)
        return pred.apply(lambda x: 1 if x>0.5 else 0)
    

    def metric_calc(self, pred: pd.Series, y: pd.Series):
        if self.metric != 'roc_auc':
            pred = pd.Series(pred).apply(lambda x: 1 if x>0.5 else 0)
            TP, TN, FP, FN = 0, 0, 0, 0
            for i in zip(y, pred):
                if i[0]==1 and i[1]==1:
                    TP+=1
                elif i[0]==0 and i[1]==0:
                    TN+=1
                elif i[0]==0 and i[1]==1:
                    FP+=1
                else:
                    FN+=1

        if self.metric == 'accuracy':
            return (TP+TN)/(TP+TN+FP+FN)
        if self.metric == 'precision':
            return TP/(TP+FP+1e-15)
        if self.metric == 'recall':
            return TP/(TP+FN+1e-15)
        if self.metric == 'f1':
            self.metric = 'precision'
            pr = self.metric_calc(pred, y)
            self.metric = 'recall'
            r =  self.metric_calc(pred, y)
            self.metric = 'f1'
            return (2*pr*r)/(pr+r+1e-15)
        if self.metric == 'roc_auc':
            zippo = sorted(zip(pred, y), reverse=True)
            pred, y = zip(*zippo)
            total = 0
            for i in range(len(y)):
                if y[i]==0:
                    count = sum(y[:i])
                    for j in pred[:i][::-1]:
                        if pred[i] == j:
                            count-= 0.5
                        if j>y[i]:
                            break
                    total+=count
            P, N = sum(y), len(y) - sum(y)
            return total/(P*N)


    def sigmoid(self, value):
        return 1/(1+np.exp(-value))
    

    def get_coef(self):
        return self.weights[1:]
    

    def get_best_score(self, X = None, y = None):
        return self.metric_calc(self.predict_proba(X), y)