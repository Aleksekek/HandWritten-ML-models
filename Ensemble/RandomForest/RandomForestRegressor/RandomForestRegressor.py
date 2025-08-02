import numpy as np
import pandas as pd
import random

class MyForestReg():

    def __init__(self,
            n_estimators: int = 10,
            max_features: float = 0.5,
            max_samples: float = 0.5,
            max_depth: int = 5,
            min_samples_split: int = 2,
            max_leafs: int = 20,
            bins: int = 16,
            random_state: int = 42):
        
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.random_state = random_state
        self.models = []
        self.leafs_cnt = 0


    def __repr__(self):
        return f'MyForestReg class: n_estimators={self.n_estimators}, max_features={self.max_features}, max_samples={self.max_samples}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}, random_state={self.random_state}'
    

    def fit(self, X: pd.DataFrame, y: pd.Series):
        
        X, y = X.copy(), y.copy()
        init_cols = X.columns.to_list()
        init_rows_cnt = len(X)
        random.seed(self.random_state)

        for estimator in range(self.n_estimators):

            cols_smpl_cnt = round(self.max_features * X.shape[1])
            rows_smpl_cnt = round(self.max_samples * X.shape[0])

            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(range(init_rows_cnt), rows_smpl_cnt)

            X_batch, y_batch = X[cols_idx].iloc[rows_idx], y.iloc[rows_idx]

            tree = MyTreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins)
            tree.fit(X_batch, y_batch)

            self.leafs_cnt += tree.leafs_cnt
            self.models.append(tree)



class MyTreeReg():

    def __init__(self, max_depth = 5, min_samples_split = 2, max_leafs = 20, bins: int = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins


    def __repr__(self):
        return f'MyTreeReg class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}'
    
    
    def _mse(self, y: pd.Series):
        return ((y - [y.mean()] * len(y))**2).mean()
        

    def _get_best_split(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        X, y = X.copy(), y.copy()
        mse = self._mse(y)
        max_gain = -1e15

        for col in X:
            if self.dividers and type(self.dividers[col]) != bool:
                dividers = self.dividers[col]
            else:
                values = sorted(X[col].unique())
                dividers = [(values[i]+values[i+1])/2 for i in range(len(values)-1)]

            for divider in dividers:
                split_right = y[X[col] > divider]
                split_left = y[X[col] <= divider]

                mse_right = (len(split_right)/len(y)) * self._mse(split_right)
                mse_left = (len(split_left)/len(y)) * self._mse(split_left)
                gain = mse - mse_left - mse_right

                if gain > max_gain:
                    col_name, split_value, max_gain = col, divider, gain
                        
        return col_name, split_value, max_gain


    def _is_leaf(self, X, y):
        if len(X) == 1:
            return True
        if len(X) < self.min_samples_split:
            return True
        if self._mse(y) == 0:
            return True
        if self.bins:
            for col in X:
                if type(self.dividers[col])!=bool:
                    for var in self.dividers[col]:
                        if sum(X[col]>var)>0 and sum(X[col]<var)>0:
                            return False
            return True
        return False
    

    def leafs_checker(self):
        expected_leafs = {}
        for depth_num, depth in self.model.items():
            expected_leafs[depth_num] = 0
            total_expected_leafs = 0
            for split in depth:
                if split == [0, 0]:
                    continue
                if isinstance(split, list):
                    expected_leafs[depth_num] +=2
                    if depth_num>1:
                        expected_leafs[depth_num-1] -=1
                else:
                    expected_leafs[depth_num] -= 1
            for _, leafs in expected_leafs.items():
                total_expected_leafs+=leafs
        return total_expected_leafs


    def _split(self, X: pd.DataFrame, y: pd.Series, data: pd.Series, side: str, depth: int, verbose: bool):
        if not self._is_leaf(X[data], y[data]) and self.leafs_cnt + self.leafs_checker() + 1 <= self.max_leafs and depth < self.max_depth:
            if verbose:
                print(side.lower(), len(X[data]), depth+1)
            self._splitter(X[data], y[data], verbose, depth)
        else:
            if verbose:
                print(f'\n{side} Done | size: {len(X[data])} | mse: {self._mse(y[data])} | depth: {depth+1} | val: {np.sum(y[data])/len(y[data])}\n')
            self.leaf_sum += np.sum(y[data])/len(y[data])
            self.leafs_cnt += 1
            if not self.model[depth+1]:
                self.model[depth+1] = [[0, 0] for _ in range(2**(depth))]
            self.model[depth+1][2**(depth) - self.rest[depth+1]] = sum(y[data])/len(y[data])
            for i in range(self.max_depth - depth + 1):
                self.rest[depth + i + 1] -= 2**i


    def _splitter(self, X, y, verbose, depth = 0):
        depth += 1
        if not self.model[depth]:
            self.model[depth] = [[0, 0] for _ in range(2**(depth-1))]
        split = self._get_best_split(X, y)
        self.model[depth][2**(depth-1) - self.rest[depth]] = list(split[:2])
        self.rest[depth] -= 1
        if verbose:
            print(split)
        left = X[split[0]] <= split[1]
        right = X[split[0]] > split[1]

        self.fi[split[0]] += (len(X)/self.len_X) * (self._mse(y) - len(y[left])/len(X)*self._mse(y[left]) - len(y[right])/len(X)*self._mse(y[right]))
        
        self._split(X, y, left, 'Left', depth, verbose)

        self._split(X, y, right, 'Right', depth, verbose)


    def fit(self, X: pd.DataFrame, y: pd.Series, verbose = False):
        X, y = X.copy(), y.copy()
        self.leafs_cnt = 0
        self.depth = 0
        self.model = {}
        self.rest = {}
        self.leaf_sum = 0
        self.dividers = {}
        self.fi = {}
        self.len_X = len(X)

        if self.bins:
            self.dividers = {}
            for col in X:
                values = sorted(X[col].unique())
                dividers = [(values[i]+values[i+1])/2 for i in range(len(values)-1)]
                hists = np.histogram(X[col], self.bins)[1][1:-1]
                if len(dividers) <= self.bins-1:
                    self.dividers[col] = False
                else:
                    self.dividers[col] = hists

        for col in X:
            self.fi[col] = 0

        depth = 0
        while depth < self.max_depth + 1:
            depth += 1
            self.model[depth] = []
            self.rest[depth] = 2**(depth-1)
       
        self._splitter(X, y, verbose)


    def predict(self, X: pd.DataFrame):
        X = X.copy()
        y = []
        for row in X.iterrows():
            index = 0
            for depth in range(1, self.max_depth+2):
                if type(self.model[depth][index]) == type(y):
                    col, val = self.model[depth][index][0], self.model[depth][index][1]
                    if row[1][col] <= val:
                        index *= 2
                    else:
                        index *= 2
                        index += 1
                else:
                    y.append(self.model[depth][index])
                    break
        return pd.Series(y)


    def feature_importances_(self):
        return self.fi


    def print_tree(self):
        for layer in self.model:
            if self.model[layer]:
                print(f'{layer} - {self.model[layer]}')
        print(self.leafs_cnt)
        print(round(self.leaf_sum, 6))