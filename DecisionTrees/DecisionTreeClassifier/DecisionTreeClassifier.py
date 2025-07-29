import numpy as np
import pandas as pd

class MyTreeClf():

    def __init__(self, max_depth = 5, min_samples_split = 2, max_leafs = 20, bins: int = None, criterion: str = 'entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.criterion = criterion
        self.leafs_cnt = 0
        self.depth = 0


    def __repr__(self):
        return f'MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}'


    def _entropy(self, y):
        return -np.sum([(i/len(y))*np.log2(i/len(y)) if i>0 else 0 for i in y.value_counts(sort=False)])
    

    def _gini(self, y):
        return 1 - (np.sum([(i/len(y))**2 if i>0 else 0 for i in y.value_counts(sort=False)]))


    def _get_best_split(self, X, y):
        X, y = X.copy(), y.copy()
        if self.criterion == 'entropy':
            S0 = self._entropy(y)
        elif self.criterion == 'gini':
            Gp = self._gini(y)
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
                if self.criterion == 'entropy':
                    S_right = (len(split_right)/len(y))*self._entropy(split_right)
                    S_left = (len(split_left)/len(y))*self._entropy(split_left)
                    gain = S0 - S_right - S_left
                elif self.criterion == 'gini':
                    gini_right = (len(split_right)/len(y))*self._gini(split_right)
                    gini_left = (len(split_left)/len(y))*self._gini(split_left)
                    gain = Gp - gini_left - gini_right
                if gain > max_gain:
                    col_name, split_value, max_gain = col, divider, gain
        return col_name, split_value, max_gain


    def _is_leaf(self, X, y):
        if len(X) == 1:
            return True
        if len(X) < self.min_samples_split:
            return True
        if self._entropy(y) == 0:
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
    

    def _split(self, X: pd.DataFrame, y: pd.Series, data: pd.Series, side: str, depth: int, verbose: bool, lif_t: int):
        if not self._is_leaf(X[data], y[data]) and self.leafs_cnt + self.leafs_checker() + 1 <= self.max_leafs and depth < self.max_depth:
            if verbose:
                print(side.lower(), len(X[data]), depth+1)
            self._splitter(X[data], y[data], verbose, depth, lif_t+1)
        else:
            if verbose:
                print(f'\n{side} Done | size: {len(X[data])} | entrope: {self._entropy(y[data])} | depth: {depth+1} | val: {np.sum(y[data])/len(y[data])}\n')
            self.leaf_sum += np.sum(y[data])/len(y[data])
            self.leafs_cnt += 1
            if not self.model[depth+1]:
                self.model[depth+1] = [[0, 0] for _ in range(2**(depth))]
            self.model[depth+1][2**(depth) - self.rest[depth+1]] = sum(y[data])/len(y[data])
            for i in range(self.max_depth - depth + 1):
                self.rest[depth + i + 1] -= 2**i


    def _splitter(self, X, y, verbose, depth = 0, lif_t=-1):
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

        if self.criterion == 'entropy':
            self.fi[split[0]] += (len(X)/self.len_X) * (self._entropy(y) - len(y[left])/len(X)*self._entropy(y[left]) - len(y[right])/len(X)*self._entropy(y[right]))
        elif self.criterion == 'gini':
            self.fi[split[0]] += (len(X)/self.len_X) * (self._gini(y) - len(y[left])/len(X)*self._gini(y[left]) - len(y[right])/len(X)*self._gini(y[right]))
        
        self._split(X, y, left, 'Left', depth, verbose, lif_t)

        self._split(X, y, right, 'Right', depth, verbose, 0)


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

        for col in X:
            self.fi[col] = 0

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

        depth = 0
        while depth < self.max_depth + 1:
            depth += 1
            self.model[depth] = []
            self.rest[depth] = 2**(depth-1)
       
        self._splitter(X, y, verbose)


    def predict_proba(self, X: pd.DataFrame):
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


    def predict(self, X: pd.DataFrame):
        X = X.copy()
        y_proba = self.predict_proba(X)
        y = y_proba.apply(lambda x: 1 if x > 0.5 else 0)
        return y
    

    def feature_importances_(self):
        return self.fi


    def print_tree(self):
        for layer in self.model:
            if self.model[layer]:
                print(f'{layer} - {self.model[layer]}')
        print(self.leafs_cnt)
        print(round(self.leaf_sum, 6))