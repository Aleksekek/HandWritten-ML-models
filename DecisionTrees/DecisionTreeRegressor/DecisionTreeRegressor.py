import numpy as np
import pandas as pd

class MyTreeReg():

    def __init__(self, max_depth = 5, min_samples_split = 2, max_leafs = 20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs


    def __repr__(self):
        return f'MyTreeReg class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}'