import numpy as np
import torch

class MetaState:
    def __init__(self, window=1000):
        self.window = window
        self.r, self.pl, self.vl, self.gn = [], [], [], []

    def update(self, r, pl, vl, gn):
        for arr, val in zip([self.r, self.pl, self.vl, self.gn],
                            [r, pl, vl, gn]):
            arr.append(val)
            if len(arr) > self.window:
                arr.pop(0)

    def get(self):
        def m(x): return np.mean(x) if x else 0.0
        def s(x): return np.std(x) if x else 0.0
        return torch.tensor([
            m(self.r), s(self.r),
            m(self.pl), m(self.vl),
            m(self.gn), s(self.gn)
        ], dtype=torch.float32)
