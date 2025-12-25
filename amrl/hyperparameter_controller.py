import numpy as np

class HPController:
    def __init__(self, a=3e-4, g=0.99, b=0.01):
        self.a0 = a
        self.a, self.g, self.b = a, g, b

    def update(self, d):
        da, dg, db = d
        self.a = np.clip(self.a * np.exp(da), self.a0/10, self.a0*10)
        self.g = np.clip(self.g + dg, 0.90, 0.99)
        self.b = np.clip(self.b + db, 0.0, 1.0)

    def get(self):
        return self.a, self.g, self.b
