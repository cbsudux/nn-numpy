import numpy as np 

class Linear():

    def __init__(self, in_size, out_size):

        # Xavier init
        self.W = np.random.randn(in_size, out_size) / np.sqrt(in_size + out_size/ 2.)
        self.b = np.zeros((1, out_size))
        self.params = [self.W, self.b]
        self.gradW = None
        self.gradB = None
        self.gradInput = None

    def forward(self, X, mode):
        self.X = X
        output = self.X @ self.W + self.b
        return output

    def backward(self, dout, mode):
        self.gradW = self.X.T @ dout
        self.gradB = np.sum(dout, axis=0)
        self.gradInput = dout @ self.W.T
        return self.gradInput, [self.gradW, self.gradB]

