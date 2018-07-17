import numpy as np
from nn.im2col import *


class Flatten():

    def __init__(self):
        self.params = []

    def forward(self, X, mode):
        self.X_shape = X.shape
        self.out_shape = (self.X_shape[0], -1)
        output = X.ravel().reshape(self.out_shape)
        self.out_shape = self.out_shape[1]
        return output

    def backward(self, dout, mode):
        output = dout.reshape(self.X_shape)
        return output, ()
