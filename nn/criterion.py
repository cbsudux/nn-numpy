import numpy as np
from nn.utils import softmax
from nn.layers.linear import *



class CrossEntropyLoss(object):

    def forward(self,X, y):
        self.m = y.shape[0]
        self.p = softmax(X)
        cross_entropy = -np.log(self.p[range(self.m), y])
        loss = np.sum(cross_entropy) / self.m
        return loss

    def backward(self, X, y):
        dx = self.p.copy()
        dx[range(self.m), y] -= 1
        dx /= self.m
        return dx
