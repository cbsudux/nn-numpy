import numpy as np

class Dropout():

    def __init__(self, prob=0.5, mode = 'train'):
        self.prob = prob
        self.params = []
        self.mode = mode

    def forward(self, X, mode):

        # inverted dropout

        if self.mode == 'train':
            self.mask = np.random.binomial(1, self.prob, size=X.shape) / self.prob
            output = X * self.mask
            return output.reshape(X.shape)
        elif self.mode == 'test':
            return X

    def backward(self, dout, mode):
        
        if self.mode == 'train':
            gradInput = dout * self.mask
            return gradInput, []
        elif self.mode == 'test':
            return dout, []

