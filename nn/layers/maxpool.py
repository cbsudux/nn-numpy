import numpy as np
from nn.im2col import *

class Maxpool():

    def __init__(self, X_dim, size, stride):

        self.size = size
        self.stride = stride

        self.d_X, self.h_X, self.w_X = X_dim

        self.params = []

        self.h_out = (self.h_X - size) / stride + 1
        self.w_out = (self.w_X - size) / stride + 1

        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")

        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.out_dim = (self.d_X, self.h_out, self.w_out)

    def forward(self, X, mode):
        self.n_X = X.shape[0]
        X_reshaped = X.reshape(
            X.shape[0] * X.shape[1], 1, X.shape[2], X.shape[3])

        self.X_col = im2col_indices(
            X_reshaped, self.size, self.size, padding=0, stride=self.stride)

        self.max_indexes = np.argmax(self.X_col, axis=0)
        output = self.X_col[self.max_indexes, range(self.max_indexes.size)]

        output = output.reshape(self.h_out, self.w_out, self.n_X,
                          self.d_X).transpose(2, 3, 0, 1)
        return output

    def backward(self, dout, mode):

        gradInput_col = np.zeros_like(self.X_col)
        # flatten the gradient
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()

        gradInput_col[self.max_indexes, range(self.max_indexes.size)] = dout_flat

        # get the original X_reshaped structure from col2im
        shape = (self.n_X * self.d_X, 1, self.h_X, self.w_X)
        gradInput = col2im_indices(gradInput_col, shape, self.size,
                            self.size, padding=0, stride=self.stride)
        gradInput = gradInput.reshape(self.n_X, self.d_X, self.h_X, self.w_X)
        return gradInput, []


