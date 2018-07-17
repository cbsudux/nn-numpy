import numpy as np
from nn.im2col import *


class Conv2D():

    def __init__(self, X_dim, n_filter, h_filter, w_filter, stride, padding):

        self.d_X, self.h_X, self.w_X = X_dim

        self.n_filter, self.h_filter, self.w_filter = n_filter, h_filter, w_filter
        self.stride, self.padding = stride, padding

        self.W = np.random.randn(
            n_filter, self.d_X, h_filter, w_filter) / np.sqrt(n_filter / 2.)
        self.b = np.zeros((self.n_filter, 1))
        self.params = [self.W, self.b]

        self.h_out = (self.h_X - h_filter + 2 * padding) / stride + 1
        self.w_out = (self.w_X - w_filter + 2 * padding) / stride + 1

        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")

        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.out_dim = (self.n_filter, self.h_out, self.w_out)

    def forward(self, X, mode):

        self.n_X = X.shape[0]

        self.X_col = im2col_indices(
            X, self.h_filter, self.w_filter, stride=self.stride, padding=self.padding)
        W_row = self.W.reshape(self.n_filter, -1)

        output = W_row @ self.X_col + self.b
        output = output.reshape(self.n_filter, self.h_out, self.w_out, self.n_X)
        output = output.transpose(3, 0, 1, 2)
        return output

    def backward(self, dout, mode):

        dout_flat = dout.transpose(1, 2, 3, 0).reshape(self.n_filter, -1)

        gradW = dout_flat @ self.X_col.T
        gradW = gradW.reshape(self.W.shape)

        gradB = np.sum(dout, axis=(0, 2, 3)).reshape(self.n_filter, -1)

        W_flat = self.W.reshape(self.n_filter, -1)

        gradInput_col = W_flat.T @ dout_flat
        shape = (self.n_X, self.d_X, self.h_X, self.w_X)
        gradInput = col2im_indices(gradInput_col, shape, self.h_filter,
                            self.w_filter, self.padding, self.stride)

        return gradInput, [gradW, gradB]

