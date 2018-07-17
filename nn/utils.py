import numpy as np
import _pickle as cPickle
import gzip
import os


def one_hot_encoded(y, num_class):
    n = y.shape[0]
    onehot = np.zeros((n, num_class), dtype="int32")
    for i in range(n):
        idx = y[i]
        onehot[i][idx] = 1
    return onehot


def check_accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)  # both are not one hot encoded


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def l2_reg(layers, lam=0.001):
    reg_loss = 0.0
    for layer in layers:
        if hasattr(layer, 'W'):
            reg_loss += 0.5 * lam * np.sum(layer.W * layer.W)
    return reg_loss


def delta_l2_reg(layers, grads, lam=0.001):
    for layer, grad in zip(layers, reversed(grads)):
        if hasattr(layer, 'W'):
            grad[0] += lam * layer.W
    return grads



def load_cifar10(path, num_training=1000, num_val=100):
    Xs, ys = [], []
    for batch in range(1, 6):
        f = open(os.path.join(path, "data_batch_{0}".format(batch)), 'rb')
        data = cPickle.load(f, encoding='iso-8859-1')
        f.close()
        X = data["data"].reshape(10000, 3, 32, 32).astype("float64")
        y = np.array(data["labels"])
        Xs.append(X)
        ys.append(y)
        
    f = open(os.path.join(path, "test_batch"), 'rb')
    data = cPickle.load(f, encoding='iso-8859-1')
    f.close()
    X_train, y_train = np.concatenate(Xs), np.concatenate(ys)
    X_test = data["data"].reshape(10000, 3, 32, 32).astype("float")
    y_test = np.array(data["labels"])
    X_train, y_train = X_train[range(
        num_training)], y_train[range(num_training)]
    X_test, y_test = X_test[range(num_val)], y_test[range(num_val)]
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train)
    X_train /= 255.0
    X_test /= 255.0
    return (X_train, y_train), (X_test, y_test)
