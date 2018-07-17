import numpy as np
from nn.layers.linear import *
from nn.layers.relu import *
from nn.layers.dropout import *
from nn.layers.conv2d import *
from nn.layers.maxpool import *
from nn.layers.flatten import *
from nn.utils import *
from nn.model import NN 
from sklearn.utils import shuffle
import sys
import copy
import pickle
from tqdm import tqdm
import argparse
import shutil


# update for sgd momentum 
def update(velocity, params, grads, learning_rate=0.001, mu=0.9):
    for v, p, g, in zip(velocity, params, reversed(grads)):
        for i in range(len(g)):
            v[i] = mu * v[i] + learning_rate * g[i]
            p[i] -= v[i]


# get minibatches
def minibatch(X, y, minibatch_size):
    n = X.shape[0]
    minibatches = []
    X, y = shuffle(X, y)

    for i in range(0, n , minibatch_size):
        X_batch = X[i:i + minibatch_size, :, :, :]
        # X_batch = X[i:i + minibatch_size, :]

        y_batch = y[i:i + minibatch_size, ]

        minibatches.append((X_batch, y_batch))
    return minibatches


# momentum based sgd
def sgd_momentum(net, X_train, y_train, minibatch_size, epoch, learning_rate, mu=0.9,
                 verbose=True, X_val=None, y_val=None, nesterov=True):
    try:
        val_loss_epoch = []
        minibatches = minibatch(X_train, y_train, minibatch_size)
        minibatches_val = minibatch(X_val, y_val, minibatch_size)

        c = 0 
        for i in range(epoch):
            loss_batch = []
            val_loss_batch = []
            velocity = []
            for param_layer in net.params:
                p = [np.zeros_like(param) for param in list(param_layer)]
                velocity.append(p)

            if verbose:
                print("Epoch {0}".format(i + 1))

            # iterate over mini batches
            for X_mini, y_mini in tqdm(minibatches):

                if nesterov:
                    for param, ve in zip(net.params, velocity):
                        for i in range(len(param)):
                            param[i] += mu * ve[i]

                loss, grads = net.train_step(X_mini, y_mini)
                loss_batch.append(loss)
                update(velocity, net.params, grads,
                                learning_rate=learning_rate, mu=mu)

            for X_mini_val, y_mini_val in tqdm(minibatches_val):
                val_loss, _ = net.train_step(X_mini, y_mini)
                val_loss_batch.append(val_loss)


             # accuracy of model at end of epoch after all mini batch updates   

            if verbose:
                m_train = X_train.shape[0]
                m_val = X_val.shape[0]
                y_train_pred = np.array([], dtype="int64")
                y_val_pred = np.array([], dtype="int64")

                for i in range(0, m_train, minibatch_size):
                    X_tr = X_train[i:i + minibatch_size, : ]
                    y_tr = y_train[i:i + minibatch_size, ]
                    y_train_pred = np.append(y_train_pred, net.predict(X_tr))

                for i in range(0, m_val, minibatch_size):
                    X_va = X_val[i:i + minibatch_size, : ]
                    y_va = y_val[i:i + minibatch_size, ]
                    y_val_pred = np.append(y_val_pred, net.predict(X_va))

                train_acc = check_accuracy(y_train, y_train_pred)
                val_acc = check_accuracy(y_val, y_val_pred)

                mean_train_loss = sum(loss_batch) / float(len(loss_batch))
                mean_val_loss = sum(val_loss_batch) / float(len(val_loss_batch))


                # early stopping with patience = 5 on val loss

                if len(val_loss_epoch) == 0:
                    val_loss_epoch.append(mean_val_loss)
                else:
                    for j in val_loss_epoch[-5:]:
                        if mean_val_loss > j:
                            c += 1
                        else:
                            c = 0
                    if c > 5:
                        print('Early stopping')
                        return net
                    else:
                        c = 0
                        val_loss_epoch.append(mean_val_loss)    


                print("Loss = {0} | Training Accuracy = {1} | Val Loss = {2} | Val Accuracy = {3}".format(
                    mean_train_loss, train_acc, mean_val_loss, val_acc))
        return net
    except KeyboardInterrupt:
        return net




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help = 'name of model to save', default  = 'best_model')
    parser.add_argument('--data_path', help = 'path to dataset', default = 'data/cifar-10')
    args = parser.parse_args()


    print('Creating directory...')
    dir = args.model_name
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.mkdir(dir)


    # Get preprocessed training and validation data
    training_set, val_set = load_cifar10(
        args.data_path, num_training=50000, num_val=1000)

    X_train, y_train = training_set
    X_val, y_val= val_set

    print(X_train.shape)
    print(X_val.shape)


    # define neural net
    nn = NN()

    nn.add_layer(Conv2D((3,32,32), n_filter=16, h_filter=5,
                w_filter=5, stride=1, padding=2))
    nn.add_layer(ReLU())
    nn.add_layer(Maxpool(nn.layers[0].out_dim, size=2, stride=2))

    nn.add_layer(Conv2D(nn.layers[2].out_dim, n_filter=20, h_filter=5,
                 w_filter=5, stride=1, padding=2))
    nn.add_layer(ReLU())
    nn.add_layer(Maxpool(nn.layers[3].out_dim, size=2, stride=2))

    nn.add_layer(Flatten())
    nn.add_layer(Linear(np.prod(nn.layers[5].out_dim), 100))
    nn.add_layer(ReLU())
    nn.add_layer(Dropout(0.5))
    nn.add_layer(Linear(100, 10))
    

    # resume?
    resume = 0
    if resume:
        print('Resuming..')
        filename = 'cnn.nn'
        nn = pickle.load(open(filename, 'rb'))    
        nn = sgd_momentum(nn, X_train , y_train, minibatch_size=50, epoch=200,
                       learning_rate=0.001, X_val=X_val, y_val=y_val)

    else:
        nn = sgd_momentum(nn, X_train , y_train, minibatch_size=50, epoch=200,
                   learning_rate=0.001, X_val=X_val, y_val=y_val)


    # Saving model
    print('Saving model')
    filename = os.path.join(args.model_name, args.model_name +'.nn')
    pickle.dump(nn, open(filename, 'wb'))
