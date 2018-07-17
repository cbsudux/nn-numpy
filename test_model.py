import numpy as np
import argparse
import os
import _pickle as cPickle
import pickle
from nn.utils import check_accuracy



def test( X_test, y_test ,filename):

	print('Loading saved model')
	nn = pickle.load(open(filename, 'rb'))

	# nn in test mode
	nn.mode = 'test'

	print('Testing...')
	y_test_pred = nn.predict(X_test)
	test_acc = check_accuracy( y_test , y_test_pred)

	print(test_acc)



# get test data
def get_test_data(testdata):

	print('Preparing test data')
	f = open(testdata, 'rb')
	data = cPickle.load(f, encoding='iso-8859-1')
	f.close()
	X_test = data["data"].reshape(10000, 3, 32, 32).astype("float")
	y_test = np.array(data["labels"])
	X_test /= 255.0

	return (X_test, y_test)



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help = 'path to saved model', default = 'best_model/best_model.nn')
	parser.add_argument('--testdata', help = 'path to test data', default = 'data/cifar-10/test_batch')
	args = parser.parse_args()

	X_test,y_test = get_test_data(args.testdata)

	test(X_test,y_test, args.model)







