import numpy as np
import matplotlib
#matplotlib.use('Agg')
import random
import matplotlib.pyplot as pl

def get_y(I, x, c):
	for i in range(c):
		if I[i][x-1] == 0:
			return i
	return -1

def im2feature(I):
	r, c = I.shape
	features = [int(3*c/8), int(6*c/8)]
	features = [50.0*get_y(I, i, r)/r for i in features]
	d = 4
	v = np.random.randn(d, 1)
	return np.array(features)

def load_data():
	"""
	loads training and testing data
	"""
	# training data
	loc = '../data/train/'
	cls = ['Class1', 'Class2']
	x_training = np.zeros((30, 2), dtype=np.float)
	y_training = np.zeros((30, 1), dtype=np.float)
	k = 0
	for c in np.arange(0, len(cls)):
		for s in np.arange(1, 16):
			I = pl.imread(loc + cls[c] + '_Sample' + str(s) + '.png')
			v = im2feature(I)
			x_training[k,:] = v.T
			y_training[k,0] = 2.0*c - 1
			k = k + 1
	# add all one column to the first axis
	x_training = np.hstack((np.ones((x_training.shape[0],1)), x_training))

	# testing data
	loc = '../data/test/'
	cls = ['Class1', 'Class2']
	x_testing = np.zeros((30, 2), dtype=np.float)
	y_testing = np.zeros((30, 1), dtype=np.float)
	k = 0
	for c in np.arange(0, len(cls)):
		for s in np.arange(1, 16):
			I = pl.imread(loc + cls[c] + '_Sample' + str(s) + '.png')
			v = im2feature(I)
			x_testing[k,:] = v.T
			y_testing[k,0] = 2.0*c - 1
			k = k + 1
	# add all one column to the first axis
	x_testing = np.hstack((np.ones((x_testing.shape[0],1)), x_testing))
	return x_training, y_training, x_testing, y_testing

def normalization_parameters(x):
	"""
	computes and returns normalization parameters (mean and std vectors of each
	column) from data x where it is assumed that the very first column is all
	1s
	"""
	mean_vector = np.mean(x, axis=0)
	mean_vector[0] = 0
	std_vector = np.std(x, axis=0)
	std_vector[0] = 1.0
	return mean_vector, std_vector

def normalize_data(x, mean_vector, std_vector):
	"""
	normalizes the input data x by parameters mean_vector and std_vector
	"""
	x = x - mean_vector
	x = x / std_vector
	return x

def KNN(x_training, y_training, x_testing, K):
	"""
	returns the class label for x_testing considering K nearest neighbours
	on training data (x_training, y_training)
	"""
	d = np.sqrt(np.sum((x_training - x_testing)**2, axis=1))
	i = np.argsort(d)
	s = np.sum(y_training[i[np.arange(K)],:])
	l = (s > 0.0) * 1.0 + (s<=0.0) * -1.0
	return l

def KNN_learn(x, y):
	i = list(range(len(y)))
	random.shuffle(i)
	p = 1.0
	for b in range(1, 20):
		e = KNN_error(b, x[i[:20],:], y[i[:20],:], x[i[20:],:], y[i[20:],:])
		if e == 0.0:
			return b
		if e > p:
			return K-1
		p = e
	return 19

def LR(Theta, x_testing):
	"""
	Theta: parameters of logistic regression type classifier
	x_testing: test data
	returns the class label of x_testing using logistic regression type classifier
	"""
	h = 1.0 / (1.0 + np.exp(-np.dot(x_testing, Theta)))
	l = (h >= 0.5) * 1.0 + (h < 0.5) * (-1.0)
	return l

def stp(t):
	return "%.2f"%(t[0])

def LR_learn(x_training, y_training):
	Theta = np.zeros((x_training.shape[1], 1), dtype=np.float)
	t = 0
	alpha = 0.1
	while True:
		if LR_error(Theta, x_training, y_training) < 1e-5:
			print("Final Theta is ["+", ".join(list(map(stp, Theta)))+"] Finished in %d iterations" % (t))
			break
		e = LR(Theta, x_training) - y_training
		p = np.mean(x_training * e, axis=0)
		Theta -= alpha * p[:, np.newaxis]
		t+=1
	return Theta

def LR_error(Theta, x_testing, y_testing):
	"""
	returns empirical error of logistic regression type classifier on
	(x_testing, y_testing)
	"""
	h = LR(Theta, x_testing)
	e = np.sum(h!=y_testing)
	e = e / np.float(y_testing.shape[0])
	return e

def KNN_error(K, x_training, y_training, x_testing, y_testing):
	h = np.zeros((x_testing.shape[0], 1))
	for n in np.arange(0, x_testing.shape[0]):
		h[n,0] = KNN(x_training, y_training, x_testing[n,:], K)
	e = np.sum(h!=y_testing)
	e = e / np.float(y_testing.shape[0])
	return e

def im2Allfeature(I):
	r, c = I.shape
	features = [int(i*c/8) for i in range(1, 8)]
	features = [50.0*get_y(I, i, r)/r for i in features]
	d = 4
	v = np.random.randn(d, 1)
	return np.array(features)

def visualize():
	loc = '../data/train/'
	cls = ['Class1', 'Class2']
	x_training = np.zeros((30, 7), dtype=np.float)
	y_training = np.zeros((30, 1), dtype=np.float)
	k = 0
	for c in np.arange(0, len(cls)):
		for s in np.arange(1, 16):
			I = pl.imread(loc + cls[c] + '_Sample' + str(s) + '.png')
			v = im2Allfeature(I)
			x_training[k,:] = v.T
			y_training[k,0] = 2.0*c - 1
			k = k + 1
	x_training = np.hstack((np.ones((x_training.shape[0],1)), x_training))
	mean_vector, std_vector = normalization_parameters(x_training)
	x_training = normalize_data(x_training, mean_vector, std_vector)
	colors = [x[0]+2 for x in y_training]
	for i in range(1,8):
		for j in range(i,8):
			X = [x[i] for x in x_training]
			Y = [x[j] for x in x_training]
			pl.xlabel('distance %d'%i)
			pl.ylabel('distance %d'%j)
			pl.title('Scatter plot of distnace taken at index %d vs distnace at index %d' % (i,j))
			pl.scatter(X,Y, c=colors)
			pl.savefig("%d-%d.png"%(i,j))
			pl.clf()

# seed random variable
random.seed(9001)

# visualize features and pick the best ones
# visualize()

# load training and testing data
x_training, y_training, x_testing, y_testing = load_data()

# normalize training and testing data
mean_vector, std_vector = normalization_parameters(x_training)
x_training = normalize_data(x_training, mean_vector, std_vector)
x_testing = normalize_data(x_testing, mean_vector, std_vector)

# learn parameters for KNN and LR
K = KNN_learn(x_training, y_training)
Theta = LR_learn(x_training, y_training)

# Compute training error rates for KNN and Logistic Regression
e = KNN_error(K, x_training, y_training, x_training, y_training)
print('{:d}NN empirical train eror: {:.2f}'.format(K, e))

e = LR_error(Theta, x_training, y_training)
print('Logistic Regression empirical train error: {:.2f}'.format(e))


# Compute testing error rates for KNN and Logistic Regression
e = KNN_error(K, x_training, y_training, x_testing, y_testing)
print('{:d}NN empirical test eror: {:.2f}'.format(K, e))

e = LR_error(Theta, x_testing, y_testing)
print('Logistic Regression empirical test error: {:.2f}'.format(e))
