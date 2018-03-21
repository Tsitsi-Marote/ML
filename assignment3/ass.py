
import numpy as np

def h(Theta, X, n):
	h = 2.0 / (1.0 + np.exp(-np.dot(X, Theta))) - 1.0
	l = (h >= 0.0) * 1.0 + (h < 0.0) * (-1.0)
	print("%d&%f&%d"%(n, h, l))

def KNN(x_training, y_training, x_testing, K):
	d = np.sqrt(np.sum((x_training - x_testing)**2, axis=1))
	i = np.argsort(d)
	s = np.sum(y_training[i[np.arange(K)],:])
	l = (s > 0.0) * 1.0 + (s<=0.0) * -1.0
	aa = np.transpose([d[i[np.arange(K)]]])
	bb = np.transpose([i[np.arange(K)]])
	cc = y_training[i[np.arange(K)],:]
	c = np.concatenate((bb, aa, cc), axis=1)
	#print(c)
	return l

def KNN_error(K, x_training, y_training, x_testing, y_testing):
	h = np.zeros((x_testing.shape[0], 1))
	for n in np.arange(0, x_testing.shape[0]):
		h[n,0] = KNN(x_training, y_training, x_testing[n,:], K)
	e = np.sum(h!=y_testing)
	e = e / np.float(y_testing.shape[0])
	return e

b = np.loadtxt("banknote_testing_data.txt", delimiter=",")
a = np.loadtxt("banknote_training_data.txt", delimiter=",")

x_tr = a[:, :-1]
x_tr = np.hstack((np.ones((x_tr.shape[0],1)), x_tr))
y_tr = a[:, -1:len(a[0])]
x_ts = b[:, :-1]
x_ts = np.hstack((np.ones((x_ts.shape[0],1)), x_ts))
y_ts = b[:, -1:len(b[0])]

q1 = np.array([1.,3.618100,-3.745400,2.827300,-0.712080])
print("Question 1")
print(KNN(x_tr, y_tr, q1, 1))

q2 = np.array([1.,-2.436500,3.602600,-1.416600,-2.894800])
print("\nQuestion 2")
print(KNN(x_tr, y_tr, q2, 3))

q3 = np.array([1.,-4.366700,6.069200,0.572080,-5.466800])
print("\nQuestion 3")
print(KNN(x_tr, y_tr, q3, 5))

print("\nQuestion 4")
print(KNN_error(1, x_tr, y_tr, x_ts, y_ts))
print(KNN_error(3, x_tr, y_tr, x_ts, y_ts))
print(KNN_error(5, x_tr, y_tr, x_ts, y_ts))

print("\nQuestion 2.3")
Theta = np.transpose([np.array([1.47791190,-1.55084741,-0.89930302,-0.91395829,-0.14429827])])
# print(Theta)
for n in range(5, 41, 5):
	h(Theta, x_ts[n-1], n)
