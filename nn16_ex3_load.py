import pickle as pckl  # to load dataset
import pylab as pl     # for graphics
import numpy as np

pl.close('all')   # closes all previous figures


def sig(x):
    return 1 / (1 + np.exp(-x))

# Load dataset
file_in = open('vehicle.pkl', 'rb')
vehicle_data = pckl.load(file_in)
file_in.close()
X = vehicle_data[0].astype(float)   # input vectors X[i,:] is i-th example
C = vehicle_data[1]   # classes C[i] is class of i-th example

C_indices = np.where(np.logical_or(C == 2, C == 3))[0]
C = np.array([C[i] for i in C_indices])
X = np.array([X[i] for i in C_indices])

X_means = np.mean(X, axis=0)
X_stds = np.std(X, axis=0)

X[:, :] = (X[:, :] - X_means[:]) / X_stds[:]
C[:] = C[:] - 2

# for eta in np.linspace(0.00001, 0.01, num=50):
for epoch_len in range(20, 10000, 250):
    eta = 0.004
    w = np.random.random((X.shape[1], 1))
    for epoch in range(0, epoch_len):
        y = sig(np.dot(X, w))
        a = np.dot(np.transpose(C), np.log(y.clip(min=0.0000000000001)))
        ce = -(a + np.dot(np.transpose(np.ones_like(C) - C), np.log((np.ones_like(y) - y).clip(min=0.00000000000001))))
        if epoch == epoch_len-1:
            # print "for eta=", eta, "error =", ce
            y_norm = (y >= 0.5).astype(int)
            class_rate = np.sum(C == y_norm) / float(C.shape[0])
            print "for epoch len=", epoch_len, "error =", ce, "class rate =", class_rate
        E_ce = np.dot(np.transpose(y - C), X[:])
        w = w - eta * np.transpose(E_ce)

# optimal eta was 0.004
