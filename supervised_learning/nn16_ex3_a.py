import pickle as pckl  # to load dataset
import pylab as pl     # for graphics
import numpy as np
import matplotlib.pyplot as plt

pl.close('all')   # closes all previous figures

def sig(x):
    return 1 / (1 + np.exp(-x))

# Load dataset
file_in = open('vehicle.pkl', 'rb')
vehicle_data = pckl.load(file_in)
file_in.close()
X = vehicle_data[0]   # input vectors X[i,:] is i-th example
C = vehicle_data[1]   # classes C[i] is class of i-th example

# convert to float to avoid calculating problems later
X = X.astype(np.dtype('float64'))

# extract only data for SAAB (2) and BUS (3) classes
C_indices = np.where(np.logical_or(C == 2, C == 3))[0]
C = np.take(C, C_indices, axis=0)
X = np.take(X, C_indices, axis=0)

# normalize the data over features
X_means = np.mean(X, axis=0)
X_stds = np.std(X, axis=0)
X = (X - X_means) / X_stds

# add bias
bias = np.ones((X.shape[0], 1))
X = np.hstack((X, bias))

# remap the class labels to 0 and 1
C[:] = C[:] - 2

# loop we used to find optimal eta
# for eta in np.linspace(0.00001, 0.01, num=50):
# eta = 0.004

epoch_plot_points = []
ce_plot_points = []
mr_plot_points = []
min_float = 0.00000000000001


# learn over different epoch lengths (needed for plot later)
eta = 0.004
for epoch_len in [10, 20] + range(100, 1050, 50):

    # initialize weights with random values
    w = np.random.random((X.shape[1], 1))
    epoch_plot_points.append(epoch_len)

    for epoch in range(0, epoch_len):

        # prediction
        y = sig(np.dot(X, w))
        # cross entropy error for plots
        a = np.dot(np.transpose(C), np.log(y.clip(min=min_float)))
        ce = -(a + np.dot(np.transpose(np.ones_like(C) - C), np.log((np.ones_like(y) - y).clip(min=min_float))))
        if epoch == epoch_len-1:
            # print "for eta=", eta, "error =", ce
            # misclassification rate
            y_norm = (y >= 0.5).astype(int)
            class_rate = np.sum(C == y_norm) / float(C.shape[0])
            ce_plot_points.append(ce[0, 0])
            mr_plot_points.append(1 - class_rate)
            print "for epoch len=", epoch_len, "error =", ce, "class rate =", class_rate
        # gradient of cross entropy
        E_ce = np.dot(np.transpose(y - C), X[:])
        # update rule
        w = w - eta * np.transpose(E_ce)

# optimal eta was 0.004

plt.figure()
plt.subplot(211)
plt.plot(epoch_plot_points, ce_plot_points, 'b-')
plt.xlabel("Number of epochs")
plt.ylabel("Cross-entropy error")
plt.subplot(212)
plt.plot(epoch_plot_points, mr_plot_points, 'r-')
plt.xlabel("Number of epochs")
plt.ylabel("Misclassification rate")
# plt.title("solution a)")

plt.tight_layout()
plt.show()
