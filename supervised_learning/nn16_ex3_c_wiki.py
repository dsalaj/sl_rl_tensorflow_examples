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
X = vehicle_data[0].astype(float)   # input vectors X[i,:] is i-th example
C = vehicle_data[1]   # classes C[i] is class of i-th example

C_indices = np.where(np.logical_or(C == 2, C == 3))[0]
C = np.array([C[i] for i in C_indices])
X = np.array([X[i] for i in C_indices])

X_means = np.mean(X, axis=0)
X_stds = np.std(X, axis=0)

X[:, :] = (X[:, :] - X_means[:]) / X_stds[:]
C[:] = C[:] - 2

epoch_plot_points = []
ce_plot_points = []
mr_plot_points = []
min_float = 1e-12
for epoch_len in range(20, 1000, 100):
    w = np.ones((X.shape[0], 1))
    W = np.diag(w[:, 0])
    B = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T.dot(W).dot(C))
    done = False
    for epoch in range(0, epoch_len):
        _B = B
        _w = abs(C - X.dot(B)).T
        w = 1./np.maximum(min_float, _w)
        W = np.diag(w[0])
        B = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T.dot(W).dot(C))
        tol = np.sum(abs(B - _B))
        # print "Tolerance =", tol
        if tol < 1e-9:
            break

    y = X.dot(B)
    y_norm = (y >= 0.5).astype(int)
    class_rate = np.sum(C == y_norm) / float(C.shape[0])
    # ce_plot_points.append(ce[0, 0])
    misclass_rate = 1 - class_rate
    mr_plot_points.append(misclass_rate)
    # print "for epoch len=", epoch, "error =", ce, "misclass rate =", misclass_rate
    print "for epoch len=", epoch, "misclass rate =", misclass_rate
    # With this method we are achieving misclassification error 0.381

    # y = sig(X.dot(w))
    # R_array = y * (np.ones_like(y) - y)
    # R = np.diag(R_array[:, 0])

    # a = np.dot(np.transpose(C), np.log(y.clip(min=min_float)))
    # ce = -(a + np.dot(np.transpose(np.ones_like(C) - C), np.log((np.ones_like(y) - y).clip(min=min_float))))
    # E_ce = np.dot(np.transpose(y - C), X[:])

    # if done:
    #     y_norm = (y >= 0.5).astype(int)
    #     class_rate = np.sum(C == y_norm) / float(C.shape[0])
    #     ce_plot_points.append(ce[0, 0])
    #     misclass_rate = 1 - class_rate
    #     mr_plot_points.append(misclass_rate)
    #     print "for epoch len=", epoch, "error =", ce, "misclass rate =", misclass_rate
    #     break


# plt.figure()
# plt.subplot(211)
# plt.plot(epoch_plot_points, ce_plot_points, 'b-')
# plt.xlabel("Number of epochs")
# plt.ylabel("Cross-entropy error")
# plt.subplot(212)
# plt.plot(epoch_plot_points, mr_plot_points, 'r-')
# plt.xlabel("Number of epochs")
# plt.ylabel("Misclassification rate")
# # plt.title("solution a)")
#
# plt.tight_layout()
# plt.show()
