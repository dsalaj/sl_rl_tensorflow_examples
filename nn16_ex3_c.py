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

# for eta in np.linspace(0.00001, 0.01, num=50):
epoch_plot_points = []
ce_plot_points = []
mr_plot_points = []
min_float = 0.00000000000001
for epoch_len in range(20, 1000, 100):
    eta = 0.004
    w = np.random.random((X.shape[1], 1))
    epoch_plot_points.append(epoch_len)
    for epoch in range(0, epoch_len):
        y = sig(X.dot(w))
        R_array = y * (np.ones_like(y) - y)
        R = np.diag(R_array[:, 0])

        # XtR = X.T.dot(R)
        # # print "XtR shape =", XtR.shape
        # XtRX = XtR.dot(X)
        # # print "XtRX shape =", XtRX.shape
        # XtRX_inv = np.linalg.inv(XtRX)
        # # print "XtRX_inv shape =", XtRX_inv.shape
        # XtRX_inv_Xt = XtRX_inv.dot(X.T)
        # # print "XtRX_inv_Xt shape =", XtRX_inv_Xt.shape
        # # print "(y - C) shape =", (y - C).shape
        # IRLS_update = XtRX_inv_Xt.dot(y - C)

        IRLS_update = np.linalg.inv(X.T.dot(R).dot(X)).dot(X.T).dot(y - C)
        # print "IRLS_update shape =", IRLS_update.shape
        w = w - IRLS_update

        a = np.dot(np.transpose(C), np.log(y.clip(min=min_float)))
        ce = -(a + np.dot(np.transpose(np.ones_like(C) - C), np.log((np.ones_like(y) - y).clip(min=min_float))))
        E_ce = np.dot(np.transpose(y - C), X[:])
        # w = w - eta * np.transpose(E_ce)

        if epoch == epoch_len-1:
            # print "for eta=", eta, "error =", ce
            y_norm = (y >= 0.5).astype(int)
            class_rate = np.sum(C == y_norm) / float(C.shape[0])
            ce_plot_points.append(ce[0, 0])
            misclass_rate = 1 - class_rate
            mr_plot_points.append(misclass_rate)
            print "for epoch len=", epoch_len, "error =", ce, "misclass rate =", misclass_rate

# optimal eta was 0.004

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
