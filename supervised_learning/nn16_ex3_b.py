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
eta_plot_points = []
min_float = 0.00000000000001
# for eta in [10**i for i in range(-10, 10)]:
eta = 0.008
# eta_plot_points.append(eta)


# Training on dataset and ploting for only 1 epoch
#
# w = np.random.random((X.shape[1], 1))
# for [x, c] in zip(X, C):
#     y = sig(np.dot(x, w))
#     # a = np.dot(c, np.log(y.clip(min=min_float)))
#     # ce = -(a + np.dot((np.ones_like(c) - c), np.log((np.ones_like(y) - y).clip(min=min_float))))
#     # if np.all(x == X[-1]):
#     #     print "for eta=", eta, "error =", ce
#
#     # E_ce = np.dot(np.transpose(y - c), x)
#     x_ = np.reshape(x, (x.shape[0], 1))
#     E_ce = x_.dot(y-c)
#     E_ce = np.reshape(E_ce, (E_ce.shape[0], 1))
#     w = w - eta * E_ce
#
#     test_y = sig(np.dot(X, w))
#     test_a = np.dot(np.transpose(C), np.log(test_y.clip(min=min_float)))
#     test_ce = -(test_a + np.dot(np.transpose(np.ones_like(C) - C), np.log((np.ones_like(test_y) - test_y).clip(min=min_float))))
#
#     test_y_norm = (test_y >= 0.5).astype(int)
#     class_rate = np.sum(C == test_y_norm) / float(C.shape[0])
#     ce_plot_points.append(test_ce[0, 0])
#     mr_plot_points.append(1 - class_rate)
#
## note 1120 was the error of the batched version after one epoch, 0.6 the misclassification rate
# plt.figure()
# plt.subplot(211)
# plt.plot(range(1, X.shape[0]+1), ce_plot_points, 'b-')
# plt.plot((1,X.shape[0]+1), (1120,1120), 'g--')
# plt.xlabel("Training samples processed")
# plt.ylabel("Cross-entropy error")
# plt.subplot(212)
# plt.plot(range(1, X.shape[0]+1), mr_plot_points, 'r-')
# plt.plot((1,X.shape[0]+1), (0.6, 0.6), 'g--')
# plt.xlabel("Training samples processed")
# plt.ylabel("Misclassification rate")
#
# plt.tight_layout()
# plt.show()
#
# exit()

w = np.random.random((X.shape[1], 1))
num_epochs = 1000
for i in range(1, 1+num_epochs):
    for [x, c] in zip(X, C):
        y = sig(np.dot(x, w))
        # a = np.dot(c, np.log(y.clip(min=min_float)))
        # ce = -(a + np.dot((np.ones_like(c) - c), np.log((np.ones_like(y) - y).clip(min=min_float))))
        # if np.all(x == X[-1]):
        #     print "for eta=", eta, "error =", ce

        # E_ce = np.dot(np.transpose(y - c), x)
        x_ = np.reshape(x, (x.shape[0], 1))
        E_ce = x_.dot(y-c)
        E_ce = np.reshape(E_ce, (E_ce.shape[0], 1))
        w = w - eta * E_ce

    test_y = sig(np.dot(X, w))
    test_a = np.dot(np.transpose(C), np.log(test_y.clip(min=min_float)))
    test_ce = -(test_a + np.dot(np.transpose(np.ones_like(C) - C), np.log((np.ones_like(test_y) - test_y).clip(min=min_float))))

    test_y_norm = (test_y >= 0.5).astype(int)
    class_rate = np.sum(C == test_y_norm) / float(C.shape[0])
    ce_plot_points.append(test_ce[0, 0])
    mr_plot_points.append(1 - class_rate)
    if i % 50 == 0:
        print "epoch =", i, "error =", test_ce, "class rate =", class_rate

# plt.figure()
# fig, ax = plt.subplots()
# plt.plot(eta_plot_points, ce_plot_points, 'b-')
# ax.set_xticks(eta_plot_points)
# plt.xlabel("Eta value")
# plt.ylabel("Cross-entropy error")
# plt.xscale('log')
# plt.tight_layout()
# plt.show()
# plt.subplots()
# optimal eta was 10^-6

plt.figure()
plt.subplot(211)
plt.plot(range(1, num_epochs+1), ce_plot_points, 'b-')
plt.xlabel("Number of epochs")
plt.ylabel("Cross-entropy error")
plt.subplot(212)
plt.plot(range(1, num_epochs+1), mr_plot_points, 'r-')
plt.xlabel("Number of epochs")
plt.ylabel("Misclassification rate")
# plt.title("solution a)")

plt.tight_layout()
plt.show()
