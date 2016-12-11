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

a_epoch_plot_points = []
b_epoch_plot_points = []
c_epoch_plot_points = []

a_ce_plot_points = []
a_mr_plot_points = []
b_ce_plot_points = []
b_mr_plot_points = []
c_ce_plot_points = []
c_mr_plot_points = []



min_float = 0.00000000000001
# initialize weights with random values
eta = 0.004
for epoch_len in range(1,100,10) + range(100, 1050, 50):
    w = np.random.random((X.shape[1], 1))
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
            a_epoch_plot_points.append(epoch_len)
            a_ce_plot_points.append(ce[0, 0])
            a_mr_plot_points.append(1 - class_rate)
            print "for epoch len=", epoch_len, "error =", ce, "class rate =", class_rate
        # gradient of cross entropy
        E_ce = np.dot(np.transpose(y - C), X[:])
        # update rule
        w = w - eta * np.transpose(E_ce)



eta = 0.008
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
    b_epoch_plot_points.append(i)
    b_ce_plot_points.append(test_ce[0, 0])
    b_mr_plot_points.append(1 - class_rate)
    if i % 50 == 0:
        print "epoch =", i, "error =", test_ce, "class rate =", class_rate


done = False
for epoch_len in range(1,100,10) + range(100, 1100, 100):
    w = np.random.uniform(-5e-5,5e-5,(X.shape[1], 1))
    for epoch in range(0, epoch_len+1):
        y = sig(X.dot(w))
        R_array = y * (np.ones_like(y) - y)
        R = np.diag(R_array[:, 0])

        try:
            # FIXME:
            # if the values are too close to zero, the matrix is singular and the inverse can not be computed
            # this will throw an exception and we can finish? stopping criterion?
            inverse = np.linalg.inv(X.T.dot(R).dot(X))
            grad = X.T.dot(y - C) / X.shape[0]
            IRLS_update = inverse.dot(grad)
            w = w - IRLS_update
        except np.linalg.linalg.LinAlgError as e:
            assert "Singular matrix" in e
            done = True

        a = np.dot(np.transpose(C), np.log(y.clip(min=min_float)))
        ce = -(a + np.dot(np.transpose(np.ones_like(C) - C), np.log((np.ones_like(y) - y).clip(min=min_float))))
        E_ce = np.dot(np.transpose(y - C), X[:])

        if done or epoch == epoch_len:
            y_norm = (y >= 0.5).astype(int)
            class_rate = np.sum(C == y_norm) / float(C.shape[0])
            c_epoch_plot_points.append(epoch_len)
            c_ce_plot_points.append(ce[0, 0])
            misclass_rate = 1 - class_rate
            c_mr_plot_points.append(misclass_rate)
            print "for epoch len=", epoch, "error =", ce, "misclass rate =", misclass_rate
            break
            # With this method we are achieving misclassification error 0.03 for 900 epochs

print "length ", len(a_ce_plot_points)
plt.figure()
plt.subplot(211)
plt.plot(a_epoch_plot_points, a_ce_plot_points, 'b-', label="batch GD")
plt.plot(b_epoch_plot_points, b_ce_plot_points, 'r-', label="stochastic GD")
plt.plot(c_epoch_plot_points, c_ce_plot_points, 'g-', label="IRLS")
plt.xlabel("Number of epochs")
plt.ylabel("Cross-entropy error")
plt.legend()
plt.subplot(212)
plt.plot(a_epoch_plot_points, a_mr_plot_points, 'b-', label="batch GD")
plt.plot(b_epoch_plot_points, b_mr_plot_points, 'r-', label="stochastic GD")
plt.plot(c_epoch_plot_points, c_mr_plot_points, 'g-', label="IRLS")
plt.xlabel("Number of epochs")
plt.ylabel("Misclassification rate")
plt.legend()
# plt.title("solution a)")

plt.tight_layout()
plt.show()
