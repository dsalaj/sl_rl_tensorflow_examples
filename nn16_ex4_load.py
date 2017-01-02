import pickle as pckl  # to load dataset
import pylab as pl     # for graphics
import numpy as np
import tensorflow as tf

learning_rate = 0.1
n_hidden = 20
batch_size = 40
n_epochs = 80

# NOTE:
# mini-batch size 40
# single hidden layer of 20 neurons
# softmax output layer
# optimize (minimize) cross entropy error
# report misclassification rate
# compare stochastic GD with RMSprop and ADAM
# implement early stopping
# chose learning rate for each algorithm and compare with plots

def iterate_minibatches(samples, labels, shuffle=False, batchsize=32):
    indices = np.arange(samples.shape[0])
    if shuffle:
    	np.random.shuffle(indices)
    for start_idx in range(0, samples.shape[0] - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        samples_yield = [samples[i] for i in excerpt]
        labels_yield = [labels[i] for i in excerpt]
        yield samples_yield, labels_yield

pl.close('all')   # closes all previous figures

# Load dataset
file_in = open('isolet_crop_train.pkl','rb')
isolet_data = pckl.load(file_in) # Python 3
#isolet_data = pckl.load(file_in, encoding='bytes') # Python 3
file_in.close()
X = isolet_data[0]   # input vectors X[i,:] is i-th example
C = isolet_data[1]   # classes C[i] is class of i-th example

file_in = open('isolet_crop_test.pkl','rb')
isolet_test = pckl.load(file_in) # Python 3
file_in.close()
X_tst = isolet_test[0]   # input vectors X[i,:] is i-th example
C_tst = isolet_test[1]   # classes C[i] is class of i-th example

# Normalize the data
X_means = np.mean(X, axis=0)
X_stds = np.std(X, axis=0)
X[:, :] = (X[:, :] - X_means[:]) / X_stds[:]
X_tst_means = np.mean(X_tst, axis=0)
X_tst_stds = np.std(X_tst, axis=0)
X_tst[:, :] = (X_tst[:, :] - X_tst_means[:]) / X_tst_stds[:]

n_data = C.shape[0]
n_tst_data = C_tst.shape[0]
n_features = X.shape[1]  # = 300
n_classes = np.max(C)  # = 26

C_onehot = np.zeros((n_data, n_classes))
C_onehot[np.arange(n_data), C - np.ones_like(C)] = 1
C_tst_onehot = np.zeros((n_tst_data, n_classes))
C_tst_onehot[np.arange(n_tst_data), C_tst - np.ones_like(C_tst)] = 1

# # Create the model
# x = tf.placeholder(tf.float32, [None, n_features])
# W = tf.Variable(tf.zeros([n_features, n_classes]))
# b = tf.Variable(tf.zeros([n_classes]))
# # W = tf.Variable(tf.truncated_normal([n_features, n_classes], stddev=0.1))
# # b = tf.Variable(tf.constant(0.1, shape=[n_classes]))
# y = tf.matmul(x, W) + b

# Create the model with hidden layer
x = tf.placeholder(tf.float32, [None, n_features])
W = tf.Variable(tf.truncated_normal([n_features, n_hidden], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[n_hidden]))
Wh = tf.Variable(tf.truncated_normal([n_hidden, n_classes], stddev=0.1))
bh = tf.Variable(tf.constant(0.1, shape=[n_classes]))
a_inpt = tf.matmul(x, W) + b
inpt = tf.nn.relu(a_inpt)
y = tf.matmul(inpt, Wh) + bh
# not applying softmax as it is implicitly used in cross_entropy bellow

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, n_classes])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
# train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train
n_train_data = int(n_data * 0.8)
X_train = X[:n_train_data]
C_train_onehot = C_onehot[:n_train_data]
X_valid = X[n_train_data:]
C_valid_onehot = C_onehot[n_train_data:]

max_valid_acc = 0
opt_epoch_num = 0
opt_batch_part = 0
for e_i in range(n_epochs):
  for b_i, (batch_xs, batch_ys) in enumerate(iterate_minibatches(X_train, C_train_onehot, shuffle=False, batchsize=batch_size)):
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    valid_acc = sess.run(accuracy, feed_dict={x: X_valid, y_: C_valid_onehot})
    if valid_acc > max_valid_acc:
      max_valid_acc = valid_acc
      opt_epoch_num = e_i
      opt_batch_part = b_i
  train_acc = sess.run(accuracy, feed_dict={x: X_train, y_: C_train_onehot})
  test_acc = sess.run(accuracy, feed_dict={x: X_tst, y_: C_tst_onehot})
  print("test accuracy after", e_i, "epochs =", test_acc,
        # int(e_i*n_data/batch_size), "batches of size", batch_size, "=",
        "train accuracy", train_acc,
        "validation accuracy", valid_acc)

print("Epoch number with best validation accuracy is", opt_epoch_num,
      ", more precisely", (opt_epoch_num*batch_size)+opt_batch_part,
      "mini matches of size", batch_size)
