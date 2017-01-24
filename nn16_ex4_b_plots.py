import pickle as pckl  # to load dataset
import pylab as pl     # for graphics
import numpy as np
import tensorflow as tf

n_hidden = 20
batch_size = 40
n_epochs = 40


def shuffle_in_unison_scary(a, b):
  rng_state = np.random.get_state()
  np.random.shuffle(a)
  np.random.set_state(rng_state)
  np.random.shuffle(b)


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

# In this step we are not allowed to use test data set
# file_in = open('isolet_crop_test.pkl','rb')
# isolet_test = pckl.load(file_in) # Python 3
# X_tst = isolet_test[0]   # input vectors X[i,:] is i-th example
# C_tst = isolet_test[1]   # classes C[i] is class of i-th example
file_in.close()

# Normalize the data
X_means = np.mean(X, axis=0)
X_stds = np.std(X, axis=0)
X[:, :] = (X[:, :] - X_means[:]) / X_stds[:]
# X_tst_means = np.mean(X_tst, axis=0)
# X_tst_stds = np.std(X_tst, axis=0)
# X_tst[:, :] = (X_tst[:, :] - X_tst_means[:]) / X_tst_stds[:]

n_data = C.shape[0]  # = 6238
# n_tst_data = C_tst.shape[0]
n_features = X.shape[1]  # = 300
n_classes = np.max(C)  # = 26

C_onehot = np.zeros((n_data, n_classes))
C_onehot[np.arange(n_data), C - np.ones_like(C)] = 1
# C_tst_onehot = np.zeros((n_tst_data, n_classes))
# C_tst_onehot[np.arange(n_tst_data), C_tst - np.ones_like(C_tst)] = 1

# Create the single layer model
x = tf.placeholder(tf.float32, [None, n_features])
W = tf.Variable(tf.truncated_normal([n_features, n_classes], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[n_classes]))
y = tf.matmul(x, W) + b
# # not applying softmax as it is implicitly used in cross_entropy bellow

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, n_classes])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
learning_rate = tf.placeholder(tf.float32)

# optimal learning rate for GD is 0.1
train_gd = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
# optimal learning rate for Adam is 0.0005
train_adam = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
# optimal learning rate for RMSProp is 0.001
train_rmsprop = tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train
shuffle_in_unison_scary(X, C_onehot)

n_train_data = int(n_data * 0.8)
X_train = X[:n_train_data]
C_train_onehot = C_onehot[:n_train_data]

X_valid = X[n_train_data:]
C_valid_onehot = C_onehot[n_train_data:]
'''
for lr in [0.0001, 0.0005, 0.0008] + list(np.arange(0.001, 0.01, 0.002)) + \
          [0.01, 0.02] + list(np.arange(0.05, 0.5, 0.05)):
  a = W.assign(tf.truncated_normal([n_features, n_classes], stddev=0.1))
  sess.run(a, feed_dict={})
  a = b.assign(tf.constant(0.1, shape=[n_classes]))
  sess.run(a, feed_dict={})

  max_valid_acc = opt_epoch_num = opt_batch_part = train_acc = 0
  train_accs = []
  valid_accs = []

  for e_i in range(n_epochs):
    for b_i, (batch_xs, batch_ys) in enumerate(iterate_minibatches(X_train, C_train_onehot,
                                                                   shuffle=True, batchsize=batch_size)):
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,
                                      learning_rate: lr})
      valid_acc = sess.run(accuracy, feed_dict={x: X_valid, y_: C_valid_onehot})
      if valid_acc > max_valid_acc:
        max_valid_acc = valid_acc
        opt_epoch_num = e_i
        opt_batch_part = b_i
    train_acc = sess.run(accuracy, feed_dict={x: X_train, y_: C_train_onehot})
    train_accs.append(train_acc)
    valid_acc = sess.run(accuracy, feed_dict={x: X_valid, y_: C_valid_onehot})
    valid_accs.append(valid_acc)

  print("Learning rate = %f" % lr,
        "Epoch number with best validation accuracy is %02d" % opt_epoch_num,
        "training acc = %.6f" % train_accs[opt_epoch_num],
        "validation acc = %.6f" % valid_accs[opt_epoch_num],
       )
'''

epochs = 30
gd_misclass_data = []
adam_misclass_data = []
rmsprop_misclass_data = []
descent_steps = 0
gd_lr = 0.05
adam_lr = 0.0005
rmsprop_lr = 0.001
reset_W = W.assign(tf.truncated_normal([n_features, n_classes], stddev=0.1))
reset_b = b.assign(tf.constant(0.1, shape=[n_classes]))

sess.run([reset_W, reset_b], feed_dict={})
for _ in range(epochs):
  for _, (batch_xs, batch_ys) in enumerate(iterate_minibatches(X_train, C_train_onehot,
                                                                 shuffle=True, batchsize=batch_size)):
    sess.run(train_gd, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: gd_lr})
    valid_acc = sess.run(accuracy, feed_dict={x: X_valid, y_: C_valid_onehot})
    gd_misclass_data.append(1-valid_acc)
    descent_steps += 1

sess.run([reset_W, reset_b], feed_dict={})
for _ in range(epochs):
  for _, (batch_xs, batch_ys) in enumerate(iterate_minibatches(X_train, C_train_onehot,
                                                                 shuffle=True, batchsize=batch_size)):
    sess.run(train_adam, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: adam_lr})
    valid_acc = sess.run(accuracy, feed_dict={x: X_valid, y_: C_valid_onehot})
    adam_misclass_data.append(1-valid_acc)

sess.run([reset_W, reset_b], feed_dict={})
for _ in range(epochs):
  for _, (batch_xs, batch_ys) in enumerate(iterate_minibatches(X_train, C_train_onehot,
                                                                 shuffle=True, batchsize=batch_size)):
    sess.run(train_rmsprop, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: rmsprop_lr})
    valid_acc = sess.run(accuracy, feed_dict={x: X_valid, y_: C_valid_onehot})
    rmsprop_misclass_data.append(1-valid_acc)

import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
gd, = ax1.plot([i for i in range(descent_steps)], gd_misclass_data,      'b-', label='GD')
adam, = ax1.plot([i for i in range(descent_steps)], adam_misclass_data,    'r-', label='ADAM')
rmsprop, = ax1.plot([i for i in range(descent_steps)], rmsprop_misclass_data, 'g-', label='RMSProp')
plt.legend(handles=[gd, adam, rmsprop])
ax1.set_xlabel("Number of descent steps")
ax1.set_ylabel("Misclassification rate")
descent_steps_per_epoch = int(descent_steps / epochs)


def tick_function(X):
    V = X / descent_steps_per_epoch
    return ["%d" % z for z in V]

epoch_ticks = np.array([i for i in range(0, descent_steps, descent_steps_per_epoch)])
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(epoch_ticks)
ax2.set_xticklabels(tick_function(epoch_ticks))
ax2.set_xlabel(r"Number of epochs")

plt.tight_layout()
plt.show()
