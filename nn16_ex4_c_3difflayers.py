import pickle as pckl  # to load dataset
import pylab as pl     # for graphics
import numpy as np
import tensorflow as tf

n_hidden = 20
n_hidden2 = 60
n_hidden3 = 10
batch_size = 50
n_epochs = 100
lr = 0.05


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

# Create the model with single hidden layer
x = tf.placeholder(tf.float32, [None, n_features])
W = tf.Variable(tf.truncated_normal([n_features, n_hidden], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[n_hidden]))
Wh = tf.Variable(tf.truncated_normal([n_hidden, n_hidden2], stddev=0.1))
bh = tf.Variable(tf.constant(0.1, shape=[n_hidden2]))
Whh = tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1))
bhh = tf.Variable(tf.constant(0.1, shape=[n_hidden3]))
Whhh = tf.Variable(tf.truncated_normal([n_hidden3, n_classes], stddev=0.1))
bhhh = tf.Variable(tf.constant(0.1, shape=[n_classes]))

a_inpt = tf.matmul(x, W) + b
inpt = tf.nn.relu(a_inpt)
b_inpt = tf.matmul(inpt, Wh) + bh
b = tf.nn.relu(b_inpt)
c_input = tf.matmul(b, Whh) + bhh
c = tf.nn.relu(c_input)
y = tf.matmul(c, Whhh) + bhhh
# not applying softmax as it is implicitly used in cross_entropy bellow

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, n_classes])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
learning_rate = tf.placeholder(tf.float32)

# optimal learning rate for GD is 0.1
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
# optimal learning rate for Adam is 0.0005
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
# optimal learning rate for RMSProp is 0.001
# train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train
shuffle_in_unison_scary(X, C_onehot)

n_train_data = int(n_data * 0.8)
n_es_data = int(n_data * 0.9)
X_train = X[:n_train_data]
C_train_onehot = C_onehot[:n_train_data]
X_es = X[n_train_data:n_es_data]
C_es_onehot = C_onehot[n_train_data:n_es_data]
X_valid = X[n_es_data:]
C_valid_onehot = C_onehot[n_es_data:]

max_valid_acc = opt_epoch_num = opt_batch_part = train_acc = test_acc = 0

best_es_validation = 0
arch_accs = []
train_accs = []
epoch_steps = 0
for e_i in range(n_epochs):
  for b_i, (batch_xs, batch_ys) in enumerate(iterate_minibatches(X_train, C_train_onehot,
                                                                 shuffle=False, batchsize=batch_size)):
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,
                                    learning_rate: lr})
  es_acc = sess.run(accuracy, feed_dict={x: X_es, y_: C_es_onehot})
  train_accs.append(1 - sess.run(accuracy, feed_dict={x: X_train, y_: C_train_onehot}))
  arch_accs.append(1 - sess.run(accuracy, feed_dict={x: X_valid, y_: C_valid_onehot}))
  print("training acc = %.6f" % train_accs[-1], "arch valid acc = %.6f" % arch_accs[-1])
  epoch_steps += 1
  if es_acc >= best_es_validation:
      best_es_validation = es_acc
  else:
      break

import matplotlib.pyplot as plt
fig = plt.figure()
arch, = plt.plot([i for i in range(epoch_steps)], arch_accs, 'b-', label='Arch Validation Set')
tr, = plt.plot([i for i in range(epoch_steps)], train_accs, 'r-', label='Training Set')
plt.legend(handles=[arch, tr])
plt.xlabel("Number of epochs")
plt.ylabel("Misclassification rate")

plt.tight_layout()
plt.show()
