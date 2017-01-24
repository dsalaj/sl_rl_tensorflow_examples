import pickle as pckl  # to load dataset
import pylab as pl     # for graphics
import numpy as np
import tensorflow as tf

n_hidden = 60
batch_size = 50
n_epochs = 50
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
Wh = tf.Variable(tf.truncated_normal([n_hidden, n_classes], stddev=0.1))
bh = tf.Variable(tf.constant(0.1, shape=[n_classes]))

a_inpt = tf.matmul(x, W) + b
inpt = tf.nn.relu(a_inpt)
y = tf.matmul(inpt, Wh) + bh
# not applying softmax as it is implicitly used in cross_entropy bellow

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, n_classes])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
learning_rate = tf.placeholder(tf.float32)

# optimal learning rate for GD is 0.05
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
X_train = X[:n_train_data]
C_train_onehot = C_onehot[:n_train_data]
X_es = X[n_train_data:]
C_es_onehot = C_onehot[n_train_data:]

max_valid_acc = opt_epoch_num = opt_batch_part = train_acc = test_acc = 0

best_es_validation = 0
train_accs = []
es_accs = []
epoch_steps = 0
best_epoch_num = 0
for e_i in range(n_epochs):
  for b_i, (batch_xs, batch_ys) in enumerate(iterate_minibatches(X_train, C_train_onehot,
                                                                 shuffle=False, batchsize=batch_size)):
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,
                                    learning_rate: lr})
  es_acc = sess.run(accuracy, feed_dict={x: X_es, y_: C_es_onehot})
  train_accs.append(1 - sess.run(accuracy, feed_dict={x: X_train, y_: C_train_onehot}))
  # es_accs.append(1 - sess.run(accuracy, feed_dict={x: X_es, y_: C_es_onehot}))
  epoch_steps += 1
  # print("epoch", epoch_steps, "training acc = %.6f" % train_accs[-1], "es valid acc = %.6f" % es_acc)
  if es_acc >= best_es_validation:
    best_es_validation = es_acc
    best_epoch_num = epoch_steps
  else:
    break

print("Best epoch number found =", best_epoch_num)

reset_W = W.assign(tf.truncated_normal([n_features, n_hidden], stddev=0.1))
reset_b = b.assign(tf.constant(0.1, shape=[n_hidden]))
reset_hW = Wh.assign(tf.truncated_normal([n_hidden, n_classes], stddev=0.1))
reset_hb = bh.assign(tf.constant(0.1, shape=[n_classes]))
sess.run([reset_W, reset_b, reset_hW, reset_hb], feed_dict={})

final_train_mcs = []
test_mcs = []
epoch_steps = 0
for e_i in range(best_epoch_num):
  for b_i, (batch_xs, batch_ys) in enumerate(iterate_minibatches(X, C_onehot, shuffle=False, batchsize=batch_size)):
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: lr})
  final_train_mcs.append(1 - sess.run(accuracy, feed_dict={x: X, y_: C_onehot}))
  test_mcs.append(1 - sess.run(accuracy, feed_dict={x: X_tst, y_: C_tst_onehot}))
  print("epoch", e_i, "training mce = %.6f" % final_train_mcs[-1], "testing mce = %.6f" % test_mcs[-1])
  epoch_steps += 1

import matplotlib.pyplot as plt
fig = plt.figure()
arch, = plt.plot([i for i in range(epoch_steps)], test_mcs, 'b-', label='Testing Set')
tr, = plt.plot([i for i in range(epoch_steps)], final_train_mcs, 'r-', label='Training Set')
plt.legend(handles=[arch, tr])
plt.xlabel("Number of epochs")
plt.ylabel("Misclassification rate")

plt.tight_layout()
plt.show()
