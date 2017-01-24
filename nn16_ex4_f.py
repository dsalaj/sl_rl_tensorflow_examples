import pickle as pckl  # to load dataset
import pylab as pl     # for graphics
import numpy as np
import tensorflow as tf

n_hidden = 300
batch_size = 50
n_epochs = 50
lr = 0.05
weight_decay = 0.0
keep_prob = 0.99


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


# Reference function from Tensorflow CIFAR10 example
def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = tf.get_variable(name, shape,
                        initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=dtype), dtype=dtype)
  weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
  tf.add_to_collection('losses', weight_decay)
  return var


# Reference function from Tensorflow CIFAR10 example
def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

# Create the model with single hidden layer
x = tf.placeholder(tf.float32, [None, n_features])
W = _variable_with_weight_decay("W", [n_features, n_hidden], stddev=0.1, wd=weight_decay)
b = tf.Variable(tf.constant(0.1, shape=[n_hidden]))
Wh = _variable_with_weight_decay("Wh", [n_hidden, n_hidden], stddev=0.1, wd=weight_decay)
bh = tf.Variable(tf.constant(0.1, shape=[n_hidden]))
Whh = _variable_with_weight_decay("Whh", [n_hidden, n_classes], stddev=0.1, wd=weight_decay)
bhh = tf.Variable(tf.constant(0.1, shape=[n_classes]))

a_inpt = tf.matmul(x, W) + b
inpt = tf.nn.dropout(tf.nn.relu(a_inpt), keep_prob)
b_out = tf.matmul(inpt, Wh) + bh
out = tf.nn.dropout(tf.nn.relu(b_out), keep_prob)
y = tf.matmul(out, Whh) + bhh
# not applying softmax as it is implicitly used in cross_entropy bellow

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, n_classes])
# y = logits
# y_ = labels
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

learning_rate = tf.placeholder(tf.float32)

# optimal learning rate for GD is 0.05
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss(logits=y, labels=y_))
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
reset_hW = Wh.assign(tf.truncated_normal([n_hidden, n_hidden], stddev=0.1))
reset_hb = bh.assign(tf.constant(0.1, shape=[n_hidden]))
reset_hhW = Whh.assign(tf.truncated_normal([n_hidden, n_classes], stddev=0.1))
reset_hhb = bhh.assign(tf.constant(0.1, shape=[n_classes]))
sess.run([reset_W, reset_b, reset_hW, reset_hb, reset_hhW, reset_hhb], feed_dict={})

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
