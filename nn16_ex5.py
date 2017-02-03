# Neural Networks WS2016/17
# Task 5
# Template script

import numpy as np
import tensorflow as tf
import pylab as pl


def iterate_minibatches(samples, labels, shuffle=False, batchsize=32):
    indices = np.arange(samples.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, samples.shape[0] - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        samples_yield = [samples[i] for i in excerpt]
        labels_yield = [labels[i] for i in excerpt]
        yield np.asarray(samples_yield), np.asarray(labels_yield)


def generate_sequences(N, T):
    # generates sequences of random {0,1}^T strings
    # N...number of samples
    # T....sequence length to generate
    #
    # Returns SEQ....list of N {0,1}^T bit strings  
    SEQ = []
    for n in range(N):
      SEQ.append(np.random.randint(2, size=(T,1)))
    return SEQ


def generate_targets_xor(SEQ,offset=5):
    # generate targets for sequences SEQ
    # targets[i]= xor(SEQ[offset], SEQ[offset+1])
    # (Use default offset 5 for this example) 
    targets = []
    for s in SEQ:
      val = s[offset] ^ s[offset+1]
      targets.append( [val[0]] )
    return targets

###################################################
## How to define a recurrent hidden layer:
###################################################

USE_LSTM = False
num_hidden = 50
seq_len = 20
learning_rate = 0.0025
num_epochs = 50

data = tf.placeholder(tf.float32, [None, seq_len, 1])

# Setup the recurrent layer
if USE_LSTM:
  cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True) # a layer of num_hidden LSTM cells
else:
  cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden) # a layer of num_hidden basic sigmoidal neurons

# The following line wraps this layer in a recurrent neural network
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32) # val is the output (state) of this layer
# Note: this tensor is unrolled in time (contains hidden states of all time points)
# Note: the recurrent weights are encapsuled, we do not need to define them

# Obtain output of the layer
val = tf.transpose(val, [1, 0, 2]) # transpose output tensor in order to get time as the first dimension
last = tf.gather(val, int(val.get_shape()[0]) - 1) # Here, we obtain the hidden layer output at the last time step.
# Use on top of this a sigmoid neuron for sequence classification.

target = tf.placeholder(tf.float32, [None, 1])

weight = tf.Variable(tf.truncated_normal([num_hidden, 1], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[1]))
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(target * tf.log(prediction)))

# out = tf.argmax(tf.nn.sigmoid(last), 1)
# labels = tf.placeholder(tf.int32, [None, 1])
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=out))

# Test trained model
correct_prediction = tf.equal(tf.round(prediction), target)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

# Generate data
train_set = {'data': [], 'labels': []}
valid_set = {'data': [], 'labels': []}
test_set = {'data': [], 'labels': []}
sets = [train_set, valid_set, test_set]
for i in range(3):
    raw_data = generate_sequences(2000, seq_len)
    sets[i]['data'] = raw_data
    sets[i]['labels'] = generate_targets_xor(raw_data)

# Start learning
for _ in range(num_epochs):
    # for b_i, (batch_xs, batch_ys) in enumerate(iterate_minibatches(np.asarray(train_set['data']), np.asarray(train_set['labels']),
    #                                                                shuffle=True, batchsize=32)):
    #     sess.run(train_step, feed_dict={data: np.asarray(batch_xs), target: np.asarray(batch_ys)})
    sess.run(train_step, feed_dict={data: train_set['data'], target: train_set['labels']})
    weights = sess.run(weight, feed_dict={data: train_set['data'], target: train_set['labels']})
    print(weights)
    train_acc = sess.run(accuracy, feed_dict={data: train_set['data'], target: train_set['labels']})
    valid_acc = sess.run(accuracy, feed_dict={data: valid_set['data'], target: valid_set['labels']})
    print("training error", train_acc, "validation error", valid_acc)

