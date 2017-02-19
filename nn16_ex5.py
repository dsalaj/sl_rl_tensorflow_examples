# Neural Networks WS2016/17
# Task 5
# Template script

import numpy as np
import tensorflow as tf
import pylab as plt
# import matplotlib.pyplot as plt


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

USE_LSTM = True
num_hidden = 20
seq_len = 20
learning_rate = 0.0025
num_epochs = 100


class RNN():
    def __init__(self, scope, use_lstm):
        with tf.variable_scope(scope):
            self.data = tf.placeholder(tf.float32, [None, seq_len, 1])

            # Setup the recurrent layer
            if use_lstm:
                self.cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True) # a layer of num_hidden LSTM cells
            else:
                self.cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden) # a layer of num_hidden basic sigmoidal neurons

            # The following line wraps this layer in a recurrent neural network
            self.val, _ = tf.nn.dynamic_rnn(self.cell, self.data, dtype=tf.float32) # val is the output (state) of this layer
            # Note: this tensor is unrolled in time (contains hidden states of all time points)
            # Note: the recurrent weights are encapsuled, we do not need to define them

            # Obtain output of the layer
            self.val = tf.transpose(self.val, [1, 0, 2]) # transpose output tensor in order to get time as the first dimension
            self.last = tf.gather(self.val, int(self.val.get_shape()[0]) - 1) # Here, we obtain the hidden layer output at the last time step.
            # Use on top of this a sigmoid neuron for sequence classification.

            self.target = tf.placeholder(tf.float32, [None, 1])

            self.weight = tf.Variable(tf.truncated_normal([num_hidden, 1], stddev=0.1))
            self.bias = tf.Variable(tf.constant(0.1, shape=[1]))
            self.prediction = tf.nn.sigmoid(tf.matmul(self.last, self.weight) + self.bias)
            self.cross_entropy = tf.reduce_mean(-self.target * tf.log(self.prediction) - (1-self.target) * tf.log(1 - self.prediction))

            # Test trained model
            self.r_prediction = tf.round(self.prediction)
            self.correct_prediction = tf.equal(self.r_prediction, self.target)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            self.train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(self.cross_entropy)

rnn = RNN("rnn", use_lstm=False)
lstm = RNN("lstm", use_lstm=True)
sess = tf.Session()
sess.run(tf.initialize_all_variables())

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
rnn_t_mcs = []
rnn_v_mcs = []
for e_i in range(num_epochs):
    for b_i, (batch_xs, batch_ys) in enumerate(iterate_minibatches(np.asarray(train_set['data']),
                                                                   np.asarray(train_set['labels']),
                                                                   shuffle=True, batchsize=100)):
        sess.run(rnn.train_step, feed_dict={rnn.data: np.asarray(batch_xs), rnn.target: np.asarray(batch_ys)})
    train_acc = sess.run(rnn.accuracy, feed_dict={rnn.data: train_set['data'], rnn.target: train_set['labels']})
    valid_acc = sess.run(rnn.accuracy, feed_dict={rnn.data: valid_set['data'], rnn.target: valid_set['labels']})
    rnn_t_mcs.append(1. - train_acc)
    rnn_v_mcs.append(1. - valid_acc)
    # print("epoch", e_i, "training accuracy", train_acc, "validation accuracy", valid_acc)
rnn_test_err = 1. - sess.run(rnn.accuracy, feed_dict={rnn.data: test_set['data'], rnn.target: test_set['labels']})
print("Final RNN test error", rnn_test_err)

lstm_t_mcs = []
lstm_v_mcs = []
for e_i in range(num_epochs):
    for b_i, (batch_xs, batch_ys) in enumerate(iterate_minibatches(np.asarray(train_set['data']),
                                                                   np.asarray(train_set['labels']),
                                                                   shuffle=True, batchsize=100)):
        sess.run(lstm.train_step, feed_dict={lstm.data: np.asarray(batch_xs), lstm.target: np.asarray(batch_ys)})
    train_acc = sess.run(lstm.accuracy, feed_dict={lstm.data: train_set['data'], lstm.target: train_set['labels']})
    valid_acc = sess.run(lstm.accuracy, feed_dict={lstm.data: valid_set['data'], lstm.target: valid_set['labels']})
    lstm_t_mcs.append(1. - train_acc)
    lstm_v_mcs.append(1. - valid_acc)
    # print("epoch", e_i, "training accuracy", train_acc, "validation accuracy", valid_acc)
lstm_test_err = 1. - sess.run(rnn.accuracy, feed_dict={rnn.data: test_set['data'], rnn.target: test_set['labels']})
print("Final LSTM test error", lstm_test_err)

fig = plt.figure()
rnn_t_line, = plt.plot([i for i in range(num_epochs)], rnn_t_mcs, 'b-', label='rnn training error')
rnn_v_line, = plt.plot([i for i in range(num_epochs)], rnn_v_mcs, 'm-', label='rnn validation error')
lstm_t_line, = plt.plot([i for i in range(num_epochs)], lstm_t_mcs, 'y-', label='lstm training error')
lstm_v_line, = plt.plot([i for i in range(num_epochs)], lstm_v_mcs, 'r-', label='lstm validation error')
plt.legend(handles=[rnn_t_line, rnn_v_line, lstm_t_line, lstm_v_line])
plt.xlabel("Number of epochs")
plt.ylabel("Misclassification rate")

plt.tight_layout()
plt.show()

