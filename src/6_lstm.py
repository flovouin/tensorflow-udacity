#! /usr/bin/env python3

import collections
import numpy as np
import random
import string
import tensorflow as tf
from matplotlib import pylab
from sklearn.manifold import TSNE

from dataset import text8, utils
import training.batch
import training.utils

data_folder = '../data'
vocabulary_size = 50000
alphabet_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
valid_size = 1000

batch_size = 64
num_unrollings = 10
learning_rate = 10.0
gradient_max_value = 1.25
num_steps = 7001
summary_frequency = 100

num_nodes = 64

# embedding_size = 128 # Dimension of the embedding vector.
# skip_window = 1 # How many words to consider left and right.
# num_skips = 2 # How many times to reuse an input to generate a label.
# # We pick a random validation set to sample nearest neighbors. here we limit the
# # validation samples to the words that have a low numeric ID, which by
# # construction are also the most frequent.
# valid_size = 16 # Random set of words to evaluate similarity on.
# valid_window = 100 # Only pick dev samples in the head of the distribution.
# valid_examples = np.array(random.sample(range(valid_window), valid_size))
# num_sampled = 64 # Number of negative examples to sample.
# num_points = 400

# Preparing or loading the dataset.
text, _, _, _, _ = text8.prepare_dataset(vocabulary_size, data_folder)

valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)

train_batches = training.batch.BatchGenerator(train_text, batch_size, num_unrollings, alphabet_size)
valid_batches = training.batch.BatchGenerator(valid_text, 1, 1, alphabet_size)

# Model on single characters.
graph = tf.Graph()
with graph.as_default():
    # [ix, fx, cx, ox]
    x_mat = tf.Variable(tf.truncated_normal([alphabet_size, 4*num_nodes], -0.1, 0.1))
    # [im, fm, cm, om]
    o_mat = tf.Variable(tf.truncated_normal([num_nodes, 4*num_nodes], -0.1, 0.1))
    # [ib, fb, cb, ob]
    b_vec = tf.Variable(tf.zeros([1, 4*num_nodes]))

    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    # Classifier weights and biases.
    w = training.utils.gaussian_weights_variable([num_nodes, alphabet_size])
    b = tf.Variable(tf.zeros([alphabet_size]))

    # Definition of the cell computation.
    def lstm_cell(i, o, state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        mult = tf.matmul(i, x_mat) + tf.matmul(o, o_mat) + b_vec

        input_gate = tf.sigmoid(mult[:, 0:num_nodes])
        forget_gate = tf.sigmoid(mult[:, num_nodes:2*num_nodes])
        state = forget_gate * state + input_gate * tf.tanh(mult[:, 2*num_nodes:3*num_nodes])
        output_gate = tf.sigmoid(mult[:, 3*num_nodes:4*num_nodes])
        return output_gate * tf.tanh(state), state

    # Input data.
    train_data = list()
    for _ in range(num_unrollings + 1):
        train_data.append(
            tf.placeholder(tf.float32, shape=[batch_size, alphabet_size]))
    train_inputs = train_data[:num_unrollings]
    train_labels = train_data[1:]  # labels are inputs shifted by one time step.

    # Unrolled LSTM loop.
    outputs = list()
    output = saved_output
    state = saved_state
    for i in train_inputs:
        output, state = lstm_cell(i, output, state)
        outputs.append(output)

    # State saving across unrollings.
    with tf.control_dependencies([saved_output.assign(output),
                                  saved_state.assign(state)]):
        # Classifier.
        logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits, tf.concat(0, train_labels)))

    # Optimizer.
    global_step = tf.Variable(0)
    tf_lr = tf.train.exponential_decay(
        learning_rate, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(tf_lr)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, gradient_max_value)
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

    # Predictions.
    train_prediction = tf.nn.softmax(logits)

    # Sampling and validation eval: batch 1, no unrolling.
    sample_input = tf.placeholder(tf.float32, shape=[1, alphabet_size])
    saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
    saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
    reset_sample_state = tf.group(
        saved_sample_output.assign(tf.zeros([1, num_nodes])),
        saved_sample_state.assign(tf.zeros([1, num_nodes])))
    sample_output, sample_state = lstm_cell(
        sample_input, saved_sample_output, saved_sample_state)
    with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                  saved_sample_state.assign(sample_state)]):
        sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))


with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    mean_loss = 0
    for step in range(num_steps):
        batches = train_batches.next()
        feed_dict = dict()
        for i in range(num_unrollings + 1):
            feed_dict[train_data[i]] = batches[i]

        _, l, predictions, lr = session.run(
            [optimizer, loss, train_prediction, tf_lr], feed_dict=feed_dict)
        mean_loss += l
        if step % summary_frequency == 0:
            if step > 0:
                mean_loss = mean_loss / summary_frequency
            # The mean loss is an estimate of the loss over the last few batches.
            print('Average loss at step', step, ':', mean_loss, 'learning rate:', lr)
            mean_loss = 0
            labels = np.concatenate(list(batches)[1:])
            print('Minibatch perplexity: %.2f'
                % float(np.exp(training.utils.logprob(predictions, labels))))
            if step % (summary_frequency * 10) == 0:
                # Generate some samples.
                print('=' * 80)
                for _ in range(5):
                    feed = training.utils.sample(
                        training.utils.random_distribution(alphabet_size))
                    sentence = utils.characters(feed)[0]
                    reset_sample_state.run()
                    for _ in range(79):
                        prediction = sample_prediction.eval({sample_input: feed})
                        feed = training.utils.sample(prediction)
                        sentence += utils.characters(feed)[0]
                    print(sentence)
                print('=' * 80)
            # Measure validation set perplexity.
            reset_sample_state.run()
            valid_logprob = 0
            for _ in range(valid_size):
                b = valid_batches.next()
                predictions = sample_prediction.eval({sample_input: b[0]})
                valid_logprob = valid_logprob + training.utils.logprob(predictions, b[1])
            print('Validation set perplexity: %.2f' % float(np.exp(
                valid_logprob / valid_size)))
