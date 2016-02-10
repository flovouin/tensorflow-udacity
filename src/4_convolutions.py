#! /usr/bin/env python3
#
# Solutions to the fourth assignment of Tensorflow's Udacity tutorial.
#
# Flo Vouin - 2016

import numpy as np
import tensorflow as tf

from dataset import not_mnist, utils
import training.graph_optimisation
import training.models
import training.utils

train_size = 200000
valid_size = 10000
data_folder = '../data'
num_channels = 1
num_labels = not_mnist.num_classes

batch_size = 16
num_steps = 10001

dropout_prob = 0.3
learning_rate = 0.05
decay_steps = 500
decay_rate = 0.75

patch_size = 5
depth = 16
num_hidden = 64

# Creating the dataset (or retrieving it).
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = \
    not_mnist.prepare_dataset(train_size, valid_size, data_folder)

# Formatting the data by converting labels to one-hot encoding.
train_labels = utils.idx_to_onehot(train_labels, num_labels)
valid_labels = utils.idx_to_onehot(valid_labels, num_labels)
test_labels = utils.idx_to_onehot(test_labels, num_labels)
# Adding an extra dimension representing the number of channels.
train_dataset = train_dataset.reshape(
    (-1, not_mnist.image_size, not_mnist.image_size, num_channels)).astype(np.float32)
valid_dataset = valid_dataset.reshape(
    (-1, not_mnist.image_size, not_mnist.image_size, num_channels)).astype(np.float32)
test_dataset = test_dataset.reshape(
    (-1, not_mnist.image_size, not_mnist.image_size, num_channels)).astype(np.float32)

# Creating a convolutional neural network.
graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,
        shape=(batch_size, not_mnist.image_size, not_mnist.image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    layer1_weights = training.utils.gaussian_kernels_variable(
        [patch_size, patch_size, num_channels, depth])
    layer1_biases = tf.Variable(tf.zeros([depth]))

    layer2_weights = training.utils.gaussian_kernels_variable(
        [patch_size, patch_size, depth, depth])
    layer2_biases = tf.Variable(tf.zeros([depth]))

    flatten_input_size = int(not_mnist.image_size / 4 * not_mnist.image_size / 4 * depth)
    layer3_weights = training.utils.gaussian_weights_variable(
        [flatten_input_size, num_hidden])
    layer3_biases = tf.Variable(tf.zeros([num_hidden]))

    layer4_weights = training.utils.gaussian_weights_variable(
        [num_hidden, num_labels])
    layer4_biases = tf.Variable(tf.zeros([num_labels]))

    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        hidden = tf.nn.dropout(hidden, 1 - dropout_prob)

        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        hidden = tf.nn.dropout(hidden, 1 - dropout_prob)

        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        hidden = tf.nn.dropout(hidden, 1 - dropout_prob)

        return tf.matmul(hidden, layer4_weights) + layer4_biases

    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    global_step = tf.Variable(0)
    tf_lr = tf.train.exponential_decay(
        learning_rate, global_step, decay_steps, decay_rate, staircase = True)
    optimizer = tf.train.GradientDescentOptimizer(tf_lr).minimize(
        loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

tf_graph = {
    'graph': graph,
    'data_ph': tf_train_dataset,
    'labels_ph': tf_train_labels }
tf_predictions = [train_prediction, valid_prediction, test_prediction]

training.graph_optimisation.run(tf_graph, optimizer, loss, tf_predictions,
    train_dataset, train_labels, valid_labels, test_labels, num_steps, batch_size,
    verbose_frequency = 500)
