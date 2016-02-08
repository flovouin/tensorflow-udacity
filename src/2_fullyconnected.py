#! /usr/bin/env python3

import numpy as np
import tensorflow as tf

from dataset import not_mnist, utils
import training.graph_optimisation
import training.utils

train_size = 200000
valid_size = 10000
data_folder = '../data'
input_size = not_mnist.image_size ** 2
num_labels = not_mnist.num_classes

batch_size = 128
learning_rate = 0.5
num_steps = 5001
num_hidden_nodes = 1024

# Creating the dataset (or retrieving it).
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = \
    not_mnist.prepare_dataset(train_size, valid_size, data_folder)

# Formatting the data by flattening the images and converting
# labels to one-hot encoding.
train_dataset = utils.flatten_batch(train_dataset)
train_labels = utils.idx_to_onehot(train_labels, num_labels)
valid_dataset = utils.flatten_batch(valid_dataset)
valid_labels = utils.idx_to_onehot(valid_labels, num_labels)
test_dataset = utils.flatten_batch(test_dataset)
test_labels = utils.idx_to_onehot(test_labels, num_labels)

# Creating the graph.
graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, input_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    # weights = tf.Variable(tf.truncated_normal([input_size, num_labels]))
    weights = training.utils.gaussian_weights_variable([input_size, num_labels])
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

tf_graph = {
    'graph': graph,
    'data_ph': tf_train_dataset,
    'labels_ph': tf_train_labels }
tf_predictions = [train_prediction, valid_prediction, test_prediction]

training.graph_optimisation.run(tf_graph, optimizer, loss, tf_predictions,
    train_dataset, train_labels, valid_labels, test_labels, num_steps, batch_size)

# Adding a hidden layer.
print('\nRelaunching optimisation with a hidden layer...')

def create_model(weights, inputs, labels = None):
    hidden_units = tf.nn.relu(tf.matmul(inputs, weights[0]) + weights[1])
    out_logits = tf.matmul(hidden_units, weights[2]) + weights[3]
    out_prob = tf.nn.softmax(out_logits)

    if labels is not None:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out_logits, labels))
        return out_prob, loss
    return out_prob

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, input_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights1 = training.utils.gaussian_weights_variable([input_size, num_hidden_nodes])
    biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
    weights2 = training.utils.gaussian_weights_variable([num_hidden_nodes, num_labels])
    biases2 = tf.Variable(tf.zeros([num_labels]))
    weights = [weights1, biases1, weights2, biases2]

    # Training computation.
    train_prediction, loss = create_model(weights, tf_train_dataset, tf_train_labels)
    valid_prediction = create_model(weights, tf_valid_dataset)
    test_prediction = create_model(weights, tf_test_dataset)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

tf_graph = {
    'graph': graph,
    'data_ph': tf_train_dataset,
    'labels_ph': tf_train_labels }
tf_predictions = [train_prediction, valid_prediction, test_prediction]

training.graph_optimisation.run(tf_graph, optimizer, loss, tf_predictions,
    train_dataset, train_labels, valid_labels, test_labels, num_steps, batch_size)
