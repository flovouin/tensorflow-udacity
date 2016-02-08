#! /usr/bin/env python3

import numpy as np
import tensorflow as tf

from dataset import not_mnist, utils
import training.graph_optimisation
import training.models
import training.utils

train_size = 200000
valid_size = 10000
data_folder = '../data'
input_size = not_mnist.image_size ** 2
num_labels = not_mnist.num_classes

batch_size = 128
learning_rate = 0.5
num_steps = 3001

beta_logreg = 0.005

beta_hidden = 0.0005
num_hidden_nodes = 1024

num_batches = 10
dropout_prob = 0.3

nn_layers = [1024, 300, 50]
beta_nn = 0.0000005
exp_decay = {
    'decay_steps': 20000,
    'decay_rate': 0.5,
    'staircase': False
}

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

# Logistic regression with l2 regularisation
print('Logistic regression with l2 regularisation...')

tf_graph, optimizer, loss, tf_predictions = training.models.fully_connected_model(
    input_size, num_labels, [],
    valid_dataset, test_dataset, batch_size,
    learning_rate, beta = beta_logreg)

training.graph_optimisation.run(tf_graph, optimizer, loss, tf_predictions,
    train_dataset, train_labels, valid_labels, test_labels, num_steps, batch_size)

# Hidden layer with l2 regularisation
print('\nRelaunching optimisation with a hidden layer and regularisation...')

tf_graph, optimizer, loss, tf_predictions = training.models.fully_connected_model(
    input_size, num_labels, [num_hidden_nodes],
    valid_dataset, test_dataset, batch_size,
    learning_rate, beta = beta_hidden)

training.graph_optimisation.run(tf_graph, optimizer, loss, tf_predictions,
    train_dataset, train_labels, valid_labels, test_labels, num_steps, batch_size)

# Overfitting.
print('\nTraining on only {0} batches...'.format(num_batches))

num_samples = num_batches * batch_size
training.graph_optimisation.run(tf_graph, optimizer, loss, tf_predictions,
        train_dataset[:num_samples, :], train_labels[:num_samples, :],
        valid_labels, test_labels, num_steps, batch_size)

print('\nTraining on {0} batches with dropout...'.format(num_batches))

tf_graph, optimizer, loss, tf_predictions = training.models.fully_connected_model(
    input_size, num_labels, [num_hidden_nodes],
    valid_dataset, test_dataset, batch_size,
    learning_rate, beta = beta_hidden, dropout_prob = dropout_prob)

training.graph_optimisation.run(tf_graph, optimizer, loss, tf_predictions,
        train_dataset[:num_samples, :], train_labels[:num_samples, :],
        valid_labels, test_labels, num_steps, batch_size)

# Proper model with several hidden layers.
print('\nUsing 3 hidden layers with gradient decay (would be more useful with more epochs)...')

tf_graph, optimizer, loss, tf_predictions = training.models.fully_connected_model(
    input_size, num_labels, nn_layers,
    valid_dataset, test_dataset, batch_size,
    learning_rate, beta = beta_nn, dropout_prob = dropout_prob,
    exp_decay = exp_decay, method = 'adagrad')

training.graph_optimisation.run(tf_graph, optimizer, loss, tf_predictions,
    train_dataset, train_labels, valid_labels, test_labels, num_steps, batch_size)
