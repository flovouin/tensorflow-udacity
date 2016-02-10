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
import training.graph_optimisation
import training.models
import training.utils

data_folder = '../data'
vocabulary_size = 50000
alphabet_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
valid_size = 1000

batch_size = 64
num_unrollings = 10
learning_rate = 10.0
gradient_max_value = 1.25
num_steps = 10001
summary_frequency = 100
exp_decay = {
    'decay_steps': 2500,
    'decay_rate': 0.5,
    'staircase': True
}

num_nodes = 64

embedding_lr = 1.0
embedding_size = 15 # Dimension of the embedding vector.
dropout_prob = 0.2
skip_batch_size = 128
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
# valid_size = 16 # Random set of words to evaluate similarity on.
# valid_window = 100 # Only pick dev samples in the head of the distribution.
# valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.
num_steps_embedding = 100001
# num_points = 400

# Preparing or loading the dataset.
text, _, _, _, _ = text8.prepare_dataset(vocabulary_size, data_folder)

valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)

# train_batches = training.batch.BatchGenerator(train_text, batch_size, num_unrollings, alphabet_size)
# valid_batches = training.batch.BatchGenerator(valid_text, 1, 1, alphabet_size)
#
# # Model on single characters.
# tf_graph, optimizer, loss, tf_predictions, reset_sample_state = \
#     training.models.lstm_model(alphabet_size, alphabet_size, None, num_nodes,
#         num_unrollings, batch_size, learning_rate, exp_decay, gradient_max_value)
#
# # Running.
# training.graph_optimisation.run_lstm(tf_graph, optimizer, loss,
#     reset_sample_state, tf_predictions, train_batches, valid_batches,
#     valid_size, num_steps, utils.characters, summary_frequency)

# Model on bigrams.
# Learning an embedding for bigrams.
bialphabet_size = alphabet_size ** 2
train_bitext = utils.char2bigrams(train_text)
valid_bitext = utils.char2bigrams(valid_text)

skipgram_batches = training.batch.SkipgramBatchGenerator(train_bitext, skip_batch_size,
    num_skips, skip_window)

tf_graph, optimizer, loss, normalized_embeddings, _ = \
    training.models.skipgram_model(bialphabet_size, embedding_size, skip_batch_size,
        num_sampled, None, embedding_lr)

normalized_embeddings = training.graph_optimisation.run_embedding(
    tf_graph, optimizer, loss, None, normalized_embeddings, skipgram_batches,
    None, None, num_steps_embedding)

# LSTM model for bigrams.
train_bibatches = training.batch.BatchGenerator(
    train_bitext, batch_size, num_unrollings, bialphabet_size)
valid_bibatches = training.batch.BatchGenerator(valid_bitext, 1, 1, bialphabet_size)

# Model.
tf_graph, optimizer, loss, tf_predictions, reset_sample_state = \
    training.models.lstm_model(embedding_size, bialphabet_size, normalized_embeddings,
        num_nodes, num_unrollings, batch_size, learning_rate, exp_decay, gradient_max_value,
        dropout_prob)

# Running.
training.graph_optimisation.run_lstm(tf_graph, optimizer, loss,
    reset_sample_state, tf_predictions, train_bibatches, valid_bibatches,
    valid_size, num_steps, utils.biprobstochar, summary_frequency)
