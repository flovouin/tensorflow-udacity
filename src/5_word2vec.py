#! /usr/bin/env python3
#
# Solutions to the fifth assignment of Tensorflow's Udacity tutorial.
#
# Flo Vouin - 2016

import collections
import numpy as np
import random
import tensorflow as tf

from dataset import text8
from training import batch, graph_optimisation, models
import training.utils

data_folder = '../data'
vocabulary_size = 50000

batch_size_skip = 128
batch_size_cbow = 64
learning_rate = 1.0
num_steps = 100001

embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.

num_points = 400

# Preparing or loading the dataset.
_, data, count, dictionary, reverse_dictionary = \
    text8.prepare_dataset(vocabulary_size, data_folder)

# Initialisaing batch generators.
skipgram_batches = batch.SkipgramBatchGenerator(data, batch_size_skip, num_skips, skip_window)
cbow_batches = batch.CBOWBatchGenerator(data, batch_size_cbow, skip_window)

# Skipgram.
tf_graph, optimizer, loss, normalized_embeddings, similarity = \
    models.skipgram_model(vocabulary_size, embedding_size, batch_size_skip, num_sampled,
        valid_examples, learning_rate)

print('Training using skipgram model...')
normalized_embeddings = graph_optimisation.run_embedding(tf_graph, optimizer, loss, similarity,
    normalized_embeddings, skipgram_batches, valid_examples, reverse_dictionary, num_steps)

training.utils.plot_embedding(normalized_embeddings[1:num_points+1],
    [reverse_dictionary[i] for i in range(1, num_points+1)])

# CBOW.
context_length = 2 * skip_window

tf_graph, optimizer, loss, normalized_embeddings, similarity = \
    models.cbow_model(vocabulary_size, embedding_size, context_length, batch_size_cbow,
        num_sampled, valid_examples, learning_rate)

print('\nTraining using CBOW model...')
normalized_embeddings = graph_optimisation.run_embedding(tf_graph, optimizer, loss, similarity,
    normalized_embeddings, cbow_batches, valid_examples, reverse_dictionary, num_steps)

training.utils.plot_embedding(normalized_embeddings[1:num_points+1],
    [reverse_dictionary[i] for i in range(1, num_points+1)])
