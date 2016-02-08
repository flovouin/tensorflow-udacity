#! /usr/bin/env python3

import collections
import numpy as np
import random
import tensorflow as tf
from matplotlib import pylab
from sklearn.manifold import TSNE

from dataset import text8
from training import batch

data_folder = '../data'
vocabulary_size = 50000

batch_size = 128
batch_size_cbow = 64
learning_rate = 1.0
num_steps = 100001

embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
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

# Defining how batches are created when training the skip-gram model.
skipgram_batches = batch.SkipgramBatchGenerator(data, batch_size, num_skips, skip_window)
cbow_batches = batch.CBOWBatchGenerator(data, batch_size_cbow, skip_window)

# General optimisation function.
def run(tf_graph, optimizer, loss, similarity, normalized_embeddings,
    batches, num_steps, verbose_loss_frequency = 2000, verbose_frequency = 10000):
    with tf.Session(graph=tf_graph['graph']) as session:
        tf.initialize_all_variables().run()

        average_loss = 0.0
        for step in range(num_steps):
            batch_data, batch_labels = batches.next()

            _, l = session.run([optimizer, loss],
                    feed_dict={
                        tf_graph['data_ph'] : batch_data,
                        tf_graph['labels_ph'] : batch_labels})
            average_loss += l

            if step % verbose_loss_frequency == 0:
                if step > 0:
                    average_loss = average_loss / verbose_loss_frequency
                print("Average loss at step", step, ":", average_loss)
                average_loss = 0

            if step % verbose_frequency == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8 # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = "Nearest to %s:" % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log = "%s %s," % (log, close_word)
                    print(log)
        return normalized_embeddings.eval()

# Applies non-linear dimensionalty reduction using tSNE and plots
# the words.
def plot(embeddings, labels):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(embeddings)

    pylab.figure(figsize=(15,15))  # in inches
    for i, label in enumerate(labels):
        x, y = two_d_embeddings[i,:]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2),
            textcoords='offset points', ha='right', va='bottom')
    pylab.show()

# Skip-gram.
graph = tf.Graph()
with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    tf_train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables.
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # This is actually transposed compared to usual layer weights. The std is
    # deduced accordingly, from the input size (embedding_size).
    softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                           stddev=1.0 / np.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model.
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, tf_train_dataset)
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
        softmax_weights, softmax_biases, embed, tf_train_labels, num_sampled, vocabulary_size))

    # Optimizer.
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

tf_graph = {
    'graph': graph,
    'data_ph': tf_train_dataset,
    'labels_ph': tf_train_labels }

print('Training using skipgram model...')
normalized_embeddings = run(tf_graph, optimizer, loss, similarity,
    normalized_embeddings, skipgram_batches, num_steps)

plot(normalized_embeddings[1:num_points+1],
    [reverse_dictionary[i] for i in range(1, num_points+1)])

# CBOW.
context_length = 2 * skip_window
input_batch_size = context_length * batch_size_cbow

graph = tf.Graph()
with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(tf.int32, shape=[input_batch_size])
    tf_train_labels = tf.placeholder(tf.int32, shape=[batch_size_cbow, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    word_mean_op = tf.constant(
        np.kron(np.eye(batch_size_cbow), np.ones([1, context_length])), dtype=tf.float32)

    # Variables.
    embeddings = tf.Variable(tf.random_uniform(
        [vocabulary_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(tf.truncated_normal(
        [vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model.
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, tf_train_dataset)
    word_means = tf.matmul(word_mean_op, embed)
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
        softmax_weights, softmax_biases, word_means, tf_train_labels, num_sampled, vocabulary_size))

    # Optimizer.
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

tf_graph = {
    'graph': graph,
    'data_ph': tf_train_dataset,
    'labels_ph': tf_train_labels }

print('\nTraining using CBOW model...')
normalized_embeddings = run(tf_graph, optimizer, loss, similarity,
    normalized_embeddings, cbow_batches, num_steps)

plot(normalized_embeddings[1:num_points+1],
    [reverse_dictionary[i] for i in range(1, num_points+1)])
