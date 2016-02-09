import numpy as np
from matplotlib import pylab
import random
from sklearn.manifold import TSNE
import tensorflow as tf

def gaussian_weights_variable(shape):
    # The std is computed such that the variance of the sums
    # is 1/3, in order not to saturate the non-linearity too much.
    std = 1.0 / np.sqrt(3.0 * shape[0])
    return tf.Variable(tf.truncated_normal(shape, 0.0, std))

def uniform_weights_variable(shape):
    max_abs = 1.0 / np.sqrt(shape[0])
    return tf.Variable(tf.random_uniform(shape, -max_abs, max_abs))

def gaussian_kernels_variable(shape):
    # [height, width, depth, num_kernels]
    num_inputs = np.prod(shape[:3])
    std = 1.0 / np.sqrt(3.0 * num_inputs)
    return tf.Variable(tf.truncated_normal(shape, 0.0, std))

def uniform_kernels_variable(shape):
    num_inputs = np.prod(shape[:3])
    max_abs = 1.0 / np.sqrt(num_inputs)
    return tf.Variable(tf.random_uniform(shape, -max_abs, max_abs))

def logprob(predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1

def sample(prediction):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[1, prediction[0].shape[0]], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p

def random_distribution(alphabet_size):
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, alphabet_size])
    return b/np.sum(b, 1)[:,None]

# Applies non-linear dimensionalty reduction using tSNE and plots
# the words.
def plot_embedding(embeddings, labels):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(embeddings)

    pylab.figure(figsize=(15,15))  # in inches
    for i, label in enumerate(labels):
        x, y = two_d_embeddings[i,:]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2),
            textcoords='offset points', ha='right', va='bottom')
    pylab.show()
