import numpy as np
import random
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
