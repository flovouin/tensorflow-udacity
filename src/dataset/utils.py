# Provides general utilities when preprocessing datasets.
# Derived from Tensorflow's Udacity tutorial.
#
# Flo Vouin - 2016

import numpy as np
import os
from six.moves.urllib.request import urlretrieve
import string

def maybe_download(url, filepath, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filepath):
        urlretrieve(url, filepath)

    statinfo = os.stat(filepath)
    if statinfo.st_size != expected_bytes:
        raise Exception(
            'Failed to verify' + filename + '. Can you get to it with a browser?')

    return filepath

def randomize(dataset, labels):
    """Shuffles a dataset and the corresponding labels."""
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def flatten_batch(data):
    """Flattens a batch of multidimensional data and returns a 2D array."""
    return data.reshape((len(data), -1))

def idx_to_onehot(idx, num_classes):
    """Converts class indices to the one-hot representation."""
    onehot = (np.arange(num_classes) == idx[:,None]).astype(np.float32)
    return onehot

def accuracy(predictions, labels):
    """Compares probability distributions with one-hot encoded labels and
    returns the corresponding accuracy as a percentage."""
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
        / predictions.shape[0])

# Character conversion.
first_letter = ord(string.ascii_lowercase[0])
alphabet_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '

def char2id(char):
    """Converts a single character to an index."""
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character:', char)
        return 0

def id2char(dictid):
    """Converts a single index back to a character."""
    if dictid > 0 and dictid < alphabet_size:
        return chr(dictid + first_letter - 1)
    elif dictid == 0:
        return ' '
    else:
        raise Exception('Unexpected index.')

def probs2chars(probabilities):
    """Turns a one-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2strings(batches):
    """Converts a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s

def chars2ids(chars):
    """Converts characters to IDs."""
    return [char2id(c) for c in chars]

def chars2bigrams(text):
    """Converts a sequence of characters to bigram indices."""
    bitext_length = len(text) // 2
    bitext = np.empty([bitext_length], dtype=np.int32)
    for i in range(bitext_length):
        bitext[i] = text[2*i] * alphabet_size + text[2*i + 1]
    return bitext

def bigrams2chars(bigrams):
    """Converts bigram indices to a list of 2-character strings."""
    chars = []
    for b in range(len(bigrams)):
        chr1 = bigrams[b] // alphabet_size
        chr2 = bigrams[b] - chr1 * alphabet_size
        chars.append(''.join([id2char(chr1), id2char(chr2)]))
    return chars

def biprobs2chars(probabilities):
    """Converts probability distributions over bigrams to a list
    of 2-character strings."""
    return bigrams2chars(np.argmax(probabilities, 1))

def bibatches2string(bibatches):
    """Converts a sequence of bigram probability batches back into
    their (most likely) string representation."""
    s = [''] * bibatches[0].shape[0]
    for b in bibatches:
        s = [''.join(x) for x in zip(s, biprobstochar(b))]
    return s

# For sequence to sequence model.
PAD_ID = 0
GO_ID  = 1
EOS_ID = 2

def seq2seq_char2id(char):
    if char in bytearray(string.ascii_lowercase, 'utf-8'):
        return char - first_letter + 3
    else:
        print('Unexpected character:', char)
        return 0

def seq2seq_id2char(dictid):
    if dictid > 2:
        return chr(dictid + first_letter - 3)
    elif dictid == EOS_ID:
        return '.'
    else:
        return ' '
