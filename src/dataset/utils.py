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
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def flatten_batch(data):
    return data.reshape((len(data), -1))

def idx_to_onehot(idx, num_classes):
    onehot = (np.arange(num_classes) == idx[:,None]).astype(np.float32)
    return onehot

def accuracy(predictions, labels):
      return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

# Character conversion.
first_letter = ord(string.ascii_lowercase[0])

def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character:', char)
        return 0

def id2char(dictid):
    if dictid > 0:
        return chr(dictid + first_letter - 1)
    else:
        return ' '

def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (mostl likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s

def ids(chars):
    """Converts characters to IDs."""
    return [char2id(c) for c in chars]
