# Provides utilities to easily generate batches when training
# various models: word embeddings (skipgram and CBOW), and RNNs.
# Derived from Tensorflow's Udacity tutorial.
#
# Flo Vouin - 2015

import collections
import numpy as np

# Defining how batches are created when training the skip-gram model.
class SkipgramBatchGenerator:
    """This class generates skipgram batches, i.e. training samples for which
    the centre word of a window is the input and the labels / desired outputs
    are words extracted from the context (i.e. other words in the same window)."""

    def __init__(self, text, batch_size, num_skips, skip_window):
        """num_skips: Number of skipgrams in a batch.
        skip_window: length of the window on each side of the centre word."""
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window

        self._text_index = 0
        self._text = text
        self._batch_size = batch_size
        self._num_skips = num_skips
        self._skip_window = skip_window

    def next(self):
        """Returns the next batch of data."""
        batch = np.ndarray(shape=(self._batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self._batch_size, 1), dtype=np.int32)
        span = 2 * self._skip_window + 1 # [ skip_window target skip_window ]

        buffer = collections.deque(maxlen=span)
        def append_to_buffer():
            buffer.append(self._text[self._text_index])
            self._text_index = (self._text_index + 1) % len(self._text)

        for _ in range(span):
            append_to_buffer()

        for i in range(self._batch_size // self._num_skips):
            # For a given skipgram, the input is the word at the centre of the window.
            batch[i * self._num_skips:(i + 1) * self._num_skips] = buffer[self._skip_window]

            # Labels / outputs are randomly sampled from the context of the centre word.
            targets_idx = np.random.permutation(span - 1)[:self._num_skips]
            targets_idx = targets_idx + (targets_idx >= self._skip_window).astype(int)
            labels[i * self._num_skips:(i + 1) * self._num_skips, 0] = [buffer[t] for t in targets_idx]

            # Goes to the next word by shifting the register and adding a new word.
            append_to_buffer()

        return batch, labels

class CBOWBatchGenerator:
    """This class generates continuous bag of words batches, i.e. training samples
    for which the representations of several context words are averaged, and the
    expected output is the centre word in the window."""

    def __init__(self, text, batch_size, skip_window):
        """The size of the batch will actually be batch_size * 2 * skip_window.
        batch_size is the number of output labels, but each of them depends on
        2 * skip_windows inputs that are averaged together."""
        context_length = 2 * skip_window
        assert batch_size % context_length == 0

        self._text_index = 0
        self._text = text
        self._batch_size = batch_size
        self._skip_window = skip_window

    def next(self):
        """Returns the next batch of data."""
        # The number of embeddings that will be averaged in order to estimate
        # the output/centre word.
        context_length = 2 * self._skip_window

        batch = np.ndarray(shape=(context_length * self._batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self._batch_size, 1), dtype=np.int32)
        span = 2 * self._skip_window + 1  # [skip_window target skip_window].

        buffer = collections.deque(maxlen=span)
        def append_to_buffer():
            buffer.append(self._text[self._text_index])
            self._text_index = (self._text_index + 1) % len(self._text)

        for _ in range(span):
            append_to_buffer()

        for i in range(self._batch_size):
            context_idx = 0
            for j in range(context_length):
                if context_idx == self._skip_window:
                    context_idx = context_idx + 1  # Skipping target.
                batch[i * context_length + j] = buffer[context_idx]
                context_idx = context_idx + 1
            labels[i, 0] = buffer[self._skip_window]

            append_to_buffer()

        return batch, labels

class BatchGenerator(object):
    """Generates batches for a RNN when learning a character model."""

    def __init__(self, text, batch_size, num_unrollings, alphabet_size):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        self._alphabet_size = alphabet_size
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, self._alphabet_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, self._text[self._cursor[b]]] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones."""
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches
