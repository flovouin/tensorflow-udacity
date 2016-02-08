import collections
import numpy as np

# Defining how batches are created when training the skip-gram model.
class SkipgramBatchGenerator:
    def __init__(self, text, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window

        self._text_index = 0
        self._text = text
        self._batch_size = batch_size
        self._num_skips = num_skips
        self._skip_window = skip_window

    def next(self):
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
            target = self._skip_window  # target label at the center of the buffer
            batch[i * self._num_skips:(i + 1) * self._num_skips] = buffer[self._skip_window]

            targets_idx = np.random.permutation(span - 1)[:self._num_skips]
            targets_idx = targets_idx + (targets_idx >= self._skip_window).astype(int)
            labels[i * self._num_skips:(i + 1) * self._num_skips, 0] = [buffer[t] for t in targets_idx]

            append_to_buffer()

        return batch, labels

class CBOWBatchGenerator:
    def __init__(self, text, batch_size, skip_window):
        context_length = 2 * skip_window
        assert batch_size % context_length == 0

        self._text_index = 0
        self._text = text
        self._batch_size = batch_size
        self._skip_window = skip_window

    def next(self):
        context_length = 2 * self._skip_window

        batch = np.ndarray(shape=(context_length * self._batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self._batch_size, 1), dtype=np.int32)
        span = 2 * self._skip_window + 1 # [ skip_window target skip_window ]

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
                    context_idx = context_idx + 1 # Skipping target
                batch[i * context_length + j] = buffer[context_idx]
                context_idx = context_idx + 1
            labels[i, 0] = buffer[self._skip_window]

            append_to_buffer()

        return batch, labels

class BatchGenerator(object):
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
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches
