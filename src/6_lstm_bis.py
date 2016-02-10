#! /usr/bin/env python3

import os
import numpy as np
import random
import string
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

from dataset import text8
import dataset.utils
import training.utils

data_folder = '../data'
checkpoint_path = os.path.join(data_folder, "reverse.ckpt")
vocabulary_size = 50000
num_valid_words = 1000

num_units = 64
input_size = len(string.ascii_lowercase) + 3 # [a-z] + PAD + GO + EOS
max_word_length = 10
batch_size = 64
max_gradient_norm = 5.0
learning_rate = 1.0
learning_rate_decay_factor = 0.8
num_steps = 10000

verbose_frequency = 200

PAD_ID = 0
GO_ID  = 1
EOS_ID = 2

class ReverseWordModel(object):
    def __init__(self, vocab_size, sequence_length, num_units,
        max_gradient_norm, batch_size, learning_rate,
        learning_rate_decay_factor):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        w = training.utils.gaussian_weights_variable([num_units, self.vocab_size])
        b = tf.Variable(tf.zeros([self.vocab_size]))

        lstm_cell = rnn_cell.LSTMCell(num_units, vocab_size)

        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for _ in range(sequence_length):
            self.encoder_inputs.append(tf.placeholder(
                tf.float32, shape=(batch_size, self.vocab_size)))
            self.decoder_inputs.append(tf.placeholder(
                tf.float32, shape=(batch_size, self.vocab_size)))
            self.target_weights.append(tf.placeholder(
                tf.float32, shape=(batch_size,)))

        # Decoder has one extra cell because it starts with the GO symbol.
        self.decoder_inputs.append(tf.placeholder(
            tf.float32, shape=(batch_size, self.vocab_size)))
        self.target_weights.append(np.ones((batch_size,)))

        # Targets used by the sequence loss must be integer indices.
        targets = [tf.cast(tf.argmax(i, 1), dtype=tf.int32)
            for i in self.decoder_inputs[1:]]

        # Where is feed_previous?
        outputs, self.state = seq2seq.basic_rnn_seq2seq(
            self.encoder_inputs, self.decoder_inputs, lstm_cell)

        self.logits = [tf.nn.xw_plus_b(o, w, b) for o in outputs]
        self.loss = seq2seq.sequence_loss(self.logits[:self.sequence_length],
            targets, self.target_weights[:self.sequence_length],
            self.vocab_size)

        params = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, self.gradient_norms = tf.clip_by_global_norm(
            gradients, max_gradient_norm)
        self.updates = opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
        forward_only):
        input_feed = {}
        for l in range(self.sequence_length):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[self.sequence_length].name
        input_feed[last_target] = np.zeros((self.batch_size, self.vocab_size))

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates, self.gradient_norms, self.loss]
        else:
            output_feed = [self.loss]
            for l in self.logits[:self.sequence_length]:
                output_feed.append(l)

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            # Gradient norm, loss, no outputs.
            return outputs[1], outputs[2], None
        else:
            # No gradient norm, loss, outputs.
            return None, outputs[0], outputs[1:]

    def get_batch(self, data):
        encoder_inputs, decoder_inputs = [], []

        for _ in range(self.batch_size):
            encoder_input, decoder_input = random.choice(data)

            # Encoder inputs are padded.
            encoder_pad = [PAD_ID] * (self.sequence_length - len(encoder_input))
            encoder_inputs.append(list(encoder_pad + encoder_input))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = self.sequence_length - len(decoder_input) - 1
            decoder_inputs.append([GO_ID] + decoder_input +
                                 [PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(self.sequence_length):
            in_idx = [encoder_inputs[batch_idx][length_idx]
                for batch_idx in range(self.batch_size)]
            batch_encoder_inputs.append(dataset.utils.idx_to_onehot(
                np.array(in_idx), self.vocab_size))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(self.sequence_length):
            out_idx = [decoder_inputs[batch_idx][length_idx]
                for batch_idx in range(self.batch_size)]
            batch_decoder_inputs.append(dataset.utils.idx_to_onehot(
                np.array(out_idx), self.vocab_size))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in range(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < self.sequence_length - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == self.sequence_length - 1 \
                    or target == PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

first_letter = ord(string.ascii_lowercase[0])
def char2id(char):
    if char in bytearray(string.ascii_lowercase, 'utf-8'):
        return char - first_letter + 3
    else:
        print('Unexpected character:', char)
        return 0

def id2char(dictid):
    if dictid > 2:
        return chr(dictid + first_letter - 3)
    elif dictid == EOS_ID:
        return '.'
    else:
        return ' '

def create_set(words):
    data = []
    for w in words:
        in_word = [char2id(c) for c in w]
        out_word = in_word[::-1] + [EOS_ID]
        data.append((in_word, out_word))
    return data

_, _, _, dictionary, _ = text8.prepare_dataset(vocabulary_size, data_folder)
dictionary.pop("UNK", None)
too_long = {w for w in dictionary.keys() if len(w) > max_word_length}
for w in too_long:
    dictionary.pop(w, None)

words = list(dictionary.keys())

train_words = words[num_valid_words:]
valid_words = words[:num_valid_words]

train_data = create_set(train_words)
valid_data = create_set(valid_words)

with tf.Session() as session:
    model = ReverseWordModel(input_size, max_word_length + 2, num_units,
        max_gradient_norm, batch_size, learning_rate,
        learning_rate_decay_factor)
    session.run(tf.initialize_all_variables())

    loss = 0.0
    current_step = 0
    previous_losses = []
    for i in range(num_steps):
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            train_data)
        _, step_loss, _ = model.step(session, encoder_inputs, decoder_inputs,
                                     target_weights, False)
        loss += step_loss / verbose_frequency
        current_step += 1

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % verbose_frequency == 0:
            # Print statistics for the previous epoch.
            perplexity = np.exp(loss) if loss < 300 else float('inf')
            print ("global step %d learning rate %.4f perplexity %.2f" \
                  % (model.global_step.eval(), model.learning_rate.eval(),
                     perplexity))

            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                session.run(model.learning_rate_decay_op)

            previous_losses.append(loss)
            loss = 0.0

            # Run evals on development set and print their perplexity.
            encoder_inputs, decoder_inputs, target_weights = \
                model.get_batch(valid_data)
            _, eval_loss, _ = model.step(session, encoder_inputs, decoder_inputs,
                                         target_weights, True)
            eval_ppx = np.exp(eval_loss) if eval_loss < 300 else float('inf')
            print("  eval: perplexity %.2f" % (eval_ppx))

            model.saver.save(session, checkpoint_path, global_step=model.global_step)

            # Show a few results randomly
            encoder_inputs, decoder_inputs, target_weights = \
                model.get_batch(valid_data)

            outputs_idx = []
            for l in range(len(decoder_inputs)):
                _, _, outputs = model.step(session, encoder_inputs, decoder_inputs,
                                     target_weights, True)

                cur_out_idx = np.argmax(outputs[l], 1)
                outputs_idx.append(cur_out_idx)
                if l < len(decoder_inputs) - 1:
                    decoder_inputs[l + 1] = dataset.utils.idx_to_onehot(
                        cur_out_idx, input_size)

            inputs_idx = [np.argmax(i, 1) for i in encoder_inputs]

            results = []
            for b in range(batch_size):
                results.append((''.join([id2char(inputs_idx[c][b]) for c in range(len(inputs_idx))]),
                                ''.join([id2char(outputs_idx[c][b]) for c in range(len(outputs_idx))])))
            print(results[:5])
