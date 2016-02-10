#! /usr/bin/env python3
#
# Solution to the last part of the sixth assignment of
# Tensorflow's Udacity tutorial. It is actually trained
# to simply reverse words, and not whole sentences.
#
# Flo Vouin - 2016

import numpy as np
import os
import string
import tensorflow as tf

from dataset import text8
import dataset.utils
import training.models
import training.utils

data_folder = '../data'
vocabulary_size = 50000
num_valid_words = 1000

num_units = 64
input_size = len(string.ascii_lowercase) + 3 # [a-z] + PAD + GO + EOS
max_word_length = 10
batch_size = 64
max_gradient_norm = 5.0
learning_rate = 1.0
learning_rate_decay_factor = 0.8
num_steps = 5000

verbose_frequency = 200

def create_set(words):
    data = []
    for w in words:
        in_word = [dataset.utils.seq2seq_char2id(c) for c in w]
        # The output should be the reversed word and the
        # "end of sequence" symbol.
        out_word = in_word[::-1] + [dataset.utils.EOS_ID]
        data.append((in_word, out_word))
    return data

_, _, _, dictionary, _ = text8.prepare_dataset(vocabulary_size, data_folder)

# Removing UNK token and words that are longer than the max length allowed.
dictionary.pop("UNK", None)
too_long = {w for w in dictionary.keys() if len(w) > max_word_length}
for w in too_long:
    dictionary.pop(w, None)

# Creating a simple dataset from the 50000 words (or less) in the vocabulary.
words = list(dictionary.keys())

train_words = words[num_valid_words:]
valid_words = words[:num_valid_words]

train_data = create_set(train_words)
valid_data = create_set(valid_words)

# Running the optimisation.
with tf.Session() as session:
    model = training.models.ReverseWordModel(input_size, max_word_length + 2, num_units,
        max_gradient_norm, batch_size, learning_rate,
        learning_rate_decay_factor)
    session.run(tf.initialize_all_variables())

    loss = 0.0
    current_step = 0
    previous_losses = []
    for i in range(num_steps):
        encoder_inputs, decoder_inputs, target_weights = \
            model.get_batch(train_data)

        _, step_loss, _ = model.step(
            session, encoder_inputs, decoder_inputs, target_weights, False)
        loss += step_loss / verbose_frequency
        current_step += 1

        # Checking progress.
        if current_step % verbose_frequency == 0:
            perplexity = np.exp(loss) if loss < 300 else float('inf')
            print ("Global step: %d, learning rate: %.4f, perplexity: %.2f" \
                  % (model.global_step.eval(), model.learning_rate.eval(),
                     perplexity))

            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                session.run(model.learning_rate_decay_op)

            previous_losses.append(loss)
            loss = 0.0

            # Run evals on validation set and print their perplexity.
            encoder_inputs, decoder_inputs, target_weights = \
                model.get_batch(valid_data)
            _, eval_loss, _ = model.step(
                session, encoder_inputs, decoder_inputs, target_weights, True)
            eval_ppx = np.exp(eval_loss) if eval_loss < 300 else float('inf')
            print("Validation perplexity: %.2f" % (eval_ppx))

            # Displays a few results randomly from the validation set.
            encoder_inputs, decoder_inputs, target_weights = \
                model.get_batch(valid_data)

            # This is a bit messy: to avoid creating a separate model,
            # and to be able to feed previous results into following
            # cells, the network is evaluated len(decoder_inputs) times,
            # each time a new decoder_inputs is replaced with the output of
            # the previous cell.
            # This means that we actually use the output of a single cell
            # each time we run the whole unrolled network. Also, we could
            # stop when we get an EOS symbol.
            outputs_idx = []
            for l in range(len(decoder_inputs)):
                _, _, outputs = model.step(session, encoder_inputs, decoder_inputs,
                                     target_weights, True)

                # Getting the output of the network as the index of the most
                # likely symbol for the current output.
                cur_out_idx = np.argmax(outputs[l], 1)
                outputs_idx.append(cur_out_idx)

                # Feeding the current output back into the network as the
                # following input. It has to be converted back to a one-hot
                # encoding first.
                if l < len(decoder_inputs) - 1:
                    decoder_inputs[l + 1] = dataset.utils.idx_to_onehot(
                        cur_out_idx, input_size)

            inputs_idx = [np.argmax(i, 1) for i in encoder_inputs]

            results = []
            for b in range(batch_size):
                results.append((''.join([dataset.utils.seq2seq_id2char(inputs_idx[c][b]) for c in range(len(inputs_idx))]),
                                ''.join([dataset.utils.seq2seq_id2char(outputs_idx[c][b]) for c in range(len(outputs_idx))])))
            print(results[:5])
