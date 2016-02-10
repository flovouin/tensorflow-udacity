# Provides functions that create the models used in the various assignments.
# Derived from Tensorflow's Udacity tutorial.
#
# Flo Vouin - 2016

import numpy as np
import random
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq


import dataset.utils
import training.utils

def create_fully_connected_weights(input_size, num_labels, num_hidden_nodes):
    """Creates a list of weights corresponding to connections and biases of
    a multi-layer fully connected model."""
    layer_sizes = [*num_hidden_nodes, num_labels]

    weights = []
    prev_layer_size = input_size
    for cur_layer_size in layer_sizes:
        cur_weights = training.utils.gaussian_weights_variable(
            [prev_layer_size, cur_layer_size])
        cur_biases = tf.Variable(tf.zeros([cur_layer_size]))

        weights.append(cur_weights)
        weights.append(cur_biases)
        prev_layer_size = cur_layer_size

    return weights

def fully_connected_model(input_size, num_labels, num_hidden_nodes,
        valid_dataset, test_dataset, batch_size,
        learning_rate, beta = 0.0, dropout_prob = 0.0,
        exp_decay = None, method = 'gd'):
    """Creates a multi-layer fully connected neural network."""
    def create_model(weights, inputs, labels = None):
        hidden_units = inputs
        num_hidden_layers = len(weights) // 2 - 1
        regularisation_term = tf.zeros([1])

        for l in range(num_hidden_layers):
            cur_weights = weights[2*l]
            cur_biases = weights[2*l + 1]

            hidden_units = tf.nn.relu(tf.matmul(hidden_units, cur_weights) + cur_biases)
            if labels is not None:
                # If labels are specified, the graph will be used for training,
                # so we apply dropout.
                hidden_units = tf.nn.dropout(hidden_units, 1 - dropout_prob)

            regularisation_term = regularisation_term + tf.nn.l2_loss(cur_weights)

        # Output layer.
        cur_weights = weights[-2]
        cur_biases = weights[-1]
        out_logits = tf.matmul(hidden_units, cur_weights) + cur_biases
        out_prob = tf.nn.softmax(out_logits)
        regularisation_term = regularisation_term + tf.nn.l2_loss(cur_weights)

        if labels is not None:
            # Only when training.
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out_logits, labels))
            loss = loss + beta * regularisation_term
            return out_prob, loss

        return out_prob

    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, input_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        weights = create_fully_connected_weights(input_size, num_labels, num_hidden_nodes)

        # Training computation.
        train_prediction, loss = create_model(weights, tf_train_dataset, tf_train_labels)
        valid_prediction = create_model(weights, tf_valid_dataset)
        test_prediction = create_model(weights, tf_test_dataset)

        # Optimizer.
        global_step = tf.Variable(0)

        if exp_decay is not None:
            learning_rate = tf.train.exponential_decay(
                learning_rate, global_step,
                exp_decay['decay_steps'], exp_decay['decay_rate'], exp_decay['staircase'])

        optimizer = None
        if method == 'gd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
                loss, global_step=global_step)
        elif method == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(
                loss, global_step=global_step)
        else:
            raise Exception('Unknown optimiser.')

    tf_graph = {
        'graph': graph,
        'data_ph': tf_train_dataset,
        'labels_ph': tf_train_labels }
    tf_predictions = [train_prediction, valid_prediction, test_prediction]

    return tf_graph, optimizer, loss, tf_predictions

# Skip-gram.
def skipgram_model(vocabulary_size, embedding_size, batch_size, num_sampled, valid_examples,
    learning_rate):
    """Creates a model for learning a word embedding using skipgrams."""
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        tf_train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
        tf_train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

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

        similarity = None
        if valid_examples is not None:
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
            similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

    tf_graph = {
        'graph': graph,
        'data_ph': tf_train_dataset,
        'labels_ph': tf_train_labels }

    return tf_graph, optimizer, loss, normalized_embeddings, similarity

def cbow_model(vocabulary_size, embedding_size, context_length, batch_size,
    num_sampled, valid_examples, learning_rate):
    """Creates a model for learning a word embedding using CBOW."""
    input_batch_size = context_length * batch_size

    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        tf_train_dataset = tf.placeholder(tf.int32, shape=[input_batch_size])
        tf_train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # This sums the input embeddings in a batch of size input_batch_size,
        # by group of context_length. This results in an input vector with
        # batch_size rows.
        word_mean_op = tf.constant((1.0 / context_length) *
            np.kron(np.eye(batch_size), np.ones([1, context_length])), dtype=tf.float32)

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

        similarity = None
        if valid_examples is not None:
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
            similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

    tf_graph = {
        'graph': graph,
        'data_ph': tf_train_dataset,
        'labels_ph': tf_train_labels }

    return tf_graph, optimizer, loss, normalized_embeddings, similarity

# LSTM.
def lstm_model(input_size, output_size, embedding, num_nodes, num_unrollings, batch_size,
    learning_rate, exp_decay = None, gradient_max_value = 1.25, dropout_prob = 0.0):
    """Creates a LSTM model to learn symbol sequences."""

    graph = tf.Graph()
    with graph.as_default():
        # [ix, fx, cx, ox]
        x_mat = training.utils.gaussian_weights_variable([input_size, 4*num_nodes])
        # [im, fm, cm, om]
        o_mat = training.utils.gaussian_weights_variable([num_nodes, 4*num_nodes])
        # [ib, fb, cb, ob]
        b_vec = tf.Variable(tf.zeros([1, 4*num_nodes]))

        # Variables saving state across unrollings.
        saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        # Classifier weights and biases.
        w = training.utils.gaussian_weights_variable([num_nodes, output_size])
        b = tf.Variable(tf.zeros([output_size]))

        # Definition of the cell computation.
        def lstm_cell(i, o, state):
            """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
            Note that in this formulation, we omit the various connections between the
            previous state and the gates."""
            mult = tf.matmul(i, x_mat) + tf.matmul(o, o_mat) + b_vec

            input_gate = tf.sigmoid(mult[:, 0:num_nodes])
            forget_gate = tf.sigmoid(mult[:, num_nodes:2*num_nodes])
            state = forget_gate * state + input_gate * tf.tanh(mult[:, 2*num_nodes:3*num_nodes])
            output_gate = tf.sigmoid(mult[:, 3*num_nodes:4*num_nodes])
            return output_gate * tf.tanh(state), state

        # Input data.
        before_embedding_size = input_size
        if embedding is not None:
            before_embedding_size = embedding.shape[0]

        train_data = list()
        for _ in range(num_unrollings + 1):
            train_data.append(
                tf.placeholder(tf.float32, shape=[batch_size, before_embedding_size]))
        train_inputs = train_data[:num_unrollings]
        train_labels = train_data[1:]  # Labels are inputs shifted by one time step.

        # Unrolled LSTM loop.
        outputs = list()
        output = saved_output
        state = saved_state
        for i in train_inputs:
            if embedding is not None:
                # Converting the input to the embedding.
                indices = tf.argmax(i, 1)
                i = tf.nn.embedding_lookup(embedding, indices)
            # Dropout is only applied to inputs, not to recurrent connections.
            i = tf.nn.dropout(i, 1 - dropout_prob)
            output, state = lstm_cell(i, output, state)
            outputs.append(output)

        # State saving across unrollings.
        with tf.control_dependencies([saved_output.assign(output),
                                      saved_state.assign(state)]):
            # Classifier.
            # Dropout is also applied to the output of the LSTM cell, only when
            # used for the projection, as it is not recurrent.
            outputs = tf.concat(0, outputs)
            outputs = tf.nn.dropout(outputs, 1 - dropout_prob)
            logits = tf.nn.xw_plus_b(outputs, w, b)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits, tf.concat(0, train_labels)))

        # Optimizer.
        global_step = tf.Variable(0)

        if exp_decay is not None:
            learning_rate = tf.train.exponential_decay(
                learning_rate, global_step,
                exp_decay['decay_steps'], exp_decay['decay_rate'], exp_decay['staircase'])

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # Clipping to avoid exploding gradient.
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, gradient_max_value)
        optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

        # Predictions.
        train_prediction = tf.nn.softmax(logits)

        # Sampling and validation eval: batch 1, no unrolling.
        sample_input_ph = tf.placeholder(tf.float32, shape=[1, before_embedding_size])
        saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
        saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
        reset_sample_state = tf.group(
            saved_sample_output.assign(tf.zeros([1, num_nodes])),
            saved_sample_state.assign(tf.zeros([1, num_nodes])))

        sample_input = sample_input_ph
        if embedding is not None:
            indices = tf.argmax(sample_input_ph, 1)
            sample_input = tf.nn.embedding_lookup(embedding, indices)

        sample_output, sample_state = lstm_cell(
            sample_input, saved_sample_output, saved_sample_state)
        with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                      saved_sample_state.assign(sample_state)]):
            sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

    tf_graph = {
        'graph': graph,
        'data_ph': train_data,
        'sample_ph': sample_input_ph }
    tf_predictions = [train_prediction, sample_prediction]

    return tf_graph, optimizer, loss, tf_predictions, reset_sample_state

# Reverse word RNN.
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

        # Decoder has one extra cell because it starts with the GO symbol,
        # and the targets are shifted by one.
        # Not sure this is actually useful, as it is always set to 0.
        # As this is inspired by TensorFlow seq2seq models, there might be
        # something dodgy in there.
        self.decoder_inputs.append(tf.placeholder(
            tf.float32, shape=(batch_size, self.vocab_size)))
        self.target_weights.append(np.ones((batch_size,)))

        # Targets used by the sequence loss must be integer indices.
        targets = [tf.cast(tf.argmax(i, 1), dtype=tf.int32)
            for i in self.decoder_inputs[1:]]

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
        # If the last target is always 0 (PAD), and its weight is always 0,
        # what's the point of having it?
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
            encoder_pad = [dataset.utils.PAD_ID] * (self.sequence_length - len(encoder_input))
            encoder_inputs.append(list(encoder_pad + encoder_input))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            # This is weird as well. If the decoder input length is exactly
            # self.sequence_length, then we end up with a sequence of length
            # self.sequence_length + 1. The last symbol gets ignored in the
            # rest of the batch creation.
            decoder_pad_size = self.sequence_length - len(decoder_input) - 1
            decoder_inputs.append([dataset.utils.GO_ID] + decoder_input +
                                 [dataset.utils.PAD_ID] * decoder_pad_size)

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
                    or target == dataset.utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
