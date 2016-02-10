# Provides a set function that run the iterative optimisation
# of various models.
# Derived from Tensorflow's Udacity tutorial.
#
# Flo Vouin - 2016

import numpy as np
import tensorflow as tf

from dataset.utils import accuracy
from dataset import utils
import training.utils

def run(tf_graph, optimizer, loss, tf_predictions,
    train_dataset, train_labels, valid_labels, test_labels,
    num_steps, batch_size, verbose_frequency = 500):
    """Runs the optimisation of a regular forward neural network."""
    with tf.Session(graph=tf_graph['graph']) as session:
        tf.initialize_all_variables().run()

        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

            # Generates a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size)]
            batch_labels = train_labels[offset:(offset + batch_size)]

            # Runs one step of the optimisation.
            _, l, train_predictions = session.run([optimizer, loss, tf_predictions[0]],
                feed_dict={
                    tf_graph['data_ph'] : batch_data,
                    tf_graph['labels_ph'] : batch_labels})

            if (step % verbose_frequency == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(train_predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    tf_predictions[1].eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(tf_predictions[2].eval(), test_labels))

def run_embedding(tf_graph, optimizer, loss, similarity, normalized_embeddings,
    batches, valid_examples, reverse_dictionary, num_steps,
    verbose_loss_frequency = 2000, verbose_frequency = 10000):
    """Runs the optimisation of a word embedding model."""
    valid_verbose = (similarity is not None) and (valid_examples is not None) \
        and (reverse_dictionary is not None)

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

            # Printing similar words for some of the words in
            # the validation set.
            if valid_verbose and (step % verbose_frequency == 0):
                sim = similarity.eval()
                for i in range(len(valid_examples)):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # Number of nearest neighbors.
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = "Nearest to %s:" % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log = "%s %s," % (log, close_word)
                    print(log)
        return normalized_embeddings.eval()

def run_lstm(tf_graph, optimizer, loss, reset_sample_state, tf_predictions,
    train_batches, valid_batches, valid_size, num_steps, char_func, summary_frequency):
    """Runs the optimisation of a RNN LSTM model."""
    with tf.Session(graph=tf_graph['graph']) as session:
        tf.initialize_all_variables().run()

        mean_loss = 0
        for step in range(num_steps):
            batches = train_batches.next()
            input_size = batches[0].shape[1]
            feed_dict = dict()
            for i in range(len(tf_graph['data_ph'])):
                feed_dict[tf_graph['data_ph'][i]] = batches[i]

            _, l, predictions = session.run([optimizer, loss, tf_predictions[0]], feed_dict=feed_dict)
            mean_loss += l

            if step % summary_frequency == 0:
                if step > 0:
                    mean_loss = mean_loss / summary_frequency
                # The mean loss is an estimate of the loss over the last few batches.
                print('Average loss at step', step, ':', mean_loss)
                mean_loss = 0

                labels = np.concatenate(list(batches)[1:])
                print('Minibatch perplexity: %.2f'
                    % float(np.exp(training.utils.logprob(predictions, labels))))

                if step % (summary_frequency * 10) == 0:
                    # Generate some samples.
                    print('=' * 80)
                    for _ in range(5):
                        feed = training.utils.sample(
                            training.utils.random_distribution(input_size))
                        sentence = char_func(feed)[0]
                        reset_sample_state.run()
                        for _ in range(79):
                            prediction = tf_predictions[1].eval({tf_graph['sample_ph']: feed})
                            feed = training.utils.sample(prediction)
                            sentence += char_func(feed)[0]
                        print(sentence)
                    print('=' * 80)

                # Measure validation set perplexity.
                reset_sample_state.run()
                valid_logprob = 0
                for _ in range(valid_size):
                    b = valid_batches.next()
                    predictions = tf_predictions[1].eval({tf_graph['sample_ph']: b[0]})
                    valid_logprob = valid_logprob + training.utils.logprob(predictions, b[1])
                print('Validation set perplexity: %.2f' %
                    float(np.exp(valid_logprob / valid_size)))
