import numpy as np
import tensorflow as tf
from dataset.utils import accuracy
from dataset import utils
import training.utils

def run(tf_graph, optimizer, loss, tf_predictions,
    train_dataset, train_labels, valid_labels, test_labels,
    num_steps, batch_size, verbose_frequency = 500):
    """"""
    with tf.Session(graph=tf_graph['graph']) as session:
        tf.initialize_all_variables().run()

        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size)]
            batch_labels = train_labels[offset:(offset + batch_size)]

            # Running one step of the optimisation.
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

            if valid_verbose and (step % verbose_frequency == 0):
                sim = similarity.eval()
                for i in range(len(valid_examples)):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8 # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = "Nearest to %s:" % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log = "%s %s," % (log, close_word)
                    print(log)
        return normalized_embeddings.eval()
