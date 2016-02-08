import tensorflow as tf
from dataset.utils import accuracy

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
