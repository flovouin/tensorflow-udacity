import tensorflow as tf
import training.utils

def create_fully_connected_weights(input_size, num_labels, num_hidden_nodes):
    num_layers = len(num_hidden_nodes)
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
    """"""
    def create_model(weights, inputs, labels = None):
        hidden_units = inputs
        num_hidden_layers = len(weights) // 2 - 1
        regularisation_term = tf.zeros([1])

        for l in range(num_hidden_layers):
            cur_weights = weights[2*l]
            cur_biases = weights[2*l + 1]

            hidden_units = tf.nn.relu(tf.matmul(hidden_units, cur_weights) + cur_biases)
            if labels is not None:
                hidden_units = tf.nn.dropout(hidden_units, 1 - dropout_prob)

            regularisation_term = regularisation_term + tf.nn.l2_loss(cur_weights)

        # Output layer.
        cur_weights = weights[-2]
        cur_biases = weights[-1]
        out_logits = tf.matmul(hidden_units, cur_weights) + cur_biases
        out_prob = tf.nn.softmax(out_logits)
        regularisation_term = regularisation_term + tf.nn.l2_loss(cur_weights)

        if labels is not None:
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
