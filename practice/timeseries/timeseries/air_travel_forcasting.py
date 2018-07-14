import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
import matplotlib.pyplot as plt


def get_lstm_estimator(num_of_lstm):
    my_feature_columns = [tf.feature_column.numeric_column(key='passengers')]
    regressor = tf.estimator.Estimator(model_fn=lstm_model_fn,
                                       params={
                                           'feature_columns': my_feature_columns,
                                           'num_of_lstm': num_of_lstm
                                       })
    return regressor


def lstm_model_fn(
        features,
        labels,
        mode,
        params):

    lstm = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.BasicLSTMCell(30) for _ in range(params['num_of_lstm'])])

    # convert from [batch size, time step, number of input] to list of [batch size, number of input].
    # size of list is time step
    features = tf.unstack(features, axis=1)
    outputs, states = rnn.static_rnn(lstm, features, dtype=tf.float32)
    final_from_lstm = outputs[-1]

    # To regress a value, we only need 1 output node
    final_output = tf.layers.dense(final_from_lstm, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=final_output)

    labels = tf.reshape(labels, shape=[-1, 1])
    loss = tf.losses.mean_squared_error(labels=labels, predictions=final_output)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, tf.train.get_global_step())
    tf.logging.set_verbosity(tf.logging.INFO)
    logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=5)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])


def input_fn(series, time_steps, epoch, batch_size):
    features, labels = series
    # convert to [batch, timesteps, number of inputs]
    features = tf.reshape(features, shape=[-1, time_steps, 1])
    feature_elements, label_element = tf.data.Dataset.from_tensor_slices((features, labels)).repeat(epoch).batch(batch_size).make_one_shot_iterator().get_next()
    return feature_elements, label_element


def load_data(path, train_ratio, time_steps):
    passengers = pd.read_csv(path)['passengers']
    scaled = (passengers - passengers.min()) / passengers.max()
    last = len(passengers) - 1

    labels = [float(scaled[i]) for i in range(time_steps, last + 1)]
    features = [[float(scaled[j]) for j in range(i - time_steps, i)] for i in range(time_steps, last + 1)]

    size = last + 1 - time_steps
    train_size = int(size * train_ratio)
    train_set = (features[:train_size], labels[:train_size])
    test_set = (features[train_size:], labels[train_size:])
    return train_set, test_set


def examine_result(labels, predictions):
    indices = [i for i in range(0, len(labels))]
    plt.plot(indices, labels, label='ground truth')
    plt.plot(indices, predictions, label='predictions')
    plt.legend(loc='upper center', shadow=True)
    plt.show()


def run():
    path = "/Users/stevenchen/Downloads/international-airline-passengers.csv"
    time_step = 20
    train, test = load_data(path, 0.7, time_step)
    estimator = get_lstm_estimator(num_of_lstm=2)
    estimator.train(input_fn=lambda: input_fn(series=train, time_steps=time_step, epoch=1000, batch_size=5))
    r = estimator.evaluate(input_fn=lambda: input_fn(train, time_steps=time_step, epoch=1, batch_size=5))
    print(r)

    predictions = estimator.predict(input_fn=lambda: input_fn(test, time_steps=time_step, epoch=1, batch_size=5))
    # predictions is a list of list, flatten it to a list
    prediction_data_list = [p[0] for p in predictions]

    labels_in_test = test[1]
    examine_result(labels_in_test, prediction_data_list)


if __name__ == "__main__":
    run()