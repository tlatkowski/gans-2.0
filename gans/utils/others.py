import numpy as np
import tensorflow as tf


def create_test_labels(batch_size):
    labels = [[i] * 10 for i in list(range(10))]
    test_labels = [tf.random.normal([batch_size, 100]),
                   np.array(labels)]
    return test_labels
