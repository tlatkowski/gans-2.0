import tensorflow as tf
from tensorflow.python.keras import datasets


def load_data(input_params):
    fashion_mnist = datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # return (train_images, train_labels), (test_images, test_labels)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(
        input_params.buffer_size).batch(
        input_params.batch_size)
    return train_dataset
