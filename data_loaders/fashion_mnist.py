import tensorflow as tf
from tensorflow.python.keras import datasets

from utils import data_utils


def load_data(input_params):
    fashion_mnist = datasets.fashion_mnist
    (train_images, _), (_, _) = fashion_mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = data_utils.normalize_inputs(train_images)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(
        input_params.buffer_size).batch(
        input_params.batch_size)
    return train_dataset


def load_data_with_labels(input_params):
    fashion_mnist = datasets.fashion_mnist
    (train_images, train_labels), (_, _) = fashion_mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = data_utils.normalize_inputs(train_images)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
        input_params.buffer_size).batch(
        input_params.batch_size)
    return train_dataset
