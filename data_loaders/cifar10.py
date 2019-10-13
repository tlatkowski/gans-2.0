import tensorflow as tf
from tensorflow.python.keras import datasets


def normalize_inputs(data):
    """
    Normalizes the inputs to [-1, 1]
    
    :param data:
    :return:
    """
    data = (data - 127.5) / 127.5  #
    return data


def load_data(input_params):
    cifar10 = datasets.cifar10
    (train_images, train_labels), (_, _) = cifar10.load_data()
    train_images = train_images.astype('float32')
    train_images = normalize_inputs(train_images)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(
        input_params.buffer_size).batch(
        input_params.batch_size)
    return train_dataset


def load_data_with_labels(input_params):
    cifar10 = datasets.cifar10
    (train_images, train_labels), (_, _) = cifar10.load_data()
    train_images = train_images.astype('float32')
    train_images = normalize_inputs(train_images)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
        input_params.buffer_size).batch(
        input_params.batch_size)
    return train_dataset
