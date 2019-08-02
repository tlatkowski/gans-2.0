import tensorflow as tf
from easydict import EasyDict as edict


def load_data(input_params: edict):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # train_labels = tf.one_hot(indices=train_labels, depth=10)
    # train_labels = tf.one_hot(indices=train_labels, depth=10)
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
        input_params.buffer_size).batch(
        input_params.batch_size)
    return train_dataset
