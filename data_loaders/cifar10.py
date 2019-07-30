import tensorflow as tf
from tensorflow.python.keras import datasets


def load_data(input_params):
    cifar10 = datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(
        input_params.buffer_size).batch(
        input_params.batch_size)
    # return (train_images, train_labels), (test_images, test_labels)
    return (train_images, train_labels), (test_images, test_labels)
