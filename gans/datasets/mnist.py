import tensorflow as tf

from gans.datasets import abstract_dataset
from gans.utils import data_utils


class MnistDataset(abstract_dataset.Dataset):
    
    def __init__(
            self,
            model_parameters,
            with_labels=False,
    ):
        super().__init__(model_parameters, with_labels)
    
    def __call__(self, *args, **kwargs):
        return self.train_dataset
    
    def load_data(self):
        (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = data_utils.normalize_inputs(train_images)
        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(
            self.buffer_size).batch(
            self.batch_size)
        return train_dataset
    
    def load_data_with_labels(self):
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = data_utils.normalize_inputs(train_images)
        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
            self.buffer_size).batch(
            self.batch_size)
        return train_dataset
