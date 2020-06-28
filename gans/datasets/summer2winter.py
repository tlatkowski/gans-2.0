from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds

from gans.datasets import abstract_dataset

TFDS_SUMMER2WINTER_PATH = 'cycle_gan/summer2winter_yosemite'


class SummerToWinterDataset(abstract_dataset.Dataset):
    
    def __init__(
            self,
            model_parameters,
            with_labels=False,
    ):
        self.img_height = model_parameters.img_height
        self.img_width = model_parameters.img_width
        super().__init__(model_parameters, with_labels)
    
    def __call__(self, *args, **kwargs):
        return self.train_dataset
    
    def load_data(self):
        dataset, metadata = tfds.load(
            TFDS_SUMMER2WINTER_PATH,
            with_info=True,
            as_supervised=True,
        )
        
        train_summer, train_winter = dataset['trainA'], dataset['trainB']
        
        train_summer = train_summer.map(
            partial(
                preprocess_image,
                img_height=self.img_height,
                img_width=self.img_width,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        ).repeat(
        ).shuffle(
            self.buffer_size,
        ).batch(
            self.batch_size,
        )
        
        train_winter = train_winter.map(
            partial(
                preprocess_image,
                img_height=self.img_height,
                img_width=self.img_width,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        ).repeat(
        ).shuffle(
            self.buffer_size,
        ).batch(
            self.batch_size,
        )
        
        return zip(train_summer, train_winter)
    
    def load_data_with_labels(self):
        raise NotImplementedError


def preprocess_image(image, label, img_height, img_width):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(
        images=image,
        size=(img_height, img_width),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )
    return image
