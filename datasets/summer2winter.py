import tensorflow as tf
import tensorflow_datasets as tfds


# from tensorflow.image import ResizeMethod


# class SummerToWinterDataset(abstract_dataset.Dataset):
class SummerToWinterDataset():
    
    def __init__(
            self,
            # input_params,
            with_labels=False,
    ):
        # self.input_params = input_params
        self.buffer_size = 10
        self.batch_size = 4
        # super(SummerToWinterDataset, self).__init__(input_params, with_labels)
    
    def __call__(self, *args, **kwargs):
        # return self.train_dataset
        return self.load_data()
    
    def load_data(self):
        dataset, metadata = tfds.load(
            'cycle_gan/summer2winter_yosemite',
            with_info=True,
            as_supervised=True,
        )
        
        train_summer, train_winter = dataset['trainA'], dataset['trainB']
        test_summer, test_winter = dataset['testA'], dataset['testB']
        
        train_summer = train_summer.map(
            preprocess_image_train,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
            self.buffer_size).batch(self.batch_size)
        
        train_winter = train_winter.map(
            preprocess_image_train,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
            self.buffer_size).batch(self.batch_size)
        
        return zip(train_summer, train_winter)
    
    def load_data_with_labels(self):
        raise NotImplementedError


def preprocess_image_train(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image
