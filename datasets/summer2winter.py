import os
import re

import tensorflow as tf
from tensorflow.image import ResizeMethod
from tensorflow.python.keras import datasets

from utils import data_utils

SUMMER_IMAGES_PATH = '../data/summer2winter_yosemite/trainA'
WINTER_IMAGES_PATH = '../data/summer2winter_yosemite/trainB'


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
        summer_dataset = create_tf_dataset(
            SUMMER_IMAGES_PATH,
            self.batch_size,
            self.buffer_size,
        )
        winter_dataset = create_tf_dataset(
            WINTER_IMAGES_PATH,
            self.batch_size,
            self.buffer_size,
        )
        return zip(summer_dataset, winter_dataset)
    
    def load_data_with_labels(self):
        cifar10 = datasets.cifar10
        (train_images, train_labels), (_, _) = cifar10.load_data()
        train_images = train_images.astype('float32')
        train_images = data_utils.normalize_inputs(train_images)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
            self.buffer_size).batch(
            self.batch_size)
        return train_dataset


def create_tf_dataset(path_to_images, batch_size, buffer_size):
    summer_files_paths = find_all_files_in_dir(path_to_images)
    image_paths = tf.data.Dataset.from_tensor_slices(summer_files_paths)
    load_images_fn = lambda file_path: load_image_from_path(file_path, 256, 256)
    dataset = image_paths.map(load_images_fn)
    dataset = dataset.shuffle(
        buffer_size
    ).batch(
        batch_size
    )
    return dataset


def find_all_files_in_dir(root_dir, pattern='/*.jpg'):
    """
    Finds all files matching provided pattern in dir and subdirs.

    :param root_dir: the root directory
    :param pattern: the pattern for file names matching
    :return: the list of full file paths that matching patterns
    """
    file_paths = []
    regex = re.compile(pattern)
    for dir_name, subdir_list, file_list in os.walk(root_dir):
        if file_list:
            file_list = list(filter(regex.search, file_list))
            file_list = [os.path.join(dir_name, f) for f in file_list]
            file_paths += file_list
    return file_paths


def load_image_from_path(
        file_path,
        img_width,
        img_height,
        png=False,
):
    """
    Reads image from file and resizes it to provided width and height.

    :param file_path: path to image file
    :param img_width: desired image width
    :param img_height: desired image height
    :return: tensor containing image
    """
    img = read_img(file_path)
    if png:
        img = decode_png(img)
    else:
        img = decode_img(img)
    
    img = resize_img(img, img_width, img_height)
    return img


def read_img(file_path):
    return tf.io.read_file(file_path)


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = (img - 0.5) * 2
    return img


def decode_png(img):
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def resize_img(img, img_width, img_height):
    return tf.image.resize(
        images=img,
        size=(img_width, img_height),
        method=ResizeMethod.NEAREST_NEIGHBOR,
    )


a = SummerToWinterDataset()
a.load_data()
