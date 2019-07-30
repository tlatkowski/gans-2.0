import enum
import os

import matplotlib.pyplot as plt
from IPython import display
from easydict import EasyDict as edict

from data_loaders import cifar10
from data_loaders import fashion_mnist
from data_loaders import mnist
from models import vanilla_gan

SAVE_IMAGE_DIR = "./outputs"


class ModelType(enum.Enum):
    VANILLA_GAN = 0,
    WASSERSTEIN_GAN = 1


def model_type_values():
    return [i.name for i in ModelType]


class DatasetType(enum.Enum):
    MNIST = 0,
    FASHION_MNIST = 1
    CIFAR10 = 2
    COCO = 3


def dataset_type_values():
    return [i.name for i in DatasetType]


def dataset_factory(input_params, dataset_type: DatasetType):
    if dataset_type == DatasetType.MNIST.name:
        return mnist.load_data(input_params)
    if dataset_type == DatasetType.FASHION_MNIST.name:
        return fashion_mnist.load_data(input_params)
    elif dataset_type == DatasetType.CIFAR10.name:
        return cifar10.load_data()
    else:
        raise NotImplementedError


def model_factory(input_params: edict, model_type: ModelType, dataset_type):
    if model_type == ModelType.VANILLA_GAN.name:
        return vanilla_gan.Random2ImageGAN(input_params, dataset_type)
    if model_type == ModelType.WASSERSTEIN_GAN.name:
        raise NotImplementedError
    else:
        raise NotImplementedError


def generate_and_save_images(generator_model, epoch, test_input, dataset_name,
                             num_examples_to_display=16):
    display.clear_output(wait=True)
    predictions = generator_model(test_input, training=False)
    if predictions.shape[0] < num_examples_to_display:
        raise ValueError("Input batch size cannot be less than number of example to display.")
    
    for i in range(num_examples_to_display):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    save_path = os.path.join(SAVE_IMAGE_DIR, dataset_name)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'image_at_epoch_{:04d}.png'.format(epoch)))


def plot_image_grid(generated_image):
    for i in range(generated_image.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_image[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.show()
