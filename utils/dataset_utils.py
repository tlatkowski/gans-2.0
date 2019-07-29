import enum
import os

import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from models import vanilla_gan


from data_loaders import mnist
from data_loaders import fashion_mnist
from data_loaders import cifar10


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


def model_factory(input_params: edict, model_type: ModelType):
    if model_type == ModelType.VANILLA_GAN.name:
        return vanilla_gan.Random2ImageGAN(input_params)
    if model_type == ModelType.WASSERSTEIN_GAN.name:
        raise NotImplementedError
    else:
        raise NotImplementedError


def generate_and_save_images(generator_model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = generator_model(test_input, training=False)
    
    # fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)
    plt.savefig(os.path.join(SAVE_IMAGE_DIR, 'image_at_epoch_{:04d}.png'.format(epoch)))


def plot_image_grid(generated_image):
    for i in range(generated_image.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_image[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.show()
