import enum

from easydict import EasyDict as edict

from data_loaders import cifar10
from data_loaders import fashion_mnist
from data_loaders import mnist
from models import conditional_gan
from models import generators
from models import vanilla_gan


class ModelType(enum.Enum):
    VANILLA_GAN = 0,
    CONDITIONAL_GAN = 1,
    WASSERSTEIN_GAN = 2


def model_type_values():
    return [i.name for i in ModelType]


class ProblemType(enum.Enum):
    VANILLA_MNIST = 0,
    VANILLA_FASHION_MNIST = 1
    VANILLA_CIFAR10 = 2
    CONDITIONAL_MNIST = 3
    CONDITIONAL_FASHION_MNIST = 3
    CONDITIONAL_CIFAR10 = 3


def dataset_type_values():
    return [i.name for i in ProblemType]


def dataset_factory(input_params, dataset_type: ProblemType):
    if dataset_type == ProblemType.VANILLA_MNIST.name:
        return mnist.load_data(input_params)
    elif dataset_type == ProblemType.VANILLA_FASHION_MNIST.name:
        return fashion_mnist.load_data(input_params)
    elif dataset_type == ProblemType.VANILLA_CIFAR10.name:
        return cifar10.load_data(input_params)
    elif dataset_type == ProblemType.CONDITIONAL_MNIST.name:
        return mnist.load_data_with_labels(input_params)
    else:
        raise NotImplementedError


def model_factory(input_params: edict, model_type: ModelType, dataset_type: ProblemType):
    if model_type == ModelType.VANILLA_GAN.name:
        return vanilla_gan.VanillaGAN(input_params, dataset_type)
    elif model_type == ModelType.CONDITIONAL_GAN.name:
        return conditional_gan.ConditionalGAN(input_params, dataset_type)
    elif model_type == ModelType.WASSERSTEIN_GAN.name:
        raise NotImplementedError
    else:
        raise NotImplementedError


def generator_model_factory(input_params, dataset_type: ProblemType):
    if dataset_type == ProblemType.VANILLA_MNIST.name:
        return generators.RandomToImageGenerator(input_params)
    if dataset_type == ProblemType.VANILLA_FASHION_MNIST.name:
        return generators.RandomToImageGenerator(input_params)
    elif dataset_type == ProblemType.VANILLA_CIFAR10.name:
        # return generators.RandomToImageCifar10Generator(input_params)
        return generators.RandomToImageCifar10NearestNeighborUpSamplingGenerator(input_params)
    else:
        raise NotImplementedError
