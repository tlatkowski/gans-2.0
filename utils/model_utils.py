import enum

from easydict import EasyDict as edict

from models import conditional_gan
from models import generators
from models import vanilla_gan
from utils.dataset_utils import ProblemType


class ModelType(enum.Enum):
    VANILLA_GAN = 0,
    CONDITIONAL_GAN = 1,
    WASSERSTEIN_GAN = 2


def model_type_values():
    return [i.name for i in ModelType]


def model_factory(input_params: edict, input_args):
    if input_args.gan_type == ModelType.VANILLA_GAN.name:
        return vanilla_gan.VanillaGAN(input_params, input_args)
    elif input_args.gan_type == ModelType.CONDITIONAL_GAN.name:
        return conditional_gan.ConditionalGAN(input_params, input_args)
    elif input_args.gan_type == ModelType.WASSERSTEIN_GAN.name:
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
    elif dataset_type == ProblemType.CONDITIONAL_MNIST.name:
        return generators.RandomToImageConditionalGenerator(input_params)
    elif dataset_type == ProblemType.CONDITIONAL_FASHION_MNIST.name:
        return generators.RandomToImageConditionalGenerator(input_params)
    elif dataset_type == ProblemType.CONDITIONAL_CIFAR10.name:
        return generators.RandomToImageConditionalGenerator(input_params)
    else:
        raise NotImplementedError
