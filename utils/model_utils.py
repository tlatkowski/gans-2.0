import enum

from easydict import EasyDict as edict

from models import conditional_gan
from models import generators
from models import vanilla_gan
from utils.dataset_utils import ProblemType


class ModelType(enum.Enum):
    VANILLA = 0,
    CONDITIONAL = 1,
    WASSERSTEIN = 2


def model_type_values():
    return [i.name for i in ModelType]


def model_factory(input_params: edict, gan_type, input_args):
    if gan_type == ModelType.VANILLA.name:
        return vanilla_gan.VanillaGAN(input_params, input_args)
    elif gan_type == ModelType.CONDITIONAL.name:
        return conditional_gan.ConditionalGAN(input_params, input_args)
    elif gan_type == ModelType.WASSERSTEIN.name:
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
        return generators.RandomToImageCifar10CConditionalGenerator(input_params)
    else:
        raise NotImplementedError
