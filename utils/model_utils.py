import enum

from easydict import EasyDict as edict

from models import conditional_gan
from models import vanilla_gan
from models.discriminators import basic_conditional_discriminator
from models.discriminators import basic_discriminator
from models.discriminators import cifar10_conditional_discriminator
from models.generators import conditional_random_to_image
from models.generators import conditional_random_to_image_cifar10
from models.generators import random_to_image
from models.generators import random_to_image_cifar10
from utils.dataset_utils import ProblemType


class ModelType(enum.Enum):
    VANILLA = 0,
    CONDITIONAL = 1,
    WASSERSTEIN = 2


def model_type_values():
    return [i.name for i in ModelType]


def gan_model_factory(input_params: edict, gan_type, input_args):
    generator = generator_model_factory(input_params, input_args.problem_type)
    discriminator = discriminator_model_factory(input_params, input_args.problem_type)
    
    if gan_type == ModelType.VANILLA.name:
        return vanilla_gan.VanillaGAN(input_params, generator, discriminator,
                                      input_args.problem_type, input_args.continue_training)
    elif gan_type == ModelType.CONDITIONAL.name:
        return conditional_gan.ConditionalGAN(input_params, generator, discriminator,
                                              input_args.problem_type, input_args.continue_training)
    elif gan_type == ModelType.WASSERSTEIN.name:
        raise NotImplementedError
    else:
        raise NotImplementedError


def generator_model_factory(input_params, dataset_type: ProblemType):
    if dataset_type == ProblemType.VANILLA_MNIST.name:
        return random_to_image.RandomToImageGenerator(input_params)
    if dataset_type == ProblemType.VANILLA_FASHION_MNIST.name:
        return random_to_image.RandomToImageGenerator(input_params)
    elif dataset_type == ProblemType.VANILLA_CIFAR10.name:
        # return generators.RandomToImageCifar10Generator(input_params)
        return random_to_image_cifar10.RandomToImageCifar10Generator(input_params)
    elif dataset_type == ProblemType.CONDITIONAL_MNIST.name:
        return conditional_random_to_image.RandomToImageConditionalGenerator(input_params)
    elif dataset_type == ProblemType.CONDITIONAL_FASHION_MNIST.name:
        return conditional_random_to_image.RandomToImageConditionalGenerator(input_params)
    elif dataset_type == ProblemType.CONDITIONAL_CIFAR10.name:
        return conditional_random_to_image_cifar10.RandomToImageCifar10CConditionalGenerator(
            input_params)
    else:
        raise NotImplementedError


def discriminator_model_factory(input_params, dataset_type: ProblemType):
    if dataset_type == ProblemType.VANILLA_MNIST.name:
        return basic_discriminator.Discriminator(input_params)
    if dataset_type == ProblemType.VANILLA_FASHION_MNIST.name:
        return basic_discriminator.Discriminator(input_params)
    elif dataset_type == ProblemType.VANILLA_CIFAR10.name:
        return basic_discriminator.Discriminator(input_params)
    elif dataset_type == ProblemType.CONDITIONAL_MNIST.name:
        return basic_conditional_discriminator.ConditionalDiscriminator(input_params)
    elif dataset_type == ProblemType.CONDITIONAL_FASHION_MNIST.name:
        return basic_conditional_discriminator.ConditionalDiscriminator(input_params)
    elif dataset_type == ProblemType.CONDITIONAL_CIFAR10.name:
        return cifar10_conditional_discriminator.ConditionalDiscriminatorCifar10(input_params)
    else:
        raise NotImplementedError
