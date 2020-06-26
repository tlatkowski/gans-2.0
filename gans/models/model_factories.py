import enum

from easydict import EasyDict as edict

from gans.datasets import problem_type as pt
from gans.models.discriminators import basic_conditional_discriminator
from gans.models.discriminators import basic_discriminator
from gans.models.discriminators import cifar10_conditional_discriminator
from gans.models.discriminators import patch_discriminator
from gans.models.gans import conditional_gan
from gans.models.gans import cycle_gan
from gans.models.gans import vanilla_gan
from gans.models.generators.image_to_image import u_net
from gans.models.generators.latent_to_image import conditional_random_to_image
from gans.models.generators.latent_to_image import conditional_random_to_image_cifar10
from gans.models.generators.latent_to_image import random_to_image
from gans.models.generators.latent_to_image import random_to_image_cifar10
from gans.trainers import vanilla_gan_trainer


class ModelType(enum.Enum):
    VANILLA = 0,
    CONDITIONAL = 1,
    WASSERSTEIN = 2,
    CYCLE = 3


def model_type_values():
    return [i.name for i in ModelType]


def gan_model_factory(
        input_params: edict,
        gan_type,
        input_args,
):
    generator = generator_model_factory(input_params, input_args.problem_type)
    discriminator = discriminator_model_factory(input_params, input_args.problem_type)

    if gan_type == ModelType.VANILLA.name:
        gan_trainer = vanilla_gan_trainer.VanillaGANTrainer(
            batch_size=input_params.batch_size,
            generator=generator,
            discriminator=discriminator,
            dataset_type=input_args.problem_type,
            learning_rate_generator=input_params.learning_rate_generator,
            learning_rate_discriminator=input_params.learning_rate_discriminator,
            continue_training=input_args.continue_training,
            save_images_every_n_steps=input_params.save_images_every_n_steps,
        )
        return vanilla_gan.VanillaGAN(
            model_parameters=input_params,
            generator=generator,
            discriminator=discriminator,
            gan_trainer=gan_trainer,
        )
    elif gan_type == ModelType.CONDITIONAL.name:
        return conditional_gan.ConditionalGAN(
            model_parameters=input_params,
            generator=generator,
            discriminator=discriminator,
            problem_type=input_args.problem_type,
            continue_training=input_args.continue_training,
        )
    elif gan_type == ModelType.CYCLE.name:
        return cycle_gan.CycleGAN(
            input_params=input_params,
            generators=generator,
            discriminators=discriminator,
            problem_type=input_args.problem_type,
            continue_training=input_args.continue_training,
        )
    elif gan_type == ModelType.WASSERSTEIN.name:
        raise NotImplementedError
    else:
        raise NotImplementedError


def generator_model_factory(
        input_params,
        problem_type: pt.ProblemType,
):
    if problem_type == pt.ProblemType.VANILLA_MNIST.name:
        return random_to_image.RandomToImageGenerator(input_params)
    if problem_type == pt.ProblemType.VANILLA_FASHION_MNIST.name:
        return random_to_image.RandomToImageGenerator(input_params)
    elif problem_type == pt.ProblemType.VANILLA_CIFAR10.name:
        # return generators.RandomToImageCifar10Generator(input_params)
        return random_to_image_cifar10.RandomToImageCifar10Generator(input_params)
    elif problem_type == pt.ProblemType.CONDITIONAL_MNIST.name:
        return conditional_random_to_image.RandomToImageConditionalGenerator(input_params)
    elif problem_type == pt.ProblemType.CONDITIONAL_FASHION_MNIST.name:
        return conditional_random_to_image.RandomToImageConditionalGenerator(input_params)
    elif problem_type == pt.ProblemType.CONDITIONAL_CIFAR10.name:
        return conditional_random_to_image_cifar10.RandomToImageCifar10CConditionalGenerator(
            input_params)
    elif problem_type == pt.ProblemType.CYCLE_SUMMER2WINTER.name:
        return [u_net.UNetGenerator(input_params), u_net.UNetGenerator(input_params)]
    else:
        raise NotImplementedError


def discriminator_model_factory(
        input_params,
        dataset_type: pt.ProblemType,
):
    if dataset_type == pt.ProblemType.VANILLA_MNIST.name:
        return basic_discriminator.Discriminator(input_params)
    if dataset_type == pt.ProblemType.VANILLA_FASHION_MNIST.name:
        return basic_discriminator.Discriminator(input_params)
    elif dataset_type == pt.ProblemType.VANILLA_CIFAR10.name:
        return basic_discriminator.Discriminator(input_params)
    elif dataset_type == pt.ProblemType.CONDITIONAL_MNIST.name:
        return basic_conditional_discriminator.ConditionalDiscriminator(input_params)
    elif dataset_type == pt.ProblemType.CONDITIONAL_FASHION_MNIST.name:
        return basic_conditional_discriminator.ConditionalDiscriminator(input_params)
    elif dataset_type == pt.ProblemType.CONDITIONAL_CIFAR10.name:
        return cifar10_conditional_discriminator.ConditionalDiscriminatorCifar10(input_params)
    elif dataset_type == pt.ProblemType.CYCLE_SUMMER2WINTER.name:
        return [patch_discriminator.PatchDiscriminator(input_params),
                patch_discriminator.PatchDiscriminator(input_params)]
    else:
        raise NotImplementedError
