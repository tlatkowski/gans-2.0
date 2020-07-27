import enum
from enum import unique

import tensorflow as tf
from easydict import EasyDict as edict

from gans.datasets import problem_type as pt
from gans.models.discriminators import conditional_discriminator
from gans.models.discriminators import discriminator
from gans.models.discriminators import patch_discriminator
from gans.models.gans import conditional_gan
from gans.models.gans import cycle_gan
from gans.models.gans import vanilla_gan
from gans.models.generators.image_to_image import unet
from gans.models.generators.latent_to_image import conditional_latent_to_image
from gans.models.generators.latent_to_image import latent_to_image
from gans.trainers import conditional_gan_trainer
from gans.trainers import cycle_gan_trainer
from gans.trainers import vanilla_gan_trainer


@unique
class GANType(enum.Enum):
    VANILLA = 'vanilla',
    CONDITIONAL = 'conditional',
    WASSERSTEIN = 'wasserstein',
    CYCLE = 'cycle'


def model_type_values():
    return [i.name for i in GANType]


def gan_model_factory(
        input_params: edict,
        gan_type,
        input_args,
):
    generator = generator_model_factory(input_params, input_args.problem_type)
    discriminator = discriminator_model_factory(input_params, input_args.problem_type)

    if gan_type == GANType.VANILLA.name:
        generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=input_params.learning_rate_generator,
            beta_1=0.5,
        )
        discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=input_params.learning_rate_discriminator,
            beta_1=0.5,
        )

        gan_trainer = vanilla_gan_trainer.VanillaGANTrainer(
            batch_size=input_params.batch_size,
            generator=generator,
            discriminator=discriminator,
            training_name=input_args.problem_type,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            continue_training=input_args.continue_training,
            save_images_every_n_steps=input_params.save_images_every_n_steps,
        )
        return vanilla_gan.VanillaGAN(
            model_parameters=input_params,
            generator=generator,
            discriminator=discriminator,
            gan_trainer=gan_trainer,
        )
    elif gan_type == GANType.CONDITIONAL.name:
        generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=input_params.learning_rate_generator,
            beta_1=0.5,
        )
        discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=input_params.learning_rate_discriminator,
            beta_1=0.5,
        )

        gan_trainer = conditional_gan_trainer.ConditionalGANTrainer(
            batch_size=input_params.batch_size,
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            training_name='VANILLA_MNIST',
            continue_training=False,
            save_images_every_n_steps=input_params.save_images_every_n_steps,
        )
        return conditional_gan.ConditionalGAN(
            model_parameters=input_params,
            generator=generator,
            discriminator=discriminator,
            gan_trainer=gan_trainer,
        )
    elif gan_type == GANType.CYCLE.name:
        generator_f = unet.UNetGenerator(input_params)
        generator_g = unet.UNetGenerator(input_params)

        discriminator_f = patch_discriminator.PatchDiscriminator(input_params)
        discriminator_g = patch_discriminator.PatchDiscriminator(input_params)

        generator_optimizer_f = tf.keras.optimizers.Adam(
            learning_rate=input_params.learning_rate_generator,
            beta_1=0.5,
        )
        generator_optimizer_g = tf.keras.optimizers.Adam(
            learning_rate=input_params.learning_rate_generator,
            beta_1=0.5,
        )

        discriminator_optimizer_f = tf.keras.optimizers.Adam(
            learning_rate=input_params.learning_rate_discriminator,
            beta_1=0.5,
        )
        discriminator_optimizer_g = tf.keras.optimizers.Adam(
            learning_rate=input_params.learning_rate_discriminator,
            beta_1=0.5,
        )

        gan_trainer = cycle_gan_trainer.CycleGANTrainer(
            batch_size=input_params.batch_size,
            generators=[generator_f, generator_g],
            discriminators=[discriminator_f, discriminator_g],
            training_name='SUMMER2WINTER',
            generators_optimizers=[generator_optimizer_f, generator_optimizer_g],
            discriminators_optimizers=[discriminator_optimizer_f, discriminator_optimizer_g],
            continue_training=False,
            save_images_every_n_steps=input_params.save_images_every_n_steps,
        )

        return cycle_gan.CycleGAN(
            model_parameters=input_params,
            generators=[generator_f, generator_g],
            discriminators=[discriminator_f, discriminator_g],
            gan_trainer=gan_trainer,
        )
    elif gan_type == GANType.WASSERSTEIN.name:
        raise NotImplementedError
    else:
        raise NotImplementedError


def generator_model_factory(
        input_params,
        problem_type: pt.ProblemType,
):
    if problem_type == pt.ProblemType.VANILLA_MNIST.name:
        return latent_to_image.LatentToImageGenerator(input_params)
    if problem_type == pt.ProblemType.VANILLA_FASHION_MNIST.name:
        return latent_to_image.LatentToImageGenerator(input_params)
    elif problem_type == pt.ProblemType.VANILLA_CIFAR10.name:
        return latent_to_image.LatentToImageCifar10Generator(input_params)
    elif problem_type == pt.ProblemType.CONDITIONAL_MNIST.name:
        return conditional_latent_to_image.LatentToImageConditionalGenerator(input_params)
    elif problem_type == pt.ProblemType.CONDITIONAL_FASHION_MNIST.name:
        return conditional_latent_to_image.LatentToImageConditionalGenerator(input_params)
    elif problem_type == pt.ProblemType.CONDITIONAL_CIFAR10.name:
        return conditional_latent_to_image.LatentToImageCifar10CConditionalGenerator(
            input_params)
    elif problem_type == pt.ProblemType.CYCLE_SUMMER2WINTER.name:
        return [unet.UNetGenerator(input_params), unet.UNetGenerator(input_params)]
    else:
        raise NotImplementedError


def discriminator_model_factory(
        input_params,
        dataset_type: pt.ProblemType,
):
    if dataset_type == pt.ProblemType.VANILLA_MNIST.name:
        return discriminator.Discriminator(input_params)
    if dataset_type == pt.ProblemType.VANILLA_FASHION_MNIST.name:
        return discriminator.Discriminator(input_params)
    elif dataset_type == pt.ProblemType.VANILLA_CIFAR10.name:
        return discriminator.Discriminator(input_params)
    elif dataset_type == pt.ProblemType.CONDITIONAL_MNIST.name:
        return conditional_discriminator.ConditionalDiscriminator(input_params)
    elif dataset_type == pt.ProblemType.CONDITIONAL_FASHION_MNIST.name:
        return conditional_discriminator.ConditionalDiscriminator(input_params)
    elif dataset_type == pt.ProblemType.CONDITIONAL_CIFAR10.name:
        return conditional_discriminator.ConditionalDiscriminatorCifar10(input_params)
    elif dataset_type == pt.ProblemType.CYCLE_SUMMER2WINTER.name:
        return [patch_discriminator.PatchDiscriminator(input_params),
                patch_discriminator.PatchDiscriminator(input_params)]
    else:
        raise NotImplementedError
