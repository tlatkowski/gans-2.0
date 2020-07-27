from gans.models.gans import vanilla_gan
from gans.trainers import vanilla_gan_trainer


def build_vanilla_gan(
        model_parameters,
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
):
    gan_trainer = vanilla_gan_trainer.VanillaGANTrainer(
        batch_size=model_parameters.batch_size,
        generator=generator,
        discriminator=discriminator,
        training_name='VANILLA_MNIST',
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        continue_training=False,
        save_images_every_n_steps=model_parameters.save_images_every_n_steps,
    )
    vanilla_gan_model = vanilla_gan.VanillaGAN(
        model_parameters=model_parameters,
        generator=generator,
        discriminator=discriminator,
        gan_trainer=gan_trainer,
    )
    return vanilla_gan_model
