import tensorflow as tf
from easydict import EasyDict as edict

from gans.datasets import summer2winter
from gans.models.discriminators import patch_discriminator
from gans.models.gans import cycle_gan
from gans.models.generators.image_to_image import u_net
from gans.trainers import cycle_gan_trainer

model_parameters = edict({
    'img_height':                  128,
    'img_width':                   128,
    'num_channels':                3,
    'batch_size':                  16,
    'num_epochs':                  10,
    'buffer_size':                 1000,
    'latent_size':                 100,
    'learning_rate_generator':     0.0001,
    'learning_rate_discriminator': 0.0001,
    'save_images_every_n_steps':   10
})

generator_f = u_net.UNetGenerator(model_parameters)
generator_g = u_net.UNetGenerator(model_parameters)

discriminator_f = patch_discriminator.PatchDiscriminator(model_parameters)
discriminator_g = patch_discriminator.PatchDiscriminator(model_parameters)

generator_optimizer_f = tf.keras.optimizers.Adam(
    learning_rate=model_parameters.learning_rate_generator,
    beta_1=0.5,
)
generator_optimizer_g = tf.keras.optimizers.Adam(
    learning_rate=model_parameters.learning_rate_generator,
    beta_1=0.5,
)

discriminator_optimizer_f = tf.keras.optimizers.Adam(
    learning_rate=model_parameters.learning_rate_discriminator,
    beta_1=0.5,
)
discriminator_optimizer_g = tf.keras.optimizers.Adam(
    learning_rate=model_parameters.learning_rate_discriminator,
    beta_1=0.5,
)

gan_trainer = cycle_gan_trainer.CycleGANTrainer(
    batch_size=model_parameters.batch_size,
    generators=[generator_f, generator_g],
    discriminators=[discriminator_f, discriminator_g],
    dataset_type='CYCLE_GAN_SUMMER2WINTER',
    generators_optimizers=[generator_optimizer_f, generator_optimizer_g],
    discriminators_optimizers=[discriminator_optimizer_f, discriminator_optimizer_g],
    continue_training=False,
    save_images_every_n_steps=model_parameters.save_images_every_n_steps,
    visualization_type='image',
)
cycle_gan_model = cycle_gan.CycleGAN(
    model_parameters=model_parameters,
    generators=[generator_f, generator_g],
    discriminators=[discriminator_f, discriminator_g],
    gan_trainer=gan_trainer,
)

dataset = summer2winter.SummerToWinterDataset(model_parameters)

cycle_gan_model.fit(dataset)
