from easydict import EasyDict as edict

from gans.callbacks import saver
from gans.datasets import summer2winter
from gans.models.discriminators import patch_discriminator
from gans.models.generators.image_to_image import unet
from gans.trainers import cycle_gan_trainer
from gans.trainers import optimizers

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
dataset = summer2winter.SummerToWinterDataset(model_parameters)


def validation_dataset(dataset):
    summer, _ = next(dataset.train_dataset)
    summer = summer[:4]
    return summer


validation_dataset = validation_dataset(dataset)

generator_f = unet.UNetGenerator(model_parameters)
generator_g = unet.UNetGenerator(model_parameters)

discriminator_f = patch_discriminator.PatchDiscriminator(model_parameters)
discriminator_g = patch_discriminator.PatchDiscriminator(model_parameters)

generator_optimizer_f = optimizers.Adam(
    learning_rate=model_parameters.learning_rate_generator,
    beta_1=0.5,
)
generator_optimizer_g = optimizers.Adam(
    learning_rate=model_parameters.learning_rate_generator,
    beta_1=0.5,
)

discriminator_optimizer_f = optimizers.Adam(
    learning_rate=model_parameters.learning_rate_discriminator,
    beta_1=0.5,
)
discriminator_optimizer_g = optimizers.Adam(
    learning_rate=model_parameters.learning_rate_discriminator,
    beta_1=0.5,
)

callbacks = [
    saver.ImageProblemSaver(
        save_images_every_n_steps=model_parameters.save_images_every_n_steps,
    )
]

gan_trainer = cycle_gan_trainer.CycleGANTrainer(
    batch_size=model_parameters.batch_size,
    generators=[generator_f, generator_g],
    discriminators=[discriminator_f, discriminator_g],
    training_name='CYCLE_GAN_SUMMER2WINTER',
    generators_optimizers=[generator_optimizer_f, generator_optimizer_g],
    discriminators_optimizers=[discriminator_optimizer_f, discriminator_optimizer_g],
    continue_training=False,
    save_images_every_n_steps=model_parameters.save_images_every_n_steps,
    validation_dataset=validation_dataset,
    callbacks=callbacks,
)

gan_trainer.train(
    dataset=dataset,
    num_epochs=model_parameters.num_epochs,
)
