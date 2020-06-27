from easydict import EasyDict as edict

from gans.datasets import mnist
from gans.models.discriminators import basic_discriminator
from gans.models.gans import conditional_gan
from gans.models.generators.latent_to_image import random_to_image
from gans.trainers import conditional_gan_trainer

model_parameters = edict({
    'img_height': 28,
    'img_width': 28,
    'num_channels': 1,
    'batch_size': 16,
    'num_epochs': 10,
    'buffer_size': 1000,
    'hidden_size': 100,
    'learning_rate_generator': 0.0001,
    'learning_rate_discriminator': 0.0001,
    'save_images_every_n_steps': 10
})

generator = random_to_image.RandomToImageGenerator(model_parameters)
discriminator = basic_discriminator.Discriminator(model_parameters)

gan_trainer = conditional_gan_trainer.ConditionalGANTrainer(
    batch_size=model_parameters.batch_size,
    generator=generator,
    discriminator=discriminator,
    dataset_type='VANILLA_MNIST',
    learning_rate_generator=model_parameters.learning_rate_generator,
    learning_rate_discriminator=model_parameters.learning_rate_discriminator,
    continue_training=False,
    save_images_every_n_steps=model_parameters.save_images_every_n_steps,
)
conditional_gan_model = conditional_gan.ConditionalGAN(
    model_parameters=model_parameters,
    generator=generator,
    discriminator=discriminator,
    gan_trainer=gan_trainer,
)

dataset = mnist.MnistDataset(model_parameters, with_labels=True)

conditional_gan_model.fit(dataset)
