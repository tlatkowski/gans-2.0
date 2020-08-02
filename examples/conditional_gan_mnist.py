import numpy as np
import tensorflow as tf

from gans.callbacks import saver
from gans.datasets import mnist
from gans.models.discriminators import discriminator
from gans.models.generators.latent_to_image import latent_to_image
from gans.trainers import conditional_gan_trainer
from gans.trainers import optimizers

IMG_HEIGHT = 28
IMG_WIDTH = 28
NUM_CHANNELS = 1
BATCH_SIZE = 16
NUM_EPOCHS = 10
BUFFER_SIZE = 1000
LATENT_SIZE = 100
NUM_CLASSES = 10
LEARNING_RATE_GENERATOR = 0.0001
LEARNING_RATE_DISCRIMINATOR = 0.0001
SAVE_IMAGES_EVERY_N_STEPS = 10

dataset = mnist.MnistDataset(model_parameters, with_labels=True)


def validation_dataset():
    test_batch_size = NUM_CLASSES ** 2
    labels = np.repeat(list(range(NUM_CLASSES)), NUM_CLASSES)
    validation_samples = [tf.random.normal([test_batch_size, LATENT_SIZE]), np.array(labels)]
    return validation_samples


validation_dataset = validation_dataset()

generator = latent_to_image.LatentToImageGenerator(model_parameters)
discriminator = discriminator.Discriminator(model_parameters)

generator_optimizer = optimizers.Adam(
    learning_rate=LEARNING_RATE_GENERATOR,
    beta_1=0.5,
)
discriminator_optimizer = optimizers.Adam(
    learning_rate=LEARNING_RATE_DISCRIMINATOR,
    beta_1=0.5,
)

callbacks = [
    saver.ImageProblemSaver(
        save_images_every_n_steps=SAVE_IMAGES_EVERY_N_STEPS,
    )
]

gan_trainer = conditional_gan_trainer.ConditionalGANTrainer(
    batch_size=BATCH_SIZE,
    generator=generator,
    discriminator=discriminator,
    training_name='CONDITIONAL_GAN_MNIST',
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    latent_size=LATENT_SIZE,
    num_classes=NUM_CLASSES,
    continue_training=False,
    save_images_every_n_steps=SAVE_IMAGES_EVERY_N_STEPS,
    validation_dataset=validation_dataset,
    callbacks=callbacks,
)

gan_trainer.train(
    dataset=dataset,
    num_epochs=NUM_EPOCHS,
)
