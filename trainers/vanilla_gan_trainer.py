import tensorflow as tf

from layers import losses
from trainers import gan_trainer
from utils import logging

SEED = 0
LATENT_SPACE_SIZE = 100
NUM_TEST_EXAMPLES = 16

logger = logging.get_logger(__name__)


class VanillaGANTrainer(gan_trainer.GANTrainer):

    def __init__(
            self,
            batch_size,
            generator,
            discriminator,
            dataset_type,
            lr_generator,
            lr_discriminator,
            continue_training,
            save_images_every_n_steps,
            checkpoint_step=10,
    ):
        super(VanillaGANTrainer, self).__init__(
            batch_size,
            generator,
            discriminator,
            dataset_type,
            lr_generator,
            lr_discriminator,
            continue_training,
            save_images_every_n_steps,
            NUM_TEST_EXAMPLES,
            checkpoint_step,
        )

    @tf.function
    def train_step(self, batch):
        real_images = batch
        generator_inputs = tf.random.normal([self.batch_size, 100])

        with tf.GradientTape(persistent=True) as tape:
            fake_images = self.generator(generator_inputs, training=True)

            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)

            generator_loss = losses.generator_loss(fake_output)
            discriminator_loss = losses.discriminator_loss(real_output, fake_output)

        gradients_of_generator = tape.gradient(
            target=generator_loss,
            sources=self.generator.trainable_variables,
        )
        gradients_of_discriminator = tape.gradient(
            target=discriminator_loss,
            sources=self.discriminator.trainable_variables,
        )

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return {
            'generator_loss':     generator_loss,
            'discriminator_loss': discriminator_loss
        }

    def test_seed(self):
        return tf.random.normal([self.batch_size, LATENT_SPACE_SIZE])
