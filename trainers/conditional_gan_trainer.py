import numpy as np
import tensorflow as tf

from layers import losses
from trainers import gan_trainer

SEED = 0
LATENT_SPACE_SIZE = 100
NUM_CLASSES = 10


class ConditionalGANTrainer(gan_trainer.GANTrainer):

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
        super(ConditionalGANTrainer, self).__init__(
            batch_size,
            generator,
            discriminator,
            dataset_type,
            lr_generator,
            lr_discriminator,
            continue_training,
            save_images_every_n_steps,
            checkpoint_step,
        )

    @tf.function
    def train_step(self, batch):
        real_images, real_labels = batch
        batch_size = real_images.shape[0]
        generator_inputs = tf.random.normal([batch_size, LATENT_SPACE_SIZE])
        fake_labels = np.random.randint(0, NUM_CLASSES, batch_size)
        
        with tf.GradientTape(persistent=True) as tape:
            fake_images = self.generator([generator_inputs, fake_labels], training=True)

            real_output = self.discriminator([real_images, real_labels], training=True)
            fake_output = self.discriminator([fake_images, fake_labels], training=True)

            gen_loss = losses.generator_loss(fake_output)
            disc_loss = losses.discriminator_loss(real_output, fake_output)

        gradients_of_generator = tape.gradient(
            target=gen_loss,
            sources=self.generator.trainable_variables,
        )
        gradients_of_discriminator = tape.gradient(
            target=disc_loss,
            sources=self.discriminator.trainable_variables,
        )

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )

        return gen_loss, disc_loss

    def test_seed(self):
        test_batch_size = NUM_CLASSES ** 2
        labels = np.repeat(list(range(NUM_CLASSES)), NUM_CLASSES)
        test_seed = [tf.random.normal([test_batch_size, LATENT_SPACE_SIZE]), np.array(labels)]
        return test_seed
