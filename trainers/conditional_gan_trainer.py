import numpy as np
import tensorflow as tf

from layers import losses
from trainers import gan_trainer

SEED = 0
LATENT_SPACE_SIZE = 100
NUM_CLASSES = 10
NUM_TEST_EXAMPLES = 100


class ConditionalGANTrainer(gan_trainer.GANTrainer):

    def __init__(
            self,
            batch_size,
            generator,
            discriminator,
            dataset_type,
            learning_rate_generator,
            learning_rate_discriminator,
            continue_training,
            save_images_every_n_steps,
            checkpoint_step=10,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_generator, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_discriminator, beta_1=0.5)
        super().__init__(
            batch_size=batch_size,
            generators={'generator': generator},
            discriminators={'discriminator': discriminator},
            dataset_type=dataset_type,
            generators_optimizers={
                'generator_optimizer': self.generator_optimizer
            },
            discriminators_optimizers={
                'discriminator_optimizer': self.discriminator_optimizer
            },
            continue_training=continue_training,
            save_images_every_n_steps=save_images_every_n_steps,
            num_test_examples=NUM_TEST_EXAMPLES,
            checkpoint_step=checkpoint_step,
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
            zip(gradients_of_generator, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )

        return {
            'generator_loss':     generator_loss,
            'discriminator_loss': discriminator_loss
        }

    def test_inputs(self, dataset):
        del dataset
        test_batch_size = NUM_CLASSES ** 2
        labels = np.repeat(list(range(NUM_CLASSES)), NUM_CLASSES)
        test_seed = [tf.random.normal([test_batch_size, LATENT_SPACE_SIZE]), np.array(labels)]
        return test_seed
