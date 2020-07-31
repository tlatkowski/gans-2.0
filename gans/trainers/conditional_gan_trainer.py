import numpy as np
import tensorflow as tf

from gans.layers import losses
from gans.models import model
from gans.trainers import gan_trainer

SEED = 0


class ConditionalGANTrainer(gan_trainer.GANTrainer):

    def __init__(
            self,
            batch_size: int,
            generator: model.Model,
            discriminator: model.Model,
            training_name: str,
            generator_optimizer,
            discriminator_optimizer,
            latent_size: int,
            num_classes: int,
            continue_training: bool,
            save_images_every_n_steps: int,
            visualization_type: str,
            checkpoint_step: int = 10,
            validation_dataset=None,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.latent_size = latent_size
        self.num_classes = num_classes
        super().__init__(
            batch_size=batch_size,
            generators={'generator': generator},
            discriminators={'discriminator': discriminator},
            training_name=training_name,
            generators_optimizers={
                'generator_optimizer': self.generator_optimizer
            },
            discriminators_optimizers={
                'discriminator_optimizer': self.discriminator_optimizer
            },
            continue_training=continue_training,
            save_images_every_n_steps=save_images_every_n_steps,
            visualization_type=visualization_type,
            checkpoint_step=checkpoint_step,
            validation_dataset=validation_dataset,
        )

    @tf.function
    def train_step(self, batch):
        real_examples, real_labels = batch
        batch_size = real_examples.shape[0]
        generator_inputs = tf.random.normal([batch_size, self.latent_size])
        fake_labels = np.random.randint(0, self.num_classes, batch_size)

        with tf.GradientTape(persistent=True) as tape:
            fake_examples = self.generator([generator_inputs, fake_labels], training=True)

            real_output = self.discriminator([real_examples, real_labels], training=True)
            fake_output = self.discriminator([fake_examples, fake_labels], training=True)

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
            grads_and_vars=zip(gradients_of_generator, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            grads_and_vars=zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )

        return {
            'generator_loss':     generator_loss,
            'discriminator_loss': discriminator_loss
        }
