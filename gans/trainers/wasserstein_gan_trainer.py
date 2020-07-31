import tensorflow as tf

from gans.layers import losses
from gans.models import model
from gans.trainers import gan_trainer

SEED = 0


class WassersteinGANTrainer(gan_trainer.GANTrainer):

    def __init__(
            self,
            batch_size: int,
            generator: model.Model,
            discriminator: model.Model,
            training_name: str,
            generator_optimizer,
            discriminator_optimizer,
            latent_size: int,
            continue_training: bool,
            save_images_every_n_steps: int,
            n_critic=5,
            gp_weight=10.0,
            checkpoint_step=10,
            validation_dataset=None,
            callbacks=None,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.latent_size = latent_size
        self.n_critic = n_critic
        self.gp_weight = gp_weight
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
            checkpoint_step=checkpoint_step,
            validation_dataset=validation_dataset,
            callbacks=callbacks,
        )

    @tf.function
    def train_step(self, batch):
        real_examples = batch

        for _ in range(self.n_critic):
            generator_inputs = tf.random.normal([self.batch_size, self.latent_size])
            with tf.GradientTape(persistent=True) as tape:
                fake_examples = self.generator(generator_inputs, training=True)

                real_output = self.discriminator(real_examples, training=True)
                fake_output = self.discriminator(fake_examples, training=True)

                discriminator_loss = losses.discriminator_loss(real_output, fake_output)
                gradient_penalty = self.gradient_penalty(real_examples, fake_examples)
                discriminator_loss = discriminator_loss + gradient_penalty * self.gp_weight

            gradients_of_discriminator = tape.gradient(
                target=discriminator_loss,
                sources=self.discriminator.trainable_variables,
            )
            self.discriminator_optimizer.apply_gradients(
                grads_and_vars=zip(gradients_of_discriminator, self.discriminator.trainable_variables)
            )

        generator_inputs = tf.random.normal([self.batch_size, self.latent_size])
        with tf.GradientTape(persistent=True) as tape:
            fake_examples = self.generator(generator_inputs, training=True)
            fake_output = self.discriminator(fake_examples, training=True)
            generator_loss = losses.generator_loss(fake_output)

        gradients_of_generator = tape.gradient(
            target=generator_loss,
            sources=self.generator.trainable_variables,
        )

        self.generator_optimizer.apply_gradients(
            grads_and_vars=zip(gradients_of_generator, self.generator.trainable_variables)
        )

        return {
            'generator_loss':     generator_loss,
            'discriminator_loss': discriminator_loss,
            'gradient_penalty':   gradient_penalty,
        }

    def gradient_penalty(self, real_examples, fake_examples):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.normal([self.batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_examples - real_examples
        interpolated = real_examples + alpha * diff

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
