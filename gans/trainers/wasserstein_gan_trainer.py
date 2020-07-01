import tensorflow as tf

from gans.layers import losses
from gans.utils import visualization

SEED = 0


class WassersteinGANTrainer:
    
    def __init__(
            self,
            batch_size,
            generator,
            discriminator,
            dataset_type,
            checkpoint_step=15,
    ):
        self.batch_size = batch_size
        self.generator = generator
        self.discriminator = discriminator
        self.checkpoint_step = checkpoint_step
        self.dataset_type = dataset_type
        
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    
    def train(self, dataset, epochs):
        seed = tf.random.normal([self.batch_size, 100])
        for epoch in range(epochs):
            print(epoch)
            for real_images in dataset:
                self.train_step(real_images)
            visualization.generate_and_save_images_for_image_problems(
                self.generator,
                epoch + 1,
                seed,
                self.dataset_type,
            )
            
            if (epoch + 1) % self.checkpoint_step == 0:
                pass
    
    @tf.function
    def train_step(self, real_images, generator_inputs=None):
        if generator_inputs is None:
            generator_inputs = tf.random.normal([self.batch_size, 100])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(generator_inputs, training=True)
            
            weights = tf.random.uniform((1, 1, 1, 1))
            average_images = (weights * real_images) + ((1 - weights) * fake_images)
            
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            average_output = self.discriminator(average_images, training=True)
            
            generator_loss = losses.generator_loss(fake_output)
            discriminator_loss = losses.discriminator_loss(real_output, fake_output)
        
        gradients_of_generator = gen_tape.gradient(
            generator_loss,
            self.generator.trainable_variables,
        )
        gradients_of_discriminator = disc_tape.gradient(
            discriminator_loss,
            self.discriminator.trainable_variables,
        )
        
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
