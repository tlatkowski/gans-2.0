import tensorflow as tf

import utils
from layers import losses

SEED = 0


class GANTrainer:
    
    def __init__(self, batch_size, generator, discriminator):
        self.batch_size = batch_size
        self.generator = generator
        self.discriminator = discriminator
        
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    
    def train(self, dataset, epochs):
        for epoch in range(epochs):
            print(epoch)
            for image_batch in dataset:
                self.train_step(image_batch)
            
            utils.generate_and_save_images(self.generator, epoch + 1, SEED)
            
            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                # checkpoint.save(file_prefix=checkpoint_prefix)
                print('ok')
    
    @tf.function
    def train_step(self, real_images, generator_inputs=None):
        if generator_inputs is None:
            generator_inputs = tf.random.normal([self.batch_size, 100])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(generator_inputs, training=True)
            
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            
            gen_loss = losses.generator_loss(fake_output)
            disc_loss = losses.discriminator_loss(real_output, fake_output)
        
        gradients_of_generator = gen_tape.gradient(gen_loss, self.g._model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                        self.d._model.trainable_variables)
        
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.g._model.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.d._model.trainable_variables))
