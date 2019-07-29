import tensorflow as tf

from layers import losses
from utils import dataset_utils

SEED = 0
CHECKPOINT_DIR = './training_checkpoints'


class GANTrainer:
    
    def __init__(self, batch_size, generator, discriminator, checkpoint_step=15):
        self.batch_size = batch_size
        self.generator = generator
        self.discriminator = discriminator
        self.checkpoint_step = checkpoint_step
        
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        
        # self.checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
        # self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
        #                                       discriminator_optimizer=self.discriminator_optimizer,
        #                                       generator=self.generator,
        #                                       discriminator=self.discriminator)
    
    def train(self, dataset, epochs):
        i = 0
        for epoch in range(epochs):
            print(epoch)
            for image_batch in dataset:
                i += 1
                self.train_step(image_batch)
                print(i)
            dataset_utils.generate_and_save_images(self.generator, epoch + 1,
                                                   tf.random.normal([self.batch_size, 100]))
            
            if (epoch + 1) % self.checkpoint_step == 0:
                # self.checkpoint.save(file_prefix=self.checkpoint_prefix)
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
        
        gradients_of_generator = gen_tape.gradient(gen_loss,
                                                   self.generator._model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                        self.discriminator._model.trainable_variables)
        
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator._model.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator._model.trainable_variables))
