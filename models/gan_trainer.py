import os

import numpy as np
import tensorflow as tf

from layers import losses
from utils import constants
from utils import visualization

SEED = 0


class GANTrainer:
    
    def __init__(self, batch_size, generator, discriminator, dataset_type, continue_training,
                 checkpoint_step=10):
        self.batch_size = batch_size
        self.generator = generator
        self.discriminator = discriminator
        self.checkpoint_step = checkpoint_step
        self.dataset_type = dataset_type
        self.continue_training = continue_training
        
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        
        self.checkpoint_path = os.path.join(constants.SAVE_IMAGE_DIR, dataset_type,
                                            constants.CHECKPOINT_DIR)
        
        self.checkpoint_prefix = os.path.join(self.checkpoint_path, "ckpt")
        
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator.model,
                                              discriminator=self.discriminator.model)
        self.summary_writer = tf.summary.create_file_writer(self.checkpoint_path)


class VanillaGANTrainer(GANTrainer):
    
    def __init__(self, batch_size, generator, discriminator, dataset_type, continue_training,
                 checkpoint_step=10):
        super(VanillaGANTrainer, self).__init__(batch_size, generator, discriminator,
                                                dataset_type, continue_training, checkpoint_step)
    
    def train(self, dataset, epochs):
        test_seed = tf.random.normal([self.batch_size, 100])
        
        latest_checkpoint_epoch = 0
        if self.continue_training:
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
            latest_checkpoint_epoch = int(latest_checkpoint[latest_checkpoint.index("-") + 1:])
            self.checkpoint.restore(latest_checkpoint)
        latest_epoch = latest_checkpoint_epoch * self.checkpoint_step
        epochs += latest_epoch
        for epoch in range(latest_epoch, epochs):
            print(epoch)
            for image_batch in dataset:
                self.train_step(image_batch)
            
            img_to_plot = visualization.generate_and_save_images(self.generator,
                                                                 epoch + 1,
                                                                 test_seed,
                                                                 self.dataset_type,
                                                                 cmap='gray')
            with self.summary_writer.as_default():
                tf.summary.image('test_images', np.reshape(img_to_plot, newshape=(1, 480, 640, 4)),
                                 step=epoch)
            
            if (epoch + 1) % self.checkpoint_step == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
    
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
                                                   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                        self.discriminator.trainable_variables)
        
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))


class ConditionalGANTrainer(GANTrainer):
    
    def __init__(self, batch_size, generator, discriminator, dataset_type, continue_training,
                 checkpoint_step=10):
        super(ConditionalGANTrainer, self).__init__(batch_size, generator, discriminator,
                                                    dataset_type, continue_training,
                                                    checkpoint_step)
    
    def train(self, dataset, epochs):
        train_step = 0
        
        latest_checkpoint_epoch = 0
        if self.continue_training:
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
            self.checkpoint.restore(latest_checkpoint)
            latest_checkpoint_epoch = int(latest_checkpoint[latest_checkpoint.index("-") + 1:])
        latest_epoch = latest_checkpoint_epoch * self.checkpoint_step
        epochs += latest_epoch
        for epoch in range(latest_epoch, epochs):
            print(epoch)
            for image_batch in dataset:
                train_step += 1
                gen_loss, dis_loss = self.train_step(image_batch)
                with self.summary_writer.as_default():
                    tf.summary.scalar("generator_loss", gen_loss, step=train_step)
                    tf.summary.scalar("discriminator_loss", dis_loss, step=train_step)
            
            test_batch = 100
            labels = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10 + [5] * 10 + [6] * 10 + [
                7] * 10 + [8] * 10 + [9] * 10
            test_seed = [tf.random.normal([test_batch, 100]),
                         np.array(labels)]
            
            img_to_plot = visualization.generate_and_save_images(self.generator, epoch + 1,
                                                                 test_seed,
                                                                 self.dataset_type,
                                                                 num_examples_to_display=test_batch)
            with self.summary_writer.as_default():
                tf.summary.image('test_images', np.reshape(img_to_plot, newshape=(1, 480, 640, 4)),
                                 step=epoch)
            if (epoch + 1) % self.checkpoint_step == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
    
    @tf.function
    def train_step(self, real_images):
        real_images, real_labels = real_images
        
        batch_size = real_images.shape[0]
        generator_inputs = tf.random.normal([batch_size, 100])
        fake_labels = np.random.randint(0, 10, batch_size)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator([generator_inputs, fake_labels], training=True)
            
            real_output = self.discriminator([real_images, real_labels], training=True)
            fake_output = self.discriminator([fake_images, fake_labels], training=True)
            
            gen_loss = losses.generator_loss(fake_output)
            disc_loss = losses.discriminator_loss(real_output, fake_output)
        
        gradients_of_generator = gen_tape.gradient(gen_loss,
                                                   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                        self.discriminator.trainable_variables)
        
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss


class WassersteinGANTrainer:
    
    def __init__(self, batch_size, generator, discriminator, dataset_type, checkpoint_step=15):
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
            visualization.generate_and_save_images(self.generator, epoch + 1, seed,
                                                   self.dataset_type)
            
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
        
        gradients_of_generator = gen_tape.gradient(generator_loss,
                                                   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(discriminator_loss,
                                                        self.discriminator.trainable_variables)
        
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
