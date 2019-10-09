import numpy as np
import tensorflow as tf

from layers import losses
from trainers import gan_trainer
from utils import visualization

SEED = 0


class VanillaGANTrainer(gan_trainer.GANTrainer):
    
    def __init__(self, batch_size, generator, discriminator, dataset_type, lr_generator,
                 lr_discriminator, continue_training, checkpoint_step=10):
        super(VanillaGANTrainer, self).__init__(batch_size, generator, discriminator,
                                                dataset_type, lr_generator, lr_discriminator,
                                                continue_training, checkpoint_step)
    
    def train(self, dataset, epochs):
        train_step = 0
        test_seed = tf.random.normal([self.batch_size, 100])
        
        latest_checkpoint_epoch = 0
        if self.continue_training:
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
            if latest_checkpoint is not None:
                latest_checkpoint_epoch = int(latest_checkpoint[latest_checkpoint.index("-") + 1:])
                self.checkpoint.restore(latest_checkpoint)
            else:
                print('No checkpoints found. Starting training from scratch.')
        latest_epoch = latest_checkpoint_epoch * self.checkpoint_step
        epochs += latest_epoch
        for epoch in range(latest_epoch, epochs):
            for image_batch in dataset:
                train_step += 1
                gen_loss, dis_loss = self.train_step(image_batch)
                with self.summary_writer.as_default():
                    tf.summary.scalar("generator_loss", gen_loss, step=train_step)
                    tf.summary.scalar("discriminator_loss", dis_loss, step=train_step)
            
            img_to_plot = visualization.generate_and_save_images(self.generator,
                                                                 epoch + 1,
                                                                 test_seed,
                                                                 self.dataset_type,
                                                                 cmap='gray')
            with self.summary_writer.as_default():
                tf.summary.image('test_images',
                                 np.reshape(img_to_plot, newshape=(1, 480, 640, 4)),
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
        
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                        self.discriminator.trainable_variables)
        
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss
