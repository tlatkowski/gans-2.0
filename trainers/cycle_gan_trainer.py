import numpy as np
import tensorflow as tf

from layers import losses
from trainers import gan_trainer
from utils import visualization

SEED = 0


class CycleGANTrainer(gan_trainer.GANTrainer):
    
    def __init__(
            self,
            batch_size,
            generator,
            discriminator,
            dataset_type,
            lr_generator,
            lr_discriminator,
            continue_training,
            checkpoint_step=10,
    ):
        super(CycleGANTrainer, self).__init__(
            batch_size,
            generator,
            discriminator,
            dataset_type,
            lr_generator,
            lr_discriminator,
            continue_training,
            checkpoint_step,
        )
    
    def train(self, dataset, num_epochs):
        train_step = 0
        test_seed = tf.random.normal([self.batch_size, 100])
        
        latest_checkpoint_epoch = self.regenerate_training()
        latest_epoch = latest_checkpoint_epoch * self.checkpoint_step
        num_epochs += latest_epoch
        for epoch in range(latest_epoch, num_epochs):
            for first_second_image_batch in dataset():
                train_step += 1
                print(train_step)
                gen_loss, dis_loss = self.train_step(first_second_image_batch)
                with self.summary_writer.as_default():
                    tf.summary.scalar("generator_loss", gen_loss, step=train_step)
                    tf.summary.scalar("discriminator_loss", dis_loss, step=train_step)
            
                img_to_plot = visualization.generate_and_save_images(
                    generator_model=self.generator,
                    epoch=epoch + 1,
                    test_input=first_second_image_batch[0],
                    dataset_name=self.dataset_type,
                    cmap='gray',
                    num_examples_to_display=1,
                )
            with self.summary_writer.as_default():
                tf.summary.image(
                    name='test_images',
                    data=np.reshape(img_to_plot, newshape=(1, 480, 640, 4)),
                    step=epoch,
                )
            
            if (epoch + 1) % self.checkpoint_step == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
    
    @tf.function
    def train_step(self, train_batch):
        first_dataset_batch, second_dataset_batch = train_batch
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(first_dataset_batch, training=True)
            
            real_output = self.discriminator(second_dataset_batch, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            
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
        
        return generator_loss, discriminator_loss
    
    def regenerate_training(self):
        latest_checkpoint_epoch = 0
        if self.continue_training:
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
            if latest_checkpoint is not None:
                latest_checkpoint_epoch = int(latest_checkpoint[latest_checkpoint.index("-") + 1:])
                self.checkpoint.restore(latest_checkpoint)
            else:
                print('No checkpoints found. Starting training from scratch.')
        return latest_checkpoint_epoch

