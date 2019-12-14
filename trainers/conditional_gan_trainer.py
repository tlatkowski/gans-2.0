import numpy as np
import tensorflow as tf

from layers import losses
from trainers import gan_trainer
from utils import visualization

SEED = 0


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
    
    def train(self, dataset, epochs):
        test_batch_size = 100
        labels = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10 + [5] * 10 + [6] * 10 + [
            7] * 10 + [8] * 10 + [9] * 10
        test_seed = [tf.random.normal([test_batch_size, 100]),
                     np.array(labels)]
        
        train_step = 0
        latest_checkpoint_epoch = 0
        
        if self.continue_training:
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
            self.checkpoint.restore(latest_checkpoint)
            latest_checkpoint_epoch = int(latest_checkpoint[latest_checkpoint.index("-") + 1:])
        latest_epoch = latest_checkpoint_epoch * self.checkpoint_step
        epochs += latest_epoch
        for epoch in range(latest_epoch, epochs):
            for image_batch in dataset.train_dataset:
                gen_loss, dis_loss = self.train_step(image_batch)
                with self.summary_writer.as_default():
                    tf.summary.scalar("generator_loss", gen_loss, step=train_step)
                    tf.summary.scalar("discriminator_loss", dis_loss, step=train_step)
                
                if train_step % self.save_images_every_n_steps == 0:
                    img_to_plot = visualization.generate_and_save_images(
                        generator_model=self.generator,
                        epoch=train_step,
                        test_input=test_seed,
                        dataset_name=self.dataset_type,
                        num_examples_to_display=test_batch_size,
                    )
                train_step += 1
            
            with self.summary_writer.as_default():
                pass
                # tf.summary.image('test_images', np.reshape(img_to_plot, newshape=(1, 480, 640, 4)),
                #                  step=epoch)
            if (epoch + 1) % self.checkpoint_step == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
    
    @tf.function
    def train_step(self, real_images):
        real_images, real_labels = real_images
        
        batch_size = real_images.shape[0]
        generator_inputs = tf.random.normal([batch_size, 100])
        fake_labels = np.random.randint(0, 10, batch_size)
        # with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
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
