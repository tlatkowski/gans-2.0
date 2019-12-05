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
        save_image_step = 50
        for epoch in range(latest_epoch, num_epochs):
            for first_second_image_batch in dataset():
                first_image_batch, second_image_batch = first_second_image_batch
                train_step += 1
                print(train_step)
                gen_loss_b, dis_loss_b, gen_loss_a, dis_loss_a = self.train_step(
                    first_image_batch, second_image_batch)
                with self.summary_writer.as_default():
                    tf.summary.scalar("generator_loss_b", gen_loss_b, step=train_step)
                    tf.summary.scalar("discriminator_loss_b", dis_loss_b, step=train_step)
                    tf.summary.scalar("generator_loss_a", gen_loss_a, step=train_step)
                    tf.summary.scalar("discriminator_loss_a", dis_loss_a, step=train_step)
                if train_step % save_image_step ==0:
                    img_to_plot = visualization.generate_and_save_images_in(
                        generator_model=self.generator_g,
                        epoch=train_step,
                        test_input=first_image_batch,
                        dataset_name='summer2winter',
                        cmap='gray',
                        num_examples_to_display=1,
                    )
                    img_to_plot = visualization.generate_and_save_images_in(
                        generator_model=self.generator_f,
                        epoch=train_step,
                        test_input=second_image_batch,
                        dataset_name='winter2summer',
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
    def train_step(self, real_x, real_y):
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.
            
            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)
            
            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)
            
            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)
            
            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)
            
            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)
            
            # calculate the loss
            gen_g_loss = losses.generator_loss(disc_fake_y)
            gen_f_loss = losses.generator_loss(disc_fake_x)
            
            total_cycle_loss = losses.cycle_loss(real_x, cycled_x) + losses.cycle_loss(real_y,
                                                                                       cycled_y)
            
            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + losses.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + losses.identity_loss(real_x, same_x)
            
            disc_x_loss = losses.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = losses.discriminator_loss(disc_real_y, disc_fake_y)
        
        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              self.generator_f.trainable_variables)
        
        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  self.discriminator_y.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.generator_optimizer_g.apply_gradients(zip(generator_g_gradients,
                                                       self.generator_g.trainable_variables))
        
        self.generator_optimizer_f.apply_gradients(zip(generator_f_gradients,
                                                       self.generator_f.trainable_variables))
        
        self.discriminator_optimizer_x.apply_gradients(zip(discriminator_x_gradients,
                                                           self.discriminator_x.trainable_variables))
        
        self.discriminator_optimizer_y.apply_gradients(zip(discriminator_y_gradients,
                                                           self.discriminator_y.trainable_variables))
        return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss
    
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
