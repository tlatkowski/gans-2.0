import os

import tensorflow as tf

from models import discriminator, generators
from models import vanilla_gan
from data_loaders import mnist


class Text2ImageGAN:
    
    def __init__(self):
        
        hidden_size = 100
        
        img_height = 28
        img_width = 28
        num_channels = 1
        
        self.g = generators.RandomToImageGenerator(hidden_size)
        z = tf.random.normal(shape=[16, 100])
        
        generated_image = g(z)
        
        self.d = discriminator.Discriminator(img_height, img_width, num_channels)
        decision = self.d(generated_image)
        
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        # checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
        #                                  discriminator_optimizer=discriminator_optimizer,
        #                                  generator=generator,
        #                                  discriminator=discriminator)
        
        noise_dim = 100
        num_examples_to_generate = 16
        
        # We will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        seed = tf.random.normal([num_examples_to_generate, noise_dim])
        
        self.batch_size = 200
        self.num_epochs = 10
    
    def fit(self):
        dataset = mnist.load_data(self.batch_size)
        gan_trainer = vanilla_gan.GANTrainer(self.batch_size, self.g, self.d)
        gan_trainer.train(dataset, self.num_epochs)



