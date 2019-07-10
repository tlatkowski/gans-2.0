import os

import tensorflow as tf

from data_loaders import mnist
from models import discriminators, generators
from models.gan_trainer import GANTrainer


class Text2ImageGAN:
    
    def __init__(self):
        hidden_size = 100
        
        img_height = 28
        img_width = 28
        num_channels = 1
        
        self.g = generators.RandomToImageGenerator(hidden_size)
        z = tf.random.normal(shape=[16, hidden_size])
        
        generated_image = self.g(z)
        
        self.discriminator = discriminators.Discriminator(img_height, img_width, num_channels)
        decision = self.discriminator(generated_image)
        
        noise_dim = 100
        num_examples_to_generate = 16
        
        # We will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        seed = tf.random.normal([num_examples_to_generate, noise_dim])
        
        self.batch_size = 200
        self.num_epochs = 10
    
    def fit(self):
        dataset = mnist.load_data(self.batch_size)
        gan_trainer = GANTrainer(self.batch_size, self.g, self.discriminator)
        gan_trainer.train(dataset, self.num_epochs)
