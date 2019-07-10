import os

import tensorflow as tf

from data_loaders import mnist
from models import discriminators, generators
from models.gan_trainer import GANTrainer


class Random2ImageGAN:
    
    def __init__(self):
        self.batch_size = 16
        self.num_epochs = 10
        hidden_size = 100
        
        img_height = 28
        img_width = 28
        num_channels = 1
        
        self.generator = generators.RandomToImageGenerator(hidden_size)
        z = tf.random.normal(shape=[self.batch_size, hidden_size])
        
        generated_image = self.generator(z)
        
        self.d = discriminators.Discriminator(img_height, img_width, num_channels)
        decision = self.d(generated_image)
        
        noise_dim = 100
        num_examples_to_generate = 16
        
        # We will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        seed = tf.random.normal([num_examples_to_generate, noise_dim])
        
    def fit(self):
        dataset = mnist.load_data(self.batch_size)
        gan_trainer = GANTrainer(self.batch_size, self.generator, self.d)
        gan_trainer.train(dataset, self.num_epochs)
