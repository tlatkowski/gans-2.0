import tensorflow as tf
from easydict import EasyDict as edict

from models import discriminators, generators
from models.gan_trainer import GANTrainer


class Random2ImageGAN:
    
    def __init__(self, input_params: edict):
        self.batch_size = input_params.batch_size
        self.num_epochs = input_params.num_epochs  # 10
        self.hidden_size = input_params.hidden_size  # 100
        
        self.img_height = input_params.img_height  # 28
        self.img_width = input_params.img_width  # 28
        self.num_channels = input_params.num_channels  # 1
        
        self.generator = generators.RandomToImageGenerator(self.hidden_size)
        z = tf.random.normal(shape=[self.batch_size, self.hidden_size])
        
        generated_image = self.generator(z, training=False)
        
        self.discriminator = discriminators.Discriminator(self.img_height, self.img_width,
                                                          self.num_channels)
        # decision = self.discriminator(generated_image)
        
        # We will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        # seed = tf.random.normal([num_examples_to_generate, noise_dim])
    
    def fit(self, dataset):
        gan_trainer = GANTrainer(self.batch_size, self.generator, self.discriminator)
        gan_trainer.train(dataset, self.num_epochs)
