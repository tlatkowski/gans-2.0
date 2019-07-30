from easydict import EasyDict as edict

from models import discriminators, generators
from models.gan_trainer import GANTrainer


class Random2ImageGAN:
    
    def __init__(self, input_params: edict, dataset_type):
        self.batch_size = input_params.batch_size
        self.num_epochs = input_params.num_epochs
        self.hidden_size = input_params.hidden_size
        
        self.img_height = input_params.img_height
        self.img_width = input_params.img_width
        self.num_channels = input_params.num_channels
        self.dataset_type = dataset_type
        
        self.generator = generators.RandomToImageGenerator(self.hidden_size)
        self.discriminator = discriminators.Discriminator(self.img_height,
                                                          self.img_width,
                                                          self.num_channels)
    
    def fit(self, dataset):
        gan_trainer = GANTrainer(self.batch_size, self.generator, self.discriminator, self.dataset_type)
        gan_trainer.train(dataset, self.num_epochs)
