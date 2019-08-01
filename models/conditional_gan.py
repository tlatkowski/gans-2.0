from easydict import EasyDict as edict

from models import discriminators
from models.gan_trainer import ConditionalGANTrainer
from utils import dataset_utils
from models import generators


class ConditionalGAN:
    
    def __init__(self, input_params: edict, dataset_type):
        self.batch_size = input_params.batch_size
        self.num_epochs = input_params.num_epochs
        self.hidden_size = input_params.hidden_size
        
        self.img_height = input_params.img_height
        self.img_width = input_params.img_width
        self.num_channels = input_params.num_channels
        self.dataset_type = dataset_type
        
        self.generator = generators.RandomToImageConditionalGenerator(input_params)
        self.discriminator = discriminators.ConditionalDiscriminator(input_params)
        self.conditional_gan_trainer = ConditionalGANTrainer(self.batch_size, self.generator,
                                                             self.discriminator,
                                                             self.dataset_type)
    
    def fit(self, dataset):
        self.conditional_gan_trainer.train(dataset, self.num_epochs)
