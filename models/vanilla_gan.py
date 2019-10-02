from easydict import EasyDict as edict

from models import discriminators
from trainers.gan_trainer import VanillaGANTrainer
from utils import model_utils


class VanillaGAN:
    
    def __init__(self, input_params: edict, input_args):
        self.batch_size = input_params.batch_size
        self.num_epochs = input_params.num_epochs
        self.problem_type = input_args.problem_type
        
        self.generator = model_utils.generator_model_factory(input_params, self.problem_type)
        self.discriminator = discriminators.Discriminator(input_params)
        self.vanilla_gan_trainer = VanillaGANTrainer(self.batch_size,
                                                     self.generator,
                                                     self.discriminator,
                                                     self.problem_type,
                                                     input_args.learning_rate_generator,
                                                     input_args.learning_rate_discriminator,
                                                     input_args.continue_training)
    
    def fit(self, dataset):
        self.vanilla_gan_trainer.train(dataset, self.num_epochs)
