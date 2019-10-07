from easydict import EasyDict as edict

from models.discriminators import basic_conditional_discriminator
from trainers import conditional_gan_trainer
from utils import model_utils


class ConditionalGAN:
    
    def __init__(self, input_params: edict, input_args):
        self.batch_size = input_params.batch_size
        self.num_epochs = input_params.num_epochs
        self.hidden_size = input_params.hidden_size
        
        self.img_height = input_params.img_height
        self.img_width = input_params.img_width
        self.num_channels = input_params.num_channels
        self.problem_type = input_args.problem_type
        
        self.generator = model_utils.generator_model_factory(input_params, self.problem_type)
        self.discriminator = model_utils.discriminator_model_factory(input_params, self.problem_type)
        self.conditional_gan_trainer = conditional_gan_trainer.ConditionalGANTrainer(
            self.batch_size,
            self.generator,
            self.discriminator,
            self.problem_type,
            input_params.learning_rate_generator,
            input_params.learning_rate_discriminator,
            input_args.continue_training)
    
    def fit(self, dataset):
        self.conditional_gan_trainer.train(dataset, self.num_epochs)
