from easydict import EasyDict as edict

from trainers import conditional_gan_trainer


class ConditionalGAN:
    
    def __init__(self, input_params: edict, generator, discriminator, problem_type,
                 continue_training):
        self.batch_size = input_params.batch_size
        self.num_epochs = input_params.num_epochs
        self.generator = generator
        self.discriminator = discriminator
        
        self.conditional_gan_trainer = conditional_gan_trainer.ConditionalGANTrainer(
            self.batch_size,
            self.generator,
            self.discriminator,
            problem_type,
            input_params.learning_rate_generator,
            input_params.learning_rate_discriminator,
            continue_training)
    
    def fit(self, dataset):
        self.conditional_gan_trainer.train(dataset, self.num_epochs)
