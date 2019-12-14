from easydict import EasyDict as edict

from trainers import cycle_gan_trainer


class CycleGAN:
    
    def __init__(
            self,
            input_params: edict,
            generators,
            discriminators,
            problem_type,
            continue_training,
    ):
        self.batch_size = input_params.batch_size
        self.num_epochs = input_params.num_epochs
        self.generators = generators
        self.discriminators = discriminators
        self.cycle_gan_trainer = cycle_gan_trainer.CycleGANTrainer(
            self.batch_size,
            self.generators,
            self.discriminators,
            problem_type,
            input_params.learning_rate_generator,
            input_params.learning_rate_discriminator,
            continue_training,
            input_params.save_images_every_n_steps,
        )
    
    def fit(self, dataset):
        self.cycle_gan_trainer.train(
            dataset=dataset,
            num_epochs=self.num_epochs,
        )
    
    def predict(self):
        pass
