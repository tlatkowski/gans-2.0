from easydict import EasyDict as edict

from gans.datasets import problem_type as pt
from gans.models import model
from gans.trainers import conditional_gan_trainer


class ConditionalGAN:

    def __init__(
            self,
            model_parameters: edict,
            generator: model.Model,
            discriminator: model.Model,
            problem_type: pt.ProblemType,
            continue_training: bool,
    ):
        self.batch_size = model_parameters.batch_size
        self.num_epochs = model_parameters.num_epochs
        self.generator = generator
        self.discriminator = discriminator

        self.conditional_gan_trainer = conditional_gan_trainer.ConditionalGANTrainer(
            self.batch_size,
            self.generator,
            self.discriminator,
            problem_type,
            model_parameters.learning_rate_generator,
            model_parameters.learning_rate_discriminator,
            continue_training,
            model_parameters.save_images_every_n_steps,
        )

    def fit(self, dataset):
        self.conditional_gan_trainer.train(dataset, self.num_epochs)

    def predict(self):
        pass
