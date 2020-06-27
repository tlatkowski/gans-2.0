from easydict import EasyDict as edict

from gans.models import model
from gans.trainers import gan_trainer


class ConditionalGAN:

    def __init__(
            self,
            model_parameters: edict,
            generator: model.Model,
            discriminator: model.Model,
            gan_trainer: gan_trainer.GANTrainer,
    ):
        self.batch_size = model_parameters.batch_size
        self.num_epochs = model_parameters.num_epochs
        self.generator = generator
        self.discriminator = discriminator
        self.gan_trainer = gan_trainer

    def fit(self, dataset):
        self.gan_trainer.train(dataset, self.num_epochs)

    def predict(self):
        pass
