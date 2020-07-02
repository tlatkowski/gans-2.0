from typing import List

from easydict import EasyDict as edict

from gans.models import model
from gans.trainers import gan_trainer


class CycleGAN:

    def __init__(
            self,
            model_parameters: edict,
            generators: List[model.Model],
            discriminators: List[model.Model],
            gan_trainer: gan_trainer.GANTrainer,
    ):
        self.num_epochs = model_parameters.num_epochs
        self.generators = generators
        self.discriminators = discriminators
        self.gan_trainer = gan_trainer

    def fit(self, dataset):
        self.gan_trainer.train(
            dataset=dataset,
            num_epochs=self.num_epochs,
        )

    def predict(self, inputs):
        inputs_f, inputs_g = inputs
        generator_f, generator_g = self.generators
        outputs_f = generator_f(inputs_f)
        outputs_g = generator_g(inputs_g)
        return outputs_f, outputs_g
