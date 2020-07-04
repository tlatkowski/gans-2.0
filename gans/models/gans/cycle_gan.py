from typing import List

from easydict import EasyDict as edict

from gans.models import model
from gans.models.gans import gan


class CycleGAN(gan.GAN):

    def __init__(
            self,
            model_parameters: edict,
            generators: List[model.Model],
            discriminators: List[model.Model],
    ):
        self.num_epochs = model_parameters.num_epochs
        self._generators = generators
        self._discriminators = discriminators

    @property
    def generators(self):
        return self._generators

    @property
    def discriminators(self):
        return self._discriminators

    def predict(self, inputs):
        inputs_f, inputs_g = inputs
        generator_f, generator_g = self.generators
        outputs_f = generator_f(inputs_f)
        outputs_g = generator_g(inputs_g)
        return outputs_f, outputs_g
