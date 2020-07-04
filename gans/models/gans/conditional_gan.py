from easydict import EasyDict as edict

from gans.models import model
from gans.models.gans import gan


class ConditionalGAN(gan.GAN):

    def __init__(
            self,
            model_parameters: edict,
            generator: model.Model,
            discriminator: model.Model,
    ):
        self.num_epochs = model_parameters.num_epochs
        self._generator = generator
        self._discriminator = discriminator

    @property
    def generators(self):
        return [self._generator]

    @property
    def discriminators(self):
        return [self._discriminator]

    def predict(self, inputs):
        return self._generator.model(inputs)
