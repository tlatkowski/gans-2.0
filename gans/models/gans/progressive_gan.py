from gans.models.gans import gan
import tensorflow as tf

class ProgressiveGAN(gan.GAN):

    def __init__(
            self,
            generators,
    ):
        self._generators = generators

    @property
    def generators(self):
        return self._generators

    @property
    def discriminators(self):
        pass

    def predict(self, inputs):
        x = tf.ones(shape=inputs[0].shape.as_list())
        for i, (z, generator) in enumerate(zip(inputs, self.generators)):
            size = z.shape.as_list()[1:3]
            x = tf.image.resize(x, size=size)
            x = generator([z, x])
