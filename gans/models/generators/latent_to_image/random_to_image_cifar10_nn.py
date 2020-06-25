from easydict import EasyDict as edict
from tensorflow.python.keras import Input
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers

from gans.models.generators import generator


class RandomToImageCifar10NearestNeighborUpSamplingGenerator(generator.Generator):

    def __init__(
            self,
            model_parameters: edict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        z = Input(shape=[self.model_parameters.hidden_size])

        x = layers.Dense(units=8 * 8 * 256, use_bias=False)(z)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Reshape((8, 8, 256))(x)
        x = layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')(x)

        model = Model(name=self, inputs=z, outputs=x)
        return model
