from easydict import EasyDict as edict
from tensorflow.python.keras import Input
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers

from gans.models import model


class LatentToImageGenerator(model.Model):

    def __init__(
            self,
            model_parameters: edict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        z = Input(shape=[self.model_parameters.latent_size])

        x = layers.Dense(units=7 * 7 * 256, use_bias=False)(z)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Reshape((7, 7, 256))(x)
        x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)

        model = Model(name=self.model_name, inputs=z, outputs=x)
        return model


class LatentToImageCifar10Generator(model.Model):

    def __init__(
            self,
            model_parameters: edict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        z = Input(shape=[self.model_parameters.latent_size])

        x = layers.Dense(units=8 * 8 * 256, use_bias=False)(z)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Reshape((8, 8, 256))(x)
        x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)

        model = Model(name=self.model_name, inputs=z, outputs=x)
        return model


class LatentToImageCifar10NearestNeighborUpSamplingGenerator(model.Model):

    def __init__(
            self,
            model_parameters: edict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        z = Input(shape=[self.model_parameters.latent_size])

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

        model = Model(name=self.model_name, inputs=z, outputs=x)
        return model
