import tensorflow_addons as tfa
from easydict import EasyDict as edict
from tensorflow.python.keras import Input
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers

from gans.models import model


class Discriminator(model.Model):

    def __init__(
            self,
            model_parameters: edict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        input_img = Input(shape=(
            self.model_parameters.img_height,
            self.model_parameters.img_width,
            self.model_parameters.num_channels
        ))

        x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(input_img)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = tfa.layers.InstanceNormalization(axis=-1)(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', )(x)
        x = tfa.layers.InstanceNormalization(axis=-1)(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
        )(x)
        x = tfa.layers.InstanceNormalization(axis=-1)(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = tfa.layers.InstanceNormalization(axis=-1)(x)
        x = layers.LeakyReLU()(x)

        x = layers.Flatten()(x)
        x = layers.Dense(units=1)(x)

        model = Model(name=self.model_name, inputs=input_img, outputs=x)

        return model
