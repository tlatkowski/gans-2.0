import tensorflow_addons as tfa
from easydict import EasyDict as edict
from tensorflow.python.keras import Input
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers

from gans.models import model


class PatchDiscriminator(model.Model):

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

        x = layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same')(input_img)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
        x = tfa.layers.InstanceNormalization(axis=-1)(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', )(x)
        x = tfa.layers.InstanceNormalization(axis=-1)(x)
        x = layers.LeakyReLU()(x)

        x = layers.ZeroPadding2D()(x)

        x = layers.Conv2D(filters=512, kernel_size=(4, 4), strides=(1, 1), padding='valid')(x)
        x = tfa.layers.InstanceNormalization(axis=-1)(x)
        x = layers.LeakyReLU()(x)

        x = layers.ZeroPadding2D()(x)

        x = layers.Conv2D(filters=1, kernel_size=(4, 4), strides=(1, 1), padding='valid')(x)

        model = Model(name=self.model_name, inputs=input_img, outputs=x)

        return model


class SinGANPatchDiscriminator(model.Model):

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

        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_img)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        model = Model(name=self.model_name, inputs=input_img, outputs=x)

        return model
