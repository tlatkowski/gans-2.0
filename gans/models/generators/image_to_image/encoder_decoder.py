import tensorflow_addons as tfa
from easydict import EasyDict as edict
from tensorflow.python.keras import Input
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers

from gans.layers import advanced_layers
from gans.models import model


class EncoderDecoderGenerator(model.Model):

    def __init__(
            self,
            model_parameters: edict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        input_images = Input(shape=[
            self.model_parameters.img_height,
            self.model_parameters.img_width,
            self.model_parameters.num_channels
        ])

        x = layers.Conv2D(
            filters=64,
            kernel_size=(7, 7),
            padding='same',
            use_bias=False,
        )(input_images)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=False,
        )(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=False,
        )(x)

        x = layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=False,
        )(x)
        n_resnet = 6
        for _ in range(n_resnet):
            x = advanced_layers.residual_block(256, x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.UpSampling2D()(x)

        x = layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(
            filters=32,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(
            filters=3,
            kernel_size=(7, 7),
            strides=(1, 1),
            padding='same',
            use_bias=False,
            activation='tanh',
        )(x)

        model = Model(name=self.model_name, inputs=input_images, outputs=x)
        return model
