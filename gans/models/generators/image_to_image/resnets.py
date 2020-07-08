from easydict import EasyDict as edict
from tensorflow.python import keras
from tensorflow.python.keras import Input
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers

from gans.models import model


class SingleScaleGenerator(model.Model):

    def __init__(
            self,
            model_parameters: edict,
    ):
        super().__init__(model_parameters)

    def define_model(self) -> keras.Model:
        x = Input(shape=[
            self.model_parameters.img_height,
            self.model_parameters.img_width,
            self.model_parameters.num_channels
        ])
        z = Input(shape=[
            self.model_parameters.img_height,
            self.model_parameters.img_width,
            self.model_parameters.num_channels
        ])

        xz = z
        if self.model_parameters.has_input_images:
            xz += x

        xz = layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            use_bias=False,
        )(xz)
        xz = layers.BatchNormalization()(xz)
        xz = layers.LeakyReLU(alpha=0.2)(xz)

        xz = layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            use_bias=False,
        )(xz)
        xz = layers.BatchNormalization()(xz)
        xz = layers.LeakyReLU(alpha=0.2)(xz)

        xz = layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            use_bias=False,
        )(xz)
        xz = layers.BatchNormalization()(xz)
        xz = layers.LeakyReLU(alpha=0.2)(xz)

        xz = layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            use_bias=False,
        )(xz)
        xz = layers.BatchNormalization()(xz)
        xz = layers.LeakyReLU(alpha=0.2)(xz)

        xz = layers.Conv2D(
            filters=3,
            kernel_size=(3, 3),
            padding='same',
            activation='tanh',
            use_bias=False,
        )(xz)

        if self.model_parameters.has_input_images:
            xz += x

        model = Model(name=self.model_name, inputs=[x, z], outputs=xz)
        return model


def build_progressive_generators(
        start_dim,
        numscale: int,
        model_parameters,
):
    generators = []
    for _ in range(numscale):
        singan_generator = SingleScaleGenerator(model_parameters)
        generators.append(singan_generator)
