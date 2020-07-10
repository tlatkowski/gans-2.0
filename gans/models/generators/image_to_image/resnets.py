from easydict import EasyDict as edict
from tensorflow.python import keras
from tensorflow.python.keras import Input
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers
from gans.models.discriminators import patch_discriminator
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
        num_scales: int,
        r: int,
):
    generators = []
    for i in range(num_scales):
        if i == 0:
            model_parameters = edict({
                'img_height':       start_dim[0] * (r ** i),
                'img_width':        start_dim[1] * (r ** i),
                'num_channels':     start_dim[2],
                'has_input_images': False,
            })
        else:
            model_parameters = edict({
                'img_height':       start_dim[0] * (r ** i),
                'img_width':        start_dim[1] * (r ** i),
                'num_channels':     start_dim[2],
                'has_input_images': True,
            })
        singan_generator = SingleScaleGenerator(model_parameters)
        generators.append(singan_generator)
    return generators


def build_patch_discriminators(
        start_dim,
        num_scales: int,
        r: int,
):
    discriminators = []
    for i in range(num_scales):
        model_parameters = edict({
            'img_height':       start_dim[0] * (r ** i),
            'img_width':        start_dim[1] * (r ** i),
            'num_channels':     start_dim[2],
        })
        singan_discriminator = patch_discriminator.SinGANPatchDiscriminator(model_parameters)
        discriminators.append(singan_discriminator)
    return discriminators