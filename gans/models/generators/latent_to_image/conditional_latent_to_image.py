from easydict import EasyDict as edict
from tensorflow.python.keras import Input
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers

from gans.models import model


class LatentToImageConditionalGenerator(model.Model):

    def __init__(
            self,
            model_parameters: edict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        z = Input(shape=[self.model_parameters.latent_size])
        class_id = Input(shape=[1])

        embedded_id = layers.Embedding(input_dim=10, output_dim=50)(class_id)
        embedded_id = layers.Dense(units=7 * 7)(embedded_id)
        embedded_id = layers.Reshape(target_shape=(7, 7, 1))(embedded_id)

        x = layers.Dense(units=7 * 7 * 256, use_bias=False)(z)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Reshape((7, 7, 256))(x)

        inputs = layers.Concatenate(axis=3)([x, embedded_id])

        x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)

        model = Model(name=self.model_name, inputs=[z, class_id], outputs=x)
        return model


class LatentToImageCifar10CConditionalGenerator(model.Model):

    def __init__(
            self,
            model_parameters: edict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        z = Input(shape=[self.model_parameters.latent_size])
        class_id = Input(shape=[1])

        embedded_id = layers.Embedding(input_dim=10, output_dim=50)(class_id)
        embedded_id = layers.Dense(units=8 * 8)(embedded_id)
        embedded_id = layers.Reshape(target_shape=(8, 8, 1))(embedded_id)

        x = layers.Dense(units=8 * 8 * 256, use_bias=False)(z)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Reshape((8, 8, 256))(x)

        inputs = layers.Concatenate(axis=3)([x, embedded_id])

        x = layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(3, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')(x)

        model = Model(name=self.model_name, inputs=[z, class_id], outputs=x)
        return model


class LatentToImageNNUpsamplingCifar10CConditionalGenerator(model.Model):

    def __init__(
            self,
            model_parameters: edict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        z = Input(shape=[self.model_parameters.latent_size])
        class_id = Input(shape=[1])

        embedded_id = layers.Embedding(input_dim=10, output_dim=50)(class_id)
        embedded_id = layers.Dense(units=8 * 8)(embedded_id)
        embedded_id = layers.Reshape(target_shape=(8, 8, 1))(embedded_id)

        x = layers.Dense(units=8 * 8 * 256, use_bias=False)(z)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Reshape((8, 8, 256))(x)

        inputs = layers.Concatenate(axis=3)([x, embedded_id])

        x = layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.UpSampling2D()(x)

        x = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.UpSampling2D()(x)

        x = layers.Conv2D(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')(x)

        model = Model(name=self.model_name, inputs=[z, class_id], outputs=x)
        return model


class LatentToImageNNUpSamplingConditionalGenerator(model.Model):

    def __init__(
            self,
            model_parameters: edict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        z = Input(shape=[self.model_parameters.latent_size])
        class_id = Input(shape=[1])

        embedded_id = layers.Embedding(input_dim=10, output_dim=50)(class_id)
        embedded_id = layers.Dense(units=7 * 7)(embedded_id)
        embedded_id = layers.Reshape(target_shape=(7, 7, 1))(embedded_id)

        x = layers.Dense(units=7 * 7 * 256, use_bias=False)(z)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Reshape((7, 7, 256))(x)

        inputs = layers.Concatenate(axis=3)([x, embedded_id])

        x = layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.UpSampling2D()(x)

        x = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.UpSampling2D()(x)

        x = layers.Conv2D(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')(x)

        model = Model(name=self.model_name, inputs=[z, class_id], outputs=x)
        return model
