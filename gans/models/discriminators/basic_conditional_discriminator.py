from easydict import EasyDict as edict
from tensorflow.python.keras import Input
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers

from gans.models import model


class ConditionalDiscriminator(model.Model):

    def __init__(
            self,
            model_parameters: edict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        input_img = Input(shape=[
            self.model_parameters.img_height,
            self.model_parameters.img_width,
            self.model_parameters.num_channels
        ])
        class_id = Input(shape=[1])

        embedded_id = layers.Embedding(input_dim=10, output_dim=50)(class_id)
        embedded_id = layers.Dense(units=input_img.shape[1] * input_img.shape[2])(embedded_id)
        embedded_id = layers.Reshape(target_shape=(input_img.shape[1], input_img.shape[2], 1))(embedded_id)

        x = layers.Concatenate(axis=3)([input_img, embedded_id])

        x = layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(rate=0.3)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(units=1)(x)

        model = Model(name=self.model_name, inputs=[input_img, class_id], outputs=x)

        return model
