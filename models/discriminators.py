from tensorflow.python.keras import Input, Model
from tensorflow.python.keras import layers


class Discriminator:
    
    def __init__(self, img_height, img_width, num_channels):
        self._model = create_model(img_height, img_width, num_channels)
    
    def __call__(self, inputs, **kwargs):
        return self._model(inputs=inputs, **kwargs)
    
    @property
    def trainable_vars(self):
        return self._model.trainable_variables


def create_model(img_height, img_width, num_channels):
    input_img = Input(shape=(img_height, img_width, num_channels))
    
    x = layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same')(input_img)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.3)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(units=1)(x)
    
    model = Model(name='discriminator', inputs=input_img, outputs=x)
    
    return model
