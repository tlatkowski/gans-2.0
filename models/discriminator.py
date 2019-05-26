from keras.layers import Dense, LeakyReLU, Flatten, Dropout, Conv2D
from keras.models import Input, Model


class Discriminator:
    
    def __init__(self, img_height, img_width, num_channels):
        self._model = create_model(img_height, img_width, num_channels)
    
    def __call__(self, *args, **kwargs):
        self._model(args)


def create_model(img_height, img_width, num_channels):
    input_img = Input(shape=(img_height, img_width, num_channels))
    
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same')(input_img)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.3)(x)
    
    x = Flatten()(x)
    x = Dense(units=1)(x)
    
    model = Model(name='discriminator', inputs=input_img, outputs=x)
    return model
