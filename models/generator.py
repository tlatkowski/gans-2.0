from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Conv2DTranspose
from keras.models import Input, Model


class Generator:
    
    def __init__(self, hidden_size):
        self._model = create_model(hidden_size)
    
    def __call__(self, inputs, **kwargs):
        return self._model(inputs)


def create_model(hidden_size):
    z = Input(shape=[hidden_size])
    
    x = Dense(units=7 * 7 * 256, use_bias=False)(z)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Reshape((7, 7, 256))(x)
    x = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                        activation='tanh')(x)
    
    model = Model(name='Generator', inputs=z, outputs=x)
    return model
