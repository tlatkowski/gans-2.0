from tensorflow.python.keras import Input, Model
from tensorflow.python.keras import layers


class RandomToImageGenerator:
    
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self._model = self.create_model()
    
    def __call__(self, inputs, **kwargs):
        return self._model(inputs)
    
    def create_model(self):
        z = Input(shape=[self.hidden_size])
        
        x = layers.Dense(units=7 * 7 * 256, use_bias=False)(z)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.Reshape((7, 7, 256))(x)
        x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                   activation='tanh')(x)
        
        model = Model(name='Generator', inputs=z, outputs=x)
        return model


class TextToImageGenerator:
    
    def __init__(self, max_sequence_length, embedding_size):
        self.max_sequence_length = max_sequence_length
        self.embedding_size = embedding_size
        self._model = self.create_model()
    
    def __call__(self, inputs, **kwargs):
        return self._model(inputs)
    
    def create_model(self):
        layers.Dot
        inputs = Input(shape=[self.max_sequence_length, self.embedding_size])
        # inputs_transposed = layers.Permute(dims=(2, 1))(inputs)
        # query_key = layers.Dot(axes=[1, 2])([inputs, inputs_transposed])
        query_key = layers.Dot(axes=2)([inputs, inputs])
        attentions = layers.Softmax(axis=-1)(query_key)  # TODO test it
        qkv = layers.Dot(axes=1)([attentions, inputs])
        raise NotImplementedError
