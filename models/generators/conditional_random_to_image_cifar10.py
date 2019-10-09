from easydict import EasyDict as edict
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras import layers


class RandomToImageCifar10CConditionalGenerator:
    
    def __init__(self, input_params: edict):
        self.hidden_size = input_params.hidden_size
        self._model = self.create_model()
    
    def __call__(self, inputs, **kwargs):
        return self._model(inputs=inputs, **kwargs)
    
    @property
    def trainable_variables(self):
        return self._model.trainable_variables
    
    @property
    def model(self):
        return self._model
    
    @property
    def num_channels(self):
        return self._model.output_shape[-1]
    
    def create_model(self):
        z = Input(shape=[self.hidden_size])
        class_id = Input(shape=[1])
        
        embedded_id = layers.Embedding(input_dim=10, output_dim=50)(class_id)
        embedded_id = layers.Dense(units=8 * 8)(embedded_id)
        embedded_id = layers.Reshape(target_shape=(8, 8, 1))(embedded_id)
        
        x = layers.Dense(units=8 * 8 * 256, use_bias=False)(z)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Reshape((8, 8, 256))(x)
        
        inputs = layers.Concatenate(axis=3)([x, embedded_id])
        
        x = layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                   use_bias=False)(
            inputs)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        
        x = layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(
            x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        
        x = layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                   use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        
        x = layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(
            x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        
        x = layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(
            x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        
        x = layers.Conv2D(3, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False,
                          activation='tanh')(x)
        
        model = Model(name='Generator', inputs=[z, class_id], outputs=x)
        return model
