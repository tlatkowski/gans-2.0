from easydict import EasyDict as edict
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras import layers


class Discriminator:
    
    def __init__(self, input_params: edict):
        self.img_height = input_params.img_height
        self.img_width = input_params.img_width
        self.num_channels = input_params.num_channels
        self._model = self.create_model()
    
    def __call__(self, inputs, **kwargs):
        return self._model(inputs=inputs, **kwargs)
    
    @property
    def trainable_variables(self):
        return self._model.trainable_variables
    
    @property
    def model(self):
        return self._model
    
    def create_model(self):
        input_img = Input(shape=(self.img_height, self.img_width, self.num_channels))
        
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
