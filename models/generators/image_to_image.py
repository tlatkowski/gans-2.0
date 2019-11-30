from easydict import EasyDict as edict
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras import layers


class RandomToImageCifar10Generator:
    
    def __init__(
            self,
            input_params: edict,
    ):
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
        input_images = Input(shape=[256, 256, 3])
        
        x = layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=False,
        )(input_images)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=False,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=False,
        )(x)
        
        x = layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=False,
        )(x)
        
        
        
        model = Model(name='Generator', inputs=input_images, outputs=x)
        return model
