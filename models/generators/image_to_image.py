import tensorflow_addons as tfa
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras import layers


class CycleGenerator:
    
    def __init__(
            self,
    ):
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
            filters=64,
            kernel_size=(7, 7),
            padding='same',
            use_bias=False,
        )(input_images)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=False,
        )(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(
            filters=256,
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
        n_resnet = 6
        for _ in range(n_resnet):
            x = resnet_block(256, x)
            
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.UpSampling2D()(x)

        x = layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(
            filters=32,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(
            filters=3,
            kernel_size=(7, 7),
            strides=(1, 1),
            padding='same',
            use_bias=False,
            activation='tanh',
        )(x)
        
        model = Model(name='Generator', inputs=input_images, outputs=x)
        return model


def resnet_block(n_filters, input_layer):
    g = layers.Conv2D(
        filters=n_filters,
        kernel_size=(3, 3),
        padding='same',
    )(input_layer)
    g = tfa.layers.InstanceNormalization()(g)
    g = layers.ReLU()(g)
    g = layers.Conv2D(
        filters=n_filters,
        kernel_size=(3, 3),
        padding='same',
    )(g)
    g = tfa.layers.InstanceNormalization()(g)
    # concatenate merge channel-wise with input layer
    g = layers.Concatenate()([g, input_layer])
    return g
