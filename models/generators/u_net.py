import tensorflow_addons as tfa
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras import layers


class UNetGenerator:
    
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
        
        x1 = layers.Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='same',
            use_bias=False,
        )(input_images)
        x1 = tfa.layers.InstanceNormalization()(x1)
        x1 = layers.ReLU()(x1)
        
        x2 = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=False,
        )(x1)
        x2 = tfa.layers.InstanceNormalization()(x2)
        x2 = layers.ReLU()(x2)
        
        x3 = layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=False,
        )(x2)
        x3 = tfa.layers.InstanceNormalization()(x3)
        x3 = layers.ReLU()(x3)
        
        x4 = layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=False,
        )(x3)
        x4 = tfa.layers.InstanceNormalization()(x4)
        x4 = layers.ReLU()(x4)
        
        x5 = layers.UpSampling2D()(x4)
        x5 = layers.Concatenate()([x5, x3])
        
        x5 = layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x5)
        x5 = tfa.layers.InstanceNormalization()(x5)
        x5 = layers.ReLU()(x5)
        
        x6 = layers.UpSampling2D()(x5)
        x6 = layers.Concatenate()([x6, x2])

        x6 = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x6)
        x6 = tfa.layers.InstanceNormalization()(x6)
        x6 = layers.ReLU()(x6)
        
        x7 = layers.UpSampling2D()(x6)
        x7 = layers.Concatenate()([x7, x1])
        
        x7 = layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x7)
        x7 = tfa.layers.InstanceNormalization()(x7)
        x7 = layers.ReLU()(x7)
        x8 = layers.UpSampling2D()(x7)
        
        x8 = layers.Concatenate()([x8, input_images])

        x8 = layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x8)
        x8 = tfa.layers.InstanceNormalization()(x8)
        x8 = layers.ReLU()(x8)
        
        x9 = layers.Conv2D(
            filters=3,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            use_bias=False,
            activation='tanh',
        )(x8)
        
        model = Model(name='Generator', inputs=input_images, outputs=x9)
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