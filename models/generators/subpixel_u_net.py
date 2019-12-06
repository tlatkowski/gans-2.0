import tensorflow as tf
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
        # x5 = PS(x4, r=2, batch_size=4)
        x5 = layers.Concatenate()([x5, x3])
        
        x5 = layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x5)
        x5 = tfa.layers.InstanceNormalization()(x5)
        x5 = layers.LeakyReLU(alpha=0.2)(x5)
        
        x6 = layers.UpSampling2D()(x5)
        # x6 = PS(x5, r=2, batch_size=4)

        x6 = layers.Concatenate()([x6, x2])
        
        x6 = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x6)
        x6 = tfa.layers.InstanceNormalization()(x6)
        x6 = layers.LeakyReLU(alpha=0.2)(x6)
        
        x7 = layers.UpSampling2D()(x6)
        # x7 = PS(x6, r=2, batch_size=4)
        x7 = layers.Concatenate()([x7, x1])
        x7 = layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x7)
        x7 = tfa.layers.InstanceNormalization()(x7)
        x7 = layers.LeakyReLU(alpha=0.2)(x7)
        
        x8 = PS(x7, r=2, batch_size=4)
        # x8 = layers.UpSampling2D()(x7)
        x8 = layers.Concatenate()([x8, input_images])
        x8 = layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        )(x8)
        x8 = tfa.layers.InstanceNormalization()(x8)
        x8 = layers.LeakyReLU(alpha=0.2)(x8)
        
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


def subpixel_layer(x, r, batch_size):
    _, h, w, c = x.get_shape().as_list()
    # X = tf.reshape(x, (b, h, w, r, r))
    # X = tf.reshape(x, (batch_size, h, w, r, r))
    X = tf.reshape(x, (-1, h, w, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(axis=1, num_or_size_splits=h, value=X)  # a, [bsize, b, r, r]
    X = tf.concat(axis=2, values=[tf.squeeze(x, axis=1) for x in X])  # bsize, b, a*r, r
    X = tf.split(axis=1, num_or_size_splits=w, value=X)  # b, [bsize, a*r, r]
    X = tf.concat(axis=2, values=[tf.squeeze(x, axis=1) for x in X])  # bsize, a*r, b*r
    return tf.reshape(X, (-1, h * r, w * r, 1))


def PS(X, r, batch_size):
    _, _, _, c = X.get_shape().as_list()
    size_split = int(c/(r*r))
    Xc = tf.split(axis=3, num_or_size_splits=size_split, value=X)
    X = tf.concat(axis=3, values=[subpixel_layer(x, r, batch_size) for x in Xc])
    return X
