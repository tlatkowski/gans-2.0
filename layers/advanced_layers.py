import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras import layers


def residual_block(n_filters, input_layer):
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


def subpixel_layer(x, r):
    _, h, w, c = x.get_shape().as_list()
    X = tf.reshape(x, (-1, h, w, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(axis=1, num_or_size_splits=h, value=X)  # a, [bsize, b, r, r]
    X = tf.concat(axis=2, values=[tf.squeeze(x, axis=1) for x in X])  # bsize, b, a*r, r
    X = tf.split(axis=1, num_or_size_splits=w, value=X)  # b, [bsize, a*r, r]
    X = tf.concat(axis=2, values=[tf.squeeze(x, axis=1) for x in X])  # bsize, a*r, b*r
    return tf.reshape(X, (-1, h * r, w * r, 1))


def subpixel_upsampling(X, r):
    _, _, _, c = X.get_shape().as_list()
    size_split = int(c / (r * r))
    Xc = tf.split(axis=3, num_or_size_splits=size_split, value=X)
    X = tf.concat(axis=3, values=[subpixel_layer(x, r) for x in Xc])
    return X


def densely_connected_residual_block():
    raise NotImplementedError


def channel_attention_block():
    raise NotImplementedError
