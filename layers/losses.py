import numpy as np
import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def cycle_loss(real_image, cycled_image, weight=60):
    loss = l1_loss(real_image, cycled_image)
    return weight * loss


def identity_loss(real_image, same_image, weight=5):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return weight * loss


def l1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))


def wasserstein_loss(true_output, predicted_output):
    return tf.reduce_mean(true_output * predicted_output)


def gradient_penalty_loss(predicted_output, averaged_samples):
    gradient_penalty_weight = 10
    gradients = tf.gradients(predicted_output, averaged_samples)[0]
    gradients_sqr = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(
        gradients_sqr,
        axis=np.arange(1, len(gradients_sqr.shape))
    )
    gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * tf.square(1 - gradient_l2_norm)
    return tf.reduce_mean(gradient_penalty)
