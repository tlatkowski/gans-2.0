import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    # real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    # return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output,
    #                                            from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)
