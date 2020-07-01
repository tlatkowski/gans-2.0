import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict
from tensorflow.python import keras
from tensorflow.python.keras import Input
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers

from gans.models import custom_model
from gans.models.gans import conditional_gan
from gans.trainers import conditional_gan_trainer

model_parameters = edict({
    'batch_size':                  256,
    'num_epochs':                  1000,
    'buffer_size':                 1000000,
    'latent_size':                 10,
    'num_classes':                 2,
    'learning_rate_generator':     0.0001,
    'learning_rate_discriminator': 0.0001,
    'save_images_every_n_steps':   1000
})


def generator_model():
    z = Input(shape=[model_parameters.latent_size])
    class_id = Input(shape=[1])

    embedded_id = layers.Embedding(input_dim=model_parameters.num_classes, output_dim=5)(class_id)
    embedded_id = layers.Dense(units=7 * 7)(embedded_id)
    embedded_id = layers.Reshape(target_shape=(49,))(embedded_id)

    x = layers.Dense(units=7 * 7, use_bias=False)(z)
    x = layers.ReLU()(x)

    inputs = layers.Concatenate(axis=1)([x, embedded_id])
    x = layers.ELU()(inputs)
    x = layers.Dense(units=2, activation='linear')(x)

    model = Model(name='generator', inputs=[z, class_id], outputs=x)
    return model


generator = custom_model.CustomModel(fn=generator_model)


def discriminator_model():
    z = keras.Input(shape=[2])
    class_id = Input(shape=[1])

    embedded_id = layers.Embedding(input_dim=model_parameters.num_classes, output_dim=5)(class_id)
    embedded_id = layers.Dense(units=7 * 7)(embedded_id)
    embedded_id = layers.Reshape(target_shape=(49,))(embedded_id)

    x = layers.Dense(units=7 * 7, use_bias=False)(z)
    x = layers.LeakyReLU()(x)

    inputs = layers.Concatenate(axis=1)([x, embedded_id])
    x = layers.ELU()(inputs)
    x = layers.Dense(units=2, activation='sigmoid')(x)

    model = Model(name='discriminator', inputs=[z, class_id], outputs=x)
    return model


discriminator = custom_model.CustomModel(fn=discriminator_model)

generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=model_parameters.learning_rate_generator,
    beta_1=0.5,
)
discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=model_parameters.learning_rate_discriminator,
    beta_1=0.5,
)

gan_trainer = conditional_gan_trainer.ConditionalGANTrainer(
    batch_size=model_parameters.batch_size,
    generator=generator,
    discriminator=discriminator,
    dataset_type='CONDITIONAL_GAN_MODEL_FUNCTION',
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    latent_size=model_parameters.latent_size,
    num_classes=model_parameters.num_classes,
    continue_training=False,
    save_images_every_n_steps=model_parameters.save_images_every_n_steps,
)
vanilla_gan_model = conditional_gan.ConditionalGAN(
    model_parameters=model_parameters,
    generator=generator,
    discriminator=discriminator,
    gan_trainer=gan_trainer,
)


def generate_samples(num_samples):
    x = tf.random.uniform(shape=[num_samples], minval=-2, maxval=2)
    # y1 = x * x * x
    # y2 = -(x * x * x)
    x1 = x + 5
    y3 = x * x
    y4 = x1 * x1
    labels = np.repeat(list(range(model_parameters.num_classes)), num_samples)

    # data1 = tf.stack([x, y1], axis=1)
    # data2 = tf.stack([x, y2], axis=1)
    data3 = tf.stack([x, y3], axis=1)
    data4 = tf.stack([x1, y4], axis=1)
    # data = tf.concat([data1, data2, data3, data4], axis=0)
    data = tf.concat([data3, data4], axis=0)
    # data = tf.stack([data, labels], axis=1)
    return tf.data.Dataset. \
        from_tensor_slices((data, labels)). \
        shuffle(model_parameters.buffer_size). \
        batch(model_parameters.batch_size)


train_dataset = generate_samples(num_samples=50000)

vanilla_gan_model.fit(train_dataset)
