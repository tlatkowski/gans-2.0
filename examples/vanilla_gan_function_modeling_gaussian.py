import tensorflow as tf
from easydict import EasyDict as edict
from tensorflow.python import keras
from tensorflow.python.keras import layers

from gans.callbacks import saver
from gans.models import sequential
from gans.trainers import optimizers
from gans.trainers import vanilla_gan_trainer

model_parameters = edict({
    'batch_size':                  256,
    'num_epochs':                  15,
    'buffer_size':                 100000,
    'latent_size':                 5,
    'learning_rate_generator':     0.0002,
    'learning_rate_discriminator': 0.0002,
    'save_images_every_n_steps':   20
})


def generate_samples(num_samples):
    x = tf.random.uniform(shape=[num_samples], minval=-4, maxval=4)
    y = tf.exp(-0.5 * x ** 2)
    data = tf.stack([x, y], axis=1)
    return tf.data.Dataset. \
        from_tensor_slices(data). \
        shuffle(model_parameters.buffer_size). \
        batch(model_parameters.batch_size)


dataset = generate_samples(num_samples=500000)


def validation_dataset():
    return tf.random.normal([model_parameters.batch_size, model_parameters.latent_size])


validation_dataset = validation_dataset()

generator = sequential.SequentialModel(
    layers=[
        keras.Input(shape=[model_parameters.latent_size]),
        layers.Dense(units=15),
        layers.ELU(),
        layers.Dense(units=2, activation='linear'),
    ]
)

discriminator = sequential.SequentialModel(
    [
        keras.Input(shape=[2]),
        layers.Dense(units=25, activation='relu'),
        layers.Dense(units=2, activation='sigmoid'),
    ]
)

generator_optimizer = optimizers.Adam(
    learning_rate=model_parameters.learning_rate_generator,
    beta_1=0.5,
)
discriminator_optimizer = optimizers.Adam(
    learning_rate=model_parameters.learning_rate_discriminator,
    beta_1=0.5,
)

callbacks = [
    saver.FunctionProblemSaver(
        save_images_every_n_steps=model_parameters.save_images_every_n_steps,
    )
]

gan_trainer = vanilla_gan_trainer.VanillaGANTrainer(
    batch_size=model_parameters.batch_size,
    generator=generator,
    discriminator=discriminator,
    training_name='VANILLA_GAN_MODEL_GAUSSIAN',
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    latent_size=model_parameters.latent_size,
    continue_training=False,
    save_images_every_n_steps=model_parameters.save_images_every_n_steps,
    validation_dataset=validation_dataset,
    callbacks=callbacks,
)

gan_trainer.train(
    dataset=dataset,
    num_epochs=model_parameters.num_epochs,
)
