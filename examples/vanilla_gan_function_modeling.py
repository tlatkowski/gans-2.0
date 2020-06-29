import tensorflow as tf
from easydict import EasyDict as edict
from tensorflow.python import keras
from tensorflow.python.keras import layers

from gans.models import sequential
from gans.models.gans import vanilla_gan
from gans.trainers import vanilla_gan_trainer


def squared_function(x):
    return x * x


def generate_samples(n=100):
    X1 = tf.random.uniform(shape=[50000]) - 0.5
    X2 = X1 * X1
    DATA = tf.stack([X1, X2], axis=1)
    from matplotlib import pyplot as plt
    plt.scatter(DATA[:, 0], DATA[:, 1])
    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(DATA).shuffle(
        100).batch(
        256)
    return train_dataset


train_dataset = generate_samples()

model_parameters = edict({
    'img_height':                  28,
    'img_width':                   28,
    'num_channels':                1,
    'batch_size':                  256,
    'num_epochs':                  10000,
    'buffer_size':                 10000,
    'hidden_size':                 100,
    'learning_rate_generator':     0.0001,
    'learning_rate_discriminator': 0.0001,
    'save_images_every_n_steps':   10000
})

generator = sequential.SequentialModel(
    layers=[
        keras.Input(shape=[5]),
        layers.Dense(units=15),
        layers.LeakyReLU(alpha=0.2),
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

generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=model_parameters.learning_rate_generator,
    beta_1=0.5,
)
discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=model_parameters.learning_rate_discriminator,
    beta_1=0.5,
)

gan_trainer = vanilla_gan_trainer.VanillaGANTrainer(
    batch_size=model_parameters.batch_size,
    generator=generator,
    discriminator=discriminator,
    dataset_type='VANILLA_MNIST_UNIFORM',
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    continue_training=False,
    save_images_every_n_steps=model_parameters.save_images_every_n_steps,
)
vanilla_gan_model = vanilla_gan.VanillaGAN(
    model_parameters=model_parameters,
    generator=generator,
    discriminator=discriminator,
    gan_trainer=gan_trainer,
)
# dataset = mnist.MnistDataset(model_parameters)

vanilla_gan_model.fit(train_dataset)
