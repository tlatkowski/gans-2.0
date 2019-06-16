import os

import matplotlib.pyplot as plt
import tensorflow as tf

from data_loaders import mnist
from layers import losses
from models import discriminator, generators

hidden_size = 100

img_height = 28
img_width = 28
num_channels = 1

g = generators.RandomToImageGenerator(hidden_size)
z = tf.random.normal(shape=[16, 100])

generated_image = g(z)
# for i in range(generated_image.shape[0]):
#     plt.subplot(4, 4, i + 1)
#     plt.imshow(generated_image[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
#     plt.axis('off')

# plt.show()


d = discriminator.Discriminator(img_height, img_width, num_channels)
decision = d(generated_image)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

BATCH_SIZE = 200


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = g(noise, training=True)
        
        real_output = d(images, training=True)
        fake_output = d(generated_images, training=True)
        
        gen_loss = losses.generator_loss(fake_output)
        disc_loss = losses.discriminator_loss(real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, g._model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, d._model.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, g._model.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, d._model.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    
    # fig = plt.figure(figsize=(4, 4))
    print('ok')
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()
    print('no ok')


def train(dataset, epochs):
    for epoch in range(epochs):
        print(epoch)
        for image_batch in dataset:
            train_step(image_batch)
        
        generate_and_save_images(g,
                                 epoch + 1,
                                 seed)
        
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            # checkpoint.save(file_prefix=checkpoint_prefix)
            print('ok')


dataset = mnist.load_data()
train(dataset, EPOCHS)
