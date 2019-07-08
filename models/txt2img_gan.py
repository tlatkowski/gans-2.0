import os

import tensorflow as tf

from models import discriminator, generators


class Text2ImageGAN:
    
    def __init__(self):
        
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



