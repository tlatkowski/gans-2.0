import tensorflow as tf
import matplotlib.pyplot as plt
from layers import losses


class GAN:
    
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.g = self.generator
        self.d = self.discriminator
    
    def train(self, dataset, epochs):
        for epoch in range(epochs):
            print(epoch)
            for image_batch in dataset:
                self.train_step(image_batch)
        
            generate_and_save_images(g, epoch + 1, seed)
        
            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                # checkpoint.save(file_prefix=checkpoint_prefix)
                print('ok')
    
    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, noise_dim])
    
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.g(noise, training=True)
        
            real_output = self.d(images, training=True)
            fake_output = self.d(generated_images, training=True)
        
            gen_loss = losses.generator_loss(fake_output)
            disc_loss = losses.discriminator_loss(real_output, fake_output)
    
        gradients_of_generator = gen_tape.gradient(gen_loss, g._model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, d._model.trainable_variables)
    
        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.g._model.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.d._model.trainable_variables))
        
    @property
    def discriminator(self):
        raise NotImplementedError

    @property
    def generator(self):
        raise NotImplementedError


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