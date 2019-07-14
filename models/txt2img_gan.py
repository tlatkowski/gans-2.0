import tensorflow as tf

from data_loaders import coco
from models import discriminators, generators
from models.gan_trainer import GANTrainer


class Text2ImageGAN:
    
    def __init__(self):
        hidden_size = 100
        
        img_height = 28
        img_width = 28
        num_channels = 1
        max_sentence_length = 50
        embedding_size = 64
        self.generator = generators.TextToImageGenerator(max_sequence_length=max_sentence_length,
                                                         embedding_size=embedding_size)
        z = tf.random.normal(shape=[16, hidden_size])
        
        generated_image = self.generator(z)
        
        self.discriminator = discriminators.Discriminator(img_height, img_width, num_channels)
        decision = self.discriminator(generated_image)
        
        self.batch_size = 4
        self.num_epochs = 10
    
    def fit(self):
        dataset = coco.load_data(self.batch_size)
        gan_trainer = GANTrainer(self.batch_size, self.generator, self.discriminator)
        gan_trainer.train(dataset, self.num_epochs)
