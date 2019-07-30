import tensorflow as tf

from models import vanilla_gan


class TestModels(tf.test.TestCase):
    
    def testRandom2ImageGAN(self):
        model = vanilla_gan.VanillaGAN()
        model.fit()
        raise NotImplementedError
