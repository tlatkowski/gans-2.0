import tensorflow as tf

from models import vanilla_gan
from models import txt2img_gan


class TestModels(tf.test.TestCase):
    
    def testRandom2ImageGAN(self):
        model = vanilla_gan.Random2ImageGAN()
        model.fit()
        raise NotImplementedError
    
    def testText2ImageGAN(self):
        model = txt2img_gan.Text2ImageGAN()
        model.fit()
        raise NotImplementedError

