import tensorflow as tf

from models import vanilla_gan
from easydict import EasyDict as edict
from utils import dataset_utils

class TestModels(tf.test.TestCase):
    
    def testVanillaGAN(self):
        input_parmas = edict({
            'batch_size': 4,
            'num_epochs': 1,
            'hidden_size': 100,
            'img_height': 28,
            'img_width': 28,
            'num_channels': 3
        })
        dataset_type = dataset_utils.ProblemType.VANILLA_MNIST
        model = vanilla_gan.VanillaGAN(input_parmas, dataset_type)
        model.fit()
        raise NotImplementedError
