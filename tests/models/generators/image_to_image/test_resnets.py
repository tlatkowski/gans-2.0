import tensorflow as tf
from easydict import EasyDict as edict

from gans.models.generators.image_to_image import resnets


class TestResNets(tf.test.TestCase):

    def test_single_scale_generator_output_shape(self):
        model_parameters = edict({
            'img_height':       4,
            'img_width':        4,
            'num_channels':     3,
            'has_input_images': True,
        })
        g = resnets.SingleScaleGenerator(model_parameters)
        x = tf.ones(shape=[4, 4, 4, 3])
        z = tf.ones(shape=[4, 4, 4, 3])
        output_img = g([x, z])

        actual_shape = output_img.shape
        expected_shape = (4, 4, 4, 3)
        self.assertEqual(actual_shape, expected_shape)

    def test_build_progressive_generators(self):
        generators = resnets.build_progressive_generators(
            start_dim=(4, 4, 3),
            num_scales=4,
            r=2,
        )
        raise NotImplementedError

    def test_build_discriminators(self):
        discriminators = resnets.build_patch_discriminators(
            start_dim=(4, 4, 3),
            num_scales=4,
            r=2,
        )
        raise NotImplementedError
