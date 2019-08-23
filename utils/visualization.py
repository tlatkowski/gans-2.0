import glob
import math
import os

import PIL
import imageio
import matplotlib.pyplot as plt
import numpy as np
from IPython import display

from utils import constants


def make_gif_from_images(path, anim_file='dcgan.gif'):
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(path, 'image*.png'))
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


def generate_and_save_images(generator_model, epoch, test_input, dataset_name, cmap=None,
                             num_examples_to_display=16):
    display.clear_output(wait=True)
    predictions = generator_model(test_input, training=False)
    if predictions.shape[0] < num_examples_to_display:
        raise ValueError("Input batch size cannot be less than number of example to display.")
    
    n = int(math.sqrt(num_examples_to_display))
    
    for i in range(num_examples_to_display):
        plt.subplot(n, n, i + 1)
        if generator_model.num_channels == 3:
            img_to_plot = predictions[i, :, :, :] * 127.5 + 127.5
        else:
            img_to_plot = predictions[i, :, :, 0] * 127.5 + 127.5
        plt.imshow(img_to_plot, cmap=cmap)
        plt.axis('off')
    
    save_path = os.path.join(constants.SAVE_IMAGE_DIR, dataset_name)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'image_at_epoch_{:04d}.png'.format(epoch)))
    im = np.asarray(
        PIL.Image.open(os.path.join(save_path, 'image_at_epoch_{:04d}.png'.format(epoch))))
    return im


def plot_image_grid(generated_image):
    for i in range(generated_image.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_image[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.show()
