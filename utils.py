import matplotlib.pyplot as plt


def generate_and_save_images(generator_model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = generator_model(test_input, training=False)
    
    # fig = plt.figure(figsize=(4, 4))
    print('ok')
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()
    print('no ok')


def plot_image_grid(generated_image):
    for i in range(generated_image.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_image[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.show()
