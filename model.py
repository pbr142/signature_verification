
from data import get_data_dataframe
from functools import partial
from tensorflow import keras

import matplotlib.pyplot as plt

class PlotReconstruction(keras.callbacks.Callback):
    def __init__(self, generator, num_images=5, **kwargs):
        self.generator = generator
        self.num_images = num_images
        super(PlotReconstruction, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs):
        x_batch, y_batch = self.generator.next()
        x_batch = x_batch[:self.num_images]
        reconstruct_batch = self.model.predict(x_batch)
        x_batch *= 255.
        reconstruct_batch *= 255.

        fig = plt.figure(figsize=(2*self.num_images, 3))
        num_images = min(len(x_batch), self.num_images)
        for i in range(num_images):
            plt.subplot(2, self.num_images, i+1)
            plt.imshow(x_batch[i], cmap='binary')
            plt.subplot(2, self.num_images, i+self.num_images+1)
            plt.imshow(reconstruct_batch[i], cmap='binary')
        plt.show();

def make_conv_ae(image_sizes):
    conv_layer = partial(keras.layers.Conv2D, kernel_size=(3,3), activation='selu', padding='same')
    conv_t_layer = partial(keras.layers.Conv2DTranspose, kernel_size=(3,3), strides=2, activation='selu', padding='same')

    encoder = keras.Sequential([
        conv_layer(filters=32, input_shape=(*image_sizes, 1)),
        keras.layers.MaxPooling2D(),
        conv_layer(32),
        keras.layers.MaxPooling2D(),
        conv_layer(64),
        keras.layers.MaxPooling2D(),
        conv_layer(64),
        keras.layers.MaxPooling2D(),
        conv_layer(128),
        keras.layers.MaxPooling2D(),
        conv_layer(128),
        keras.layers.MaxPooling2D()
    ])

    decoder = keras.Sequential([
        conv_t_layer(128, input_shape=encoder.output_shape[1:]),
        conv_t_layer(128),
        conv_t_layer(64),
        conv_t_layer(64),
        conv_t_layer(32),
        conv_t_layer(1, activation='sigmoid')
    ])

    conv_ae = keras.models.Sequential([encoder, decoder])

    return encoder, decoder, conv_ae