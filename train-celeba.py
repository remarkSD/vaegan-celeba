from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from models_celebA import *
from celebA_loader import *



def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_DCNN_celebA"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    Arguments:
        models (list): encoder and decoder models
        data (list): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function

    Returns:
        none
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)
    '''
    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()
    '''
    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of the digits
    n = 1
    digit_size = 64
    figure = np.zeros((digit_size * n, digit_size * n,3))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]


    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
#            z_sample = np.array([[xi, yi]])
            z_sample = np.random.uniform(size=(2,128))

            #print(z_sample.shape)
            x_decoded = decoder.predict(z_sample)
            print(x_decoded.shape)

            '''
            digit = x_decoded[0].reshape(digit_size, digit_size,3)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size,:] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()
    '''
image_size = 64
channels = 3
# network parameters
input_shape = (image_size, image_size, channels)
batch_size = 64
kernel_size = 5
filters = np.array([64,32])
z_dim = 2048
epochs = 10
dir='/home/airscan-razer04/Documents/datasets/img_align_celeba/'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    args = parser.parse_args()
    df_dim = 64
    height = np.array([64])
    width = np.array([64])

    inputs = Input(shape=input_shape)

    vaegan_encoder = encoder(num_filters=filters[0], ch=channels,
                                rows=height, cols=width, z_dim=z_dim, kernel_size=kernel_size)
    vaegan_decoder = generator(num_filters=filters[1], ch=channels,
                                z_dim=z_dim, kernel_size=kernel_size)
    vaegan_encoder.summary()
    vaegan_decoder.summary()

    models = (vaegan_encoder, vaegan_decoder)
    #data = (x_test, y_test)
    outputs = vaegan_decoder(vaegan_encoder(inputs)[2])
    print("outputshape", outputs.shape)
    vae = Model(inputs, outputs, name='vae')
    #vaegan_disc = discriminator(num_filters=32, z_dim=z_dim, rows=height, cols=width)
    #vaegan_disc.compile(optimizer='RMSProp', loss='binary_crossentropy')

    #vaegan_disc.summary()
    if args.mse:
        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    else:
        reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                                K.flatten(outputs))

    reconstruction_loss *= image_size * image_size * channels
    kl_loss = 1 + vaegan_encoder(inputs)[1] - K.square(vaegan_encoder(inputs)[0]) - K.exp(vaegan_encoder(inputs)[1])
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    vae.add_loss(vae_loss)
    rmsprop = optimizers.rmsprop(lr=0.0003)
    vae.compile(optimizer=rmsprop)
    vae.summary()
    #print(vae)
    plot_model(vae, to_file='vae_dcnn.png', show_shapes=True)

    if args.weights:
        #print("loading weights",args.weights)
        vae.load_weights(args.weights)
        #print(vae)
    else:
        checkpoint_period = 1
        checkpoint_path = 'checkpoints/'
        checkpointer = ModelCheckpoint(filepath=checkpoint_path + 'model-{epoch:05d}.hdf5',
                                        verbose=1,
                                        save_weights_only=True,
                                        period=checkpoint_period)
        #vae.load_weights('checkpoints/model-00340.hdf5')
        vae.fit_generator(celeb_loader(dir=dir,
                            randomize=True,
                            batch_size=batch_size,
                            height=image_size,
                            width=image_size,
                            norm=True),
                #epochs=1,
                #steps_per_epoch=1

                epochs=epochs,
                steps_per_epoch=int(202599/batch_size),
                callbacks=[checkpointer]
                #validation_data=(x_test, None)
                )
        vae.save_weights('vae_celebA_2048lat.h5')

    num_outputs = 25
    #z_sample = np.random.uniform(size=(num_outputs,z_dim), low=-3.0, high=3.0)
    z_sample = np.random.normal(size=(num_outputs,z_dim))
    out_random = vaegan_decoder.predict(z_sample)
    out_random = (out_random + 1)*127.5
    out_random = out_random.astype(np.uint8)
    for i in range (out_random.shape[0]):
        cv2.imshow("image from noise",out_random[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #print(vae)
    z_sample = np.random.uniform(size=(25,z_dim), low=-1.0, high=1.0)
    z_sample = np.random.normal(size=(num_outputs,z_dim))
    out_random = vaegan_decoder.predict(z_sample)
    # Unnormalize samples
    out_random = (out_random + 1)*127.5
    out_random = out_random.astype(np.uint8)
    print(out_random.shape  )
    print("MAX", np.max(out_random))
    print("MIN", np.min(out_random))

    # Put samples in grid
    fig = np.zeros((64*5,64*5,3))
    for k1 in range (5):
        for k2 in range (5):
            fig[64*k2:64*(k2+1),64*k1:64*(k1+1),:] = out_random[k1*5+k2]
    #cv2.imshow("image",out_random[0])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Write samples
    out_filename = 'vae_images/out.jpg'
    cv2.imwrite(out_filename, fig)


    some_gen = celeb_loader(batch_size=25,
                            dir=dir,
                            randomize=True,
                            height=image_size,
                            width=image_size,
                            norm=True)

    data, _ = next(some_gen)

    out_enc = vaegan_encoder.predict(data)
    print("MAX", np.max(out_enc))
    print("MIN", np.min(out_enc))
    #out = vaegan_decoder.predict(out_enc[2])

    out = vae.predict(data)
    out = (out + 1)*127.5
    out = out.astype(np.uint8)
    print("data", data.shape)
    print("out", out.shape)
    '''
    data = (data+1)*127.5
    data = data.astype(np.uint8)
    for i in range (data.shape[0]):
        cv2.imshow("image",data[i])
        cv2.waitKey(0)


        cv2.imshow("out image",out[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    '''
