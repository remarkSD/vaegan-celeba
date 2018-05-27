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
from keras.optimizers import RMSprop
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from models_celebA import *
from celebA_loader import *


#
# def plot_results(models,
#                  data,
#                  batch_size=128,
#                  model_name="vae_DCNN_celebA"):
#     """Plots labels and MNIST digits as function of 2-dim latent vector
#
#     Arguments:
#         models (list): encoder and decoder models
#         data (list): test data and label
#         batch_size (int): prediction batch size
#         model_name (string): which model is using this function
#
#     Returns:
#         none
#     """
#
#     encoder, decoder = models
#     x_test, y_test = data
#     os.makedirs(model_name, exist_ok=True)
#     '''
#     filename = os.path.join(model_name, "vae_mean.png")
#     # display a 2D plot of the digit classes in the latent space
#     z_mean, _, _ = encoder.predict(x_test,
#                                    batch_size=batch_size)
#     plt.figure(figsize=(12, 10))
#     plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
#     plt.colorbar()
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.savefig(filename)
#     plt.show()
#     '''
#     filename = os.path.join(model_name, "digits_over_latent.png")
#     # display a 30x30 2D manifold of the digits
#     n = 1
#     digit_size = 64
#     figure = np.zeros((digit_size * n, digit_size * n,3))
#     # linearly spaced coordinates corresponding to the 2D plot
#     # of digit classes in the latent space
#     grid_x = np.linspace(-4, 4, n)
#     grid_y = np.linspace(-4, 4, n)[::-1]
#
#
#     for i, yi in enumerate(grid_y):
#         for j, xi in enumerate(grid_x):
# #            z_sample = np.array([[xi, yi]])
#             z_sample = np.random.uniform(size=(2,128))
#
#             #print(z_sample.shape)
#             x_decoded = decoder.predict(z_sample)
#             print(x_decoded.shape)
#
#             '''
#             digit = x_decoded[0].reshape(digit_size, digit_size,3)
#             figure[i * digit_size: (i + 1) * digit_size,
#                    j * digit_size: (j + 1) * digit_size,:] = digit
#
#     plt.figure(figsize=(10, 10))
#     start_range = digit_size // 2
#     end_range = n * digit_size + start_range + 1
#     pixel_range = np.arange(start_range, end_range, digit_size)
#     sample_range_x = np.round(grid_x, 1)
#     sample_range_y = np.round(grid_y, 1)
#     plt.xticks(pixel_range, sample_range_x)
#     plt.yticks(pixel_range, sample_range_y)
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.imshow(figure, cmap='Greys_r')
#     plt.savefig(filename)
#     plt.show()
#     '''
#
def build_generator(inputs, image_size):
    """Build a Generator Model

    Stacks of BN-ReLU-Conv2DTranpose to generate fake images
    Output activation is sigmoid instead of tanh in [1].
    Sigmoid converges easily.

    # Arguments
        inputs (Layer): Input layer of the generator (the z-vector)
        image_size: Target size of one side (assuming square image)

    # Returns
        Model: Generator Model
    """
    print(inputs, image_size)
    image_resize = image_size // 4
    kernel_size = 5
    layer_filters = [128, 64, 32, 3]

    x = Dense(image_resize * image_resize * layer_filters[0])(inputs)
    print(x)
    x = Reshape((image_resize, image_resize, layer_filters[0]))(x)

    for filters in layer_filters:
        if filters > layer_filters[-2]:
            strides = 2
        else:
            strides = 1

        x = BN()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same')(x)

    x = Activation('sigmoid')(x)
    generator = Model(inputs, x, name='generator')
    return generator


def build_discriminator(inputs):
    """Build a Discriminator Model

    Stacks of LeakyReLU-Conv2D to discriminate real from fake
    The network does not converge with BN so it is not used here
    unlike in [1]

    # Arguments
        inputs (Layer): Input layer of the discriminator (the image)

    # Returns
        Model: Discriminator Model
    """
    kernel_size = 5
    layer_filters = [32, 64, 128, 256]

    x = inputs
    for filters in layer_filters:
        if filters == layer_filters[-1]:
            strides = 1
        else:
            strides = 2
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same')(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    discriminator = Model(inputs, x, name='discriminator')
    return discriminator

image_size = 64
channels = 3
# network parameters
input_shape = (image_size, image_size, channels)
batch_size = 64
kernel_size = (5,5)
filters = np.array([64,32])
z_dim = 128
epochs = 250
lr = 0.0003
decay = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    args = parser.parse_args()
    height = np.array([64])
    width = np.array([64])

    # Instantiate encoder, decoder/generator, discriminator models
    inputs = Input(shape=input_shape)

    #vaegan_encoder = encoder(num_filters=filters[0], ch=channels, rows=height, cols=width, z_dim=z_dim)

    vaegan_disc = discriminator(num_filters=32, z_dim=z_dim, ch=3, rows=height, cols=width)
    #vaegan_decoder = build_generator(Input(shape=(z_dim,)), image_size)
    #vaegan_disc = build_discriminator(inputs)
    #vaegan_encoder.summary()



    disc_optimizer = RMSprop(lr=lr)
    vaegan_disc.compile(loss='binary_crossentropy',
                        optimizer=disc_optimizer,
                        metrics=['accuracy'])
    vaegan_disc.summary()


    # Build Generator/Decoder
    vaegan_decoder = generator(num_filters=32, z_dim=z_dim, ch=channels)
    vaegan_decoder.summary()
    # Instantiate GAN model
    gan_input = Input(shape=(z_dim,))
    gan_output = vaegan_disc(vaegan_decoder(gan_input))
    print("gan_inputshape",  gan_input.shape)
    print("gan_outshape", gan_output.shape)

    gan_optimizer = RMSprop(lr=lr)
    vaegan_disc.trainable = False
    gan = Model(gan_input, gan_output, name='gan')
    gan.compile(loss='binary_crossentropy',
                optimizer=gan_optimizer,
                metrics=['accuracy'])
    gan.summary()

    if args.weights:
        #print("loading weights",args.weights)
        gan.load_weights(args.weights)
        #print(vae)

    '''
    checkpoint_period = 5
    checkpoint_path = 'checkpoints/'
    checkpointer = ModelCheckpoint(filepath=checkpoint_path + 'model-{epoch:05d}.hdf5',
                                    verbose=1,
                                    save_weights_only=True,
                                    period=checkpoint_period)
    #vae.load_weights('checkpoints/model-00340.hdf5')
    vae.fit_generator(celeb_loader(dir='/home/airscan-razer04/Documents/datasets/img_align_celeba/',
                        randomize=True,
                        batch_size=batch_size,
                        height=image_size,
                        width=image_size),
            #epochs=1,
            #steps_per_epoch=1

            epochs=epochs,
            steps_per_epoch=int(20599/batch_size),
            callbacks=[checkpointer]
            #validation_data=(x_test, None)
            )
    vae.save_weights('vae_dcnn_celebA-02.h5')
    '''
    save_interval = int(202599/batch_size)
    #epochs=1
    #save_interval=50
    img_loader = celeb_loader(dir='/home/raimarc/Documents/img_align_celeba/',
                            batch_size=batch_size, norm=True)

    z_sample = np.random.uniform(size=(25,z_dim), low=-1.0, high=1.0)
    for i in range (epochs):
        for j in range (int(save_interval)):
            # Load real images
            real_images, _ = next(img_loader)
            #x = np.concatenate((real_images, fake_images))
            #print(x.shape)
            # Label real and fake images
            y = np.ones([batch_size, 1])
            #print(y.shape)
            # Train Discriminator
            metrics = vaegan_disc.train_on_batch(real_images, y)
            loss = metrics[0]
            disc_loss = loss
            disc_acc = metrics[1]
            log = "%d-%d: [discriminator loss (real): %f, acc: %f]" % (i,j, loss, disc_acc)
            #print(log)

            # Generate fake images
            noise = np.random.uniform(size=(batch_size, z_dim), low=-1.0, high=1.0)
            #noise = np.random.normal(size=(batch_size, z_dim))
            fake_images = vaegan_decoder.predict(noise)
            y = np.zeros([batch_size, 1])

            metrics = vaegan_disc.train_on_batch(fake_images, y)
            loss = metrics[0]
            disc_loss = (loss + disc_loss)/2
            disc_acc = metrics[1]
            log = "%s [discriminator loss (fake): %f, acc: %f]" % (log, loss, disc_acc)
            #print(log)

            # Generate fake image
            noise = np.random.uniform(size=(batch_size, z_dim), low=-1.0, high=1.0)
            #noise = np.random.normal(size=(batch_size, z_dim))
            # Label fake images as real
            y = np.ones([batch_size, 1])
            # Train the Adversarial network
            metrics = gan.train_on_batch(noise, y)
            loss = metrics[0]
            acc = metrics[1]
            logg = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
            print(logg)
            '''
            if (loss - disc_loss > 2):
                # Generate fake image
                noise = np.random.uniform(size=(batch_size, z_dim), low=-1.0, high=1.0)
                #noise = np.random.normal(size=(batch_size, z_dim))
                # Label fake images as real
                y = np.ones([batch_size, 1])
                # Train the Adversarial network
                metrics = gan.train_on_batch(noise, y)
                loss = metrics[0]
                acc = metrics[1]
                logg = "[adversarial loss: %f, acc: %f]" % (loss, acc)
                print(logg)
            '''


            if j % 200 == 0:
                model_save_path = 'gan_checkpoints/gan-celebA-model-'+'{:05}'.format(i)+'-'+'{:05}'.format(j)+'.h5'
                print("Saving model to", model_save_path)
                gan.save_weights(model_save_path)

                #z_sample = np.random.uniform(size=(25,z_dim), low=-1.0, high=1.0)

                # Predict Sample
                out_random = vaegan_decoder.predict(z_sample)
                # Unnormalize samples
                out_random = (out_random + 1)*127.5
                out_random = out_random.astype(np.uint8)
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
                out_filename = 'gan_checkpoints/out' + '{:05}'.format(i)+'-'+'{:05}'.format(j)+'.jpg'
                cv2.imwrite(out_filename, fig)






    #output sampling
    '''
    num_outputs = 10
    z_sample = np.random.uniform(size=(num_outputs,z_dim), low=-3.0, high=3.0)
    out_random = vaegan_decoder.predict(z_sample)
    print("min", np.min(out_random))
    print("max", np.max(out_random))
    for i in range (out_random.shape[0]):
        cv2.imshow("image",out_random[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    some_gen = celeb_loader(batch_size=128)
    data, _ = next(some_gen)
    #print(vae)
    out_enc = vaegan_encoder.predict(data)
    #out = vaegan_decoder.predict(out_enc[2])

    out = vae.predict(data)
    print("data", data.shape)
    print("out", out.shape)

    for i in range (data.shape[0]):
        cv2.imshow("image",data[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        cv2.imshow("out image",out[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''


    #plot_results(models, data, batch_size=batch_size, model_name="vae_dcnn_celebA")
