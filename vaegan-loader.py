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
import math

from models_celebA import *
from celebA_loader import *

image_size = 64
channels = 3
# network parameters
input_shape = (image_size, image_size, channels)
batch_size = 64
kernel_size = 3
filters = np.array([64,32])
z_dim = 128
epochs = 100
lr = 0.0003
decay = 0
dir='/home/raimarc/Documents/img_align_celeba/'
def nll_loss(mean, x, ln_var=0):
    x_prec = 1
    x_diff = x - mean
    x_power = (x_diff * x_diff) * x_prec * -0.5
    loss = (0. + math.log(2 * math.pi)) / 2 - x_power

    return K.sum(loss)


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

    # Instantiate encoder, decoder/generator, discriminator models
    inputs = Input(shape=input_shape)

    vaegan_encoder = encoder(num_filters=filters[0], ch=channels, rows=height, cols=width, z_dim=z_dim)
    vaegan_decoder = generator(num_filters=filters[1], z_dim=z_dim, ch=channels)
    vaegan_disc, vaegan_disc_l = discriminator_l2(num_filters=32, z_dim=z_dim, ch=3, rows=height, cols=width)
    #vaegan_decoder = build_generator(Input(shape=(z_dim,)), image_size)
    #vaegan_disc = build_discriminator(inputs)
    vaegan_encoder.summary()
    vaegan_decoder.summary()
    vaegan_disc.summary()
    vaegan_disc_l.summary()
    plot_model(vaegan_encoder, to_file='enc_skel.png', show_shapes=True)
    plot_model(vaegan_decoder, to_file='dec_skel.png', show_shapes=True)
    plot_model(vaegan_disc, to_file='disc_skel.png', show_shapes=True)
    plot_model(vaegan_disc_l, to_file='disc_l_skel.png', show_shapes=True)


    # Create Encoder Model
    #outputs1 = vaegan_encoder(inputs)[2]
    #print("outputshape", outputs.shape)
    vaegan_encoder.trainable=True
    vaegan_decoder.trainable=False
    vaegan_disc.trainable=False
    vaegan_disc_l.trainable=False
    z_mean, z_logvar, z = vaegan_encoder(inputs)
    #outputs = vaegan_disc(vaegan_decoder(vaegan_encoder(inputs)[2]))
    outputs = vaegan_disc_l(vaegan_decoder(z))
    enc_optimizer = RMSprop(lr=lr*0.4)
    encoder_model = Model(inputs, outputs, name='encoder')
    #kl_loss = 1 + vaegan_encoder(inputs)[1] - K.square(vaegan_encoder(inputs)[0]) - K.exp(vaegan_encoder(inputs)[1])
    kl_loss = 1 + z_logvar - K.square(z_mean) - K.exp(z_logvar)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    llike_loss = nll_loss(mean=vaegan_disc_l(inputs), x=outputs)
    enc_loss = K.mean(kl_loss + llike_loss)

    encoder_model.add_loss(enc_loss)
    encoder_model.compile(optimizer=enc_optimizer,
                            metrics=['accuracy'])
    print("Encoder")
    encoder_model.summary()
    plot_model(encoder_model, to_file='encoder.png', show_shapes=True)


    # Create Discriminator Model
    vaegan_encoder.trainable=False
    vaegan_decoder.trainable=True
    vaegan_disc.trainable=True
    vaegan_disc_l.trainable=True
    disc_optimizer = RMSprop(lr=lr)
    disc_model = vaegan_disc
    disc_model.compile(loss='binary_crossentropy',
                        optimizer=disc_optimizer,
                        metrics=['accuracy'])
    print("Discriminator")
    disc_model.summary()
    plot_model(disc_model, to_file='discriminator.png', show_shapes=True)


    # Create Decoder Model

    outputs1 = vaegan_disc(vaegan_decoder(vaegan_encoder(inputs)[2]))
    z_in = Input(shape=(z_dim,))
    outputs2 = vaegan_disc(vaegan_decoder(z_in))
    outputs3 = vaegan_disc(inputs)
    #z_input = vaegan_decoder(vaegan_encoder(z_in)[2])
    #print("outputshape", outputs.shape)

    vaegan_encoder.trainable=False
    vaegan_decoder.trainable=True
    vaegan_disc.trainable=False
    vaegan_disc_l.trainable=False
    optimizer = RMSprop(lr=lr)
    decoder_model = Model([inputs, z_in], [outputs3, outputs1, outputs2], name='decoder')
    gamma = 1e-6
    #gan_loss_1 = binary_crossentropy(K.ones_like(outputs1[0]), outputs1[0])
    #gan_loss_2 = binary_crossentropy(K.ones_like(outputs2[0]), outputs2[0])
    gan_loss_1 = binary_crossentropy(K.zeros_like(outputs1), outputs1)
    gan_loss_2 = binary_crossentropy(K.zeros_like(outputs2), outputs2)
    gan_loss = K.mean(gan_loss_1 + gan_loss_2)
    #llike_loss = nll_loss(mean=outputs3[1], x=outputs1[1])
    #dec_loss = -gan_loss + gamma*llike_loss
    #decoder_model.add_loss(dec_loss)
    decoder_model.add_loss(-1.0*gan_loss)
    #decoder_model.add_loss(gan_loss)
    decoder_model.add_loss(gamma*llike_loss)
    decoder_model.compile(optimizer=optimizer)
    print("Decoder")
    decoder_model.summary()
    plot_model(decoder_model, to_file='decoder.png', show_shapes=True)


    #decoder_model.load_weights(args.weights)
    i = 5
    j = 1800
    folder = 'vaegan_out'
    dec_model_save_path = folder + '/dec_vaegan-model-'+'{:05}'.format(i)+'-'+'{:05}'.format(j)+'.h5'
    enc_model_save_path = folder + '/enc_vaegan-model-'+'{:05}'.format(i)+'-'+'{:05}'.format(j)+'.h5'
    disc_model_save_path = folder + '/disc_vaegan-model-'+'{:05}'.format(i)+'-'+'{:05}'.format(j)+'.h5'
    print("loading weights", dec_model_save_path)
    encoder_model.load_weights(enc_model_save_path)
    disc_model.load_weights(disc_model_save_path)
    decoder_model.load_weights(dec_model_save_path)


    while True:
        # Predict Sample
        #z_sample =  np.random.normal(size=(batch_size, z_dim))
        z_sample =  np.random.uniform(size=(batch_size, z_dim), low=-3.0, high=3.0)
        out_random = vaegan_decoder.predict(z_sample)
        # Unnormalize samples
        out_random = (out_random + 1)*127.5
        out_random = out_random.astype(np.uint8)

        # Put samples in grid
        fig = np.zeros((64*5,64*5,3), dtype=np.uint8)
        for k1 in range (5):
            for k2 in range (5):
                fig[64*k2:64*(k2+1),64*k1:64*(k1+1),:] = out_random[k1*5+k2]
        #cv2.imshow("image",out_random[0])
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Write samples
        out_filename = 'vaegan_cps/gan_out' + '{:05}'.format(i)+'-'+'{:05}'.format(j)+'.jpg'
        cv2.imwrite(out_filename, fig)
        cv2.imshow('samples from noise', fig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        # Predict VAE Sample
        out_random = vaegan_decoder.predict(vaegan_encoder.predict(real_images)[2])
        # Unnormalize samples
        out_random = (out_random + 1)*127.5
        out_random = out_random.astype(np.uint8)
        # Put samples in grid
        fig = np.zeros((64*5,64*5*2,3))
        for k1 in range (5):
            for k2 in range (5):
                fig[64*k2:64*(k2+1),64*k1:64*(k1+1),:] = out_random[k1*5+k2]
                fig[64*k2:64*(k2+1),64*(k1+5):64*(k1+1+5),:] = ((real_images[k1*5+k2] + 1)*127.5).astype(np.uint8)
        #cv2.imshow("image",out_random[0])
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Write samples
        out_filename = 'vaegan_cps/vae_out' + '{:05}'.format(i)+'-'+'{:05}'.format(j)+'.jpg'
        cv2.imwrite(out_filename, fig)

        '''





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
