
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import cv2
import tensorflow as tf
#from utils import load, save
#from layers import Deconv2D
from keras import backend as K
from keras.layers import Input, Dense, Reshape, Lambda, Activation, Conv2D, Deconv2D,Conv2DTranspose, LeakyReLU, Flatten, BatchNormalization as BN
from keras.models import Sequential, Model
#from keras import initializations

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def encoder(num_filters, ch, rows, cols,z_dim=2048, kernel_size=(5,5), strides=(2,2)):

    model = Sequential()
    X = Input(shape=(rows[-1], cols[-1], ch))

    model = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', name='enc_conv2D_01', input_shape=(rows[-1], cols[-1], ch))(X)
    model = BN(name="enc_bn_01")(model)
    model = LeakyReLU(0.2)(model)

    model = Conv2D(num_filters*2,kernel_size=kernel_size, strides=strides, padding='same', name='enc_conv2D_02')(model)
    model = BN(name="enc_bn_02")(model)
    model = LeakyReLU(0.2)(model)

    model = Conv2D(num_filters*4,kernel_size=kernel_size, strides=strides, padding='same', name='enc_conv2D_03')(model)
    model = BN( name="enc_bn_03")(model)
    model = LeakyReLU(0.2)(model)

    #model = Reshape((8,8,256))(model)
    model = Flatten()(model)
    #model = Dense(2048, name="enc_dense_01")(model)
    #model = BN(name="enc_bn_04",  epsilon=1e-5)(model)
    #encoded_model = LeakyReLU(.2)(model)

    mean = Dense(z_dim, name="e_mean_dense")(model)
    logsigma = Dense(z_dim, name="e_sigma_dense")(model)

    mean = BN(name="enc_mean_bn")(mean)
    logsigma = BN(name="enc_sigma_bn")(logsigma)

    mean = Activation('relu', name="enc_mean_relu")(mean)
    logsigma = Activation('relu', name="enc_sigma_relu")(logsigma)
    z = Lambda(sampling, output_shape=(z_dim,), name='z')([mean, logsigma])
    meansigma = Model([X], [mean, logsigma, z], name="encoder")


    #X_decode = Input(shape=(8,8,256))
    #model = Dense(256, name="dec_dense_01")(encoded_model)

#    enc_model = Model(X, encoded_model)
#    dec_model = Model(X, model)
    return meansigma

def generator(num_filters,z_dim=2048, ch=3, kernel_size=(5,5), strides=(2,2)):
    model = Sequential()
    X = Input(shape=(z_dim,))
    model = Dense(8*8*256, input_shape=(z_dim,), name="dec_dense_01")(X)
    #model = Dense(7*7*32, input_shape=(z_dim,), name="dec_dense_01")(X)
    #model = Reshape((7,7,32))(model)
    model = BN(name="dec_bn_01")(model)
    model = LeakyReLU(0.2)(model)
    #model = Activation('relu')(model)
    model = Reshape((8,8,256))(model)

    model = Conv2DTranspose(num_filters*8,kernel_size=kernel_size, strides=strides, padding='same', name='dec_deconv2D_01')(model)
    model = BN(name="dec_bn_02")(model)
    model = LeakyReLU(0.2)(model)
    #model = Activation('relu')(model)

    model = Conv2DTranspose(num_filters*4,kernel_size=kernel_size, strides=strides, padding='same', name='dec_deconv2D_02')(model)
    model = BN(name="dec_bn_03")(model)
    model = LeakyReLU(0.2)(model)
    #model = Activation('relu')(model)

    model = Conv2DTranspose(num_filters,kernel_size=kernel_size, strides=strides, padding='same', name='dec_deconv2D_03')(model)
    model = BN(name="dec_bn_04")(model)
    model = LeakyReLU(0.2)(model)
    #model = Activation('relu')(model)

    model = Conv2DTranspose(ch, kernel_size=kernel_size, strides=1, padding='same', name='dec_deconv2D_04')(model)
    model = Activation('tanh')(model)

    dec_model = Model(X, model, name="decoder")
    return dec_model

def discriminator(num_filters, ch, rows, cols, z_dim,kernel_size=(5,5), strides=(2,2)):
    model = Sequential()
    X = Input(shape=(rows[-1],cols[-1],ch))
    model = Conv2D(num_filters, kernel_size=kernel_size, strides=2, padding='same', name='disc_conv2D_01')(X)
    model = BN(name="disc_bn_01")(model)
    model = LeakyReLU(0.2)(model)
    #model = Activation('relu')(model)

    model = Conv2D(num_filters*4,kernel_size=kernel_size, strides=strides, padding='same', name='disc_conv2D_02')(model)
    model = BN( name="disc_bn_02")(model)
    model = LeakyReLU(0.2)(model)
    #model = Activation('relu')(model)

    model = Conv2D(num_filters*8,kernel_size=kernel_size, strides=strides, padding='same', name='disc_conv2D_03')(model)
    model = BN(name="disc_bn_03")(model)
    model = LeakyReLU(0.2)(model)
    #model = Activation('relu')(model)

    model = Conv2D(num_filters*8,kernel_size=kernel_size, strides=strides, padding='same', name='disc_conv2D_04')(model)
    model = BN(name="disc_bn_04")(model)
    model = LeakyReLU(0.2)(model)
    #model = Activation('relu')(model)

    #model = Reshape((8,8,256))(model)
    model = Flatten()(model)
    model = Dense(512, name="disc_dense_01")(model)
    model = BN(name="disc_bn_05")(model)
    model = LeakyReLU(0.2)(model)
    #model = Activation('relu')(model)

    model = Dense(1, name="disc_dense_02")(model)
    model = Activation('sigmoid', name='disc_sigmoid')(model)

    disc_model = Model(X, model,name="discriminator")

    return disc_model


def discriminator_l(num_filters, ch, rows, cols, z_dim,kernel_size=(5,5), strides=(2,2)):
    model = Sequential()
    X = Input(shape=(rows[-1],cols[-1],ch))
    model = Conv2D(num_filters, kernel_size=kernel_size, strides=1, padding='same', name='disc_conv2D_01')(X)
    model = BN(name="disc_bn_01")(model)
    model = LeakyReLU(0.2)(model)
    #model = Activation('relu')(model)

    model = Conv2D(num_filters*4,kernel_size=kernel_size, strides=strides, padding='same', name='disc_conv2D_02')(model)
    model = BN( name="disc_bn_02")(model)
    model = LeakyReLU(0.2)(model)
    #model = Activation('relu')(model)

    model = Conv2D(num_filters*8,kernel_size=kernel_size, strides=strides, padding='same', name='disc_conv2D_03')(model)
    model = BN(name="disc_bn_03")(model)
    model = LeakyReLU(0.2)(model)
    #model = Activation('relu')(model)

    model_l = Conv2D(num_filters*8,kernel_size=kernel_size, strides=strides, padding='same', name='disc_conv2D_04')(model)
    model = BN(name="disc_bn_04")(model_l)
    model = LeakyReLU(0.2)(model)
    #model = Activation('relu')(model)

    #model = Reshape((8,8,256))(model)
    model = Flatten()(model)
    model = Dense(512, name="disc_dense_01")(model)
    model = BN(name="disc_bn_05")(model)
    model = LeakyReLU(0.2)(model)
    #model = Activation('relu')(model)

    model = Dense(1, name="disc_dense_02")(model)
    model = Activation('sigmoid', name='disc_sigmoid')(model)

    disc_model = Model(X, [model, model_l],name="discriminator")

    return disc_model

def discriminator_l2(num_filters, ch, rows, cols, z_dim,kernel_size=(5,5), strides=(2,2)):
    model = Sequential()
    X = Input(shape=(rows[-1],cols[-1],ch))
    model = Conv2D(num_filters, kernel_size=kernel_size, strides=1, padding='same', name='disc_conv2D_01')(X)
    model = BN(name="disc_bn_01")(model)
    model = LeakyReLU(0.2)(model)
    #model = Activation('relu')(model)

    model = Conv2D(num_filters*4,kernel_size=kernel_size, strides=strides, padding='same', name='disc_conv2D_02')(model)
    model = BN( name="disc_bn_02")(model)
    model = LeakyReLU(0.2)(model)
    #model = Activation('relu')(model)

    model = Conv2D(num_filters*8,kernel_size=kernel_size, strides=strides, padding='same', name='disc_conv2D_03')(model)
    model = BN(name="disc_bn_03")(model)
    model = LeakyReLU(0.2)(model)
    #model = Activation('relu')(model)

    model_l = Conv2D(num_filters*8,kernel_size=kernel_size, strides=strides, padding='same', name='disc_conv2D_04')(model)
    d_model = BN(name="disc_bn_04")(model_l)
    d_model = LeakyReLU(0.2)(d_model)
    #model = Activation('relu')(model)

    #model = Reshape((8,8,256))(model)
    d_model = Flatten()(d_model)
    d_model = Dense(512, name="disc_dense_01")(d_model)
    d_model = BN(name="disc_bn_05")(d_model)
    d_model = LeakyReLU(0.2)(d_model)
    #model = Activation('relu')(model)

    d_model = Dense(1, name="disc_dense_02")(d_model)
    d_model = Activation('sigmoid', name='disc_sigmoid')(d_model)

    disc_model = Model(X, d_model,name="discriminator")
    disc_model_l2 = Model(X, model_l,name="discriminator-l2")
    return disc_model, disc_model_l2

if __name__ == '__main__':
    df_dim = 64
    batch_size = 64
    channels = 3
    height = np.array([64])
    width = np.array([64])

    vaegan_encoder = encoder(num_filters=df_dim, ch=channels, rows=height, cols=width)
    vaegan_decoder = generator(num_filters=32, z_dim=256)
    vaegan_disc = discriminator(num_filters=32, z_dim=256, rows=height, cols=width)
    vaegan_encoder.compile(optimizer='RMSProp', loss='binary_crossentropy')
    vaegan_decoder.compile(optimizer='RMSProp', loss='binary_crossentropy')
    vaegan_disc.compile(optimizer='RMSProp', loss='binary_crossentropy')
    vaegan_decoder.summary()
    vaegan_encoder.summary()
    vaegan_disc.summary()
    '''
    Z2 = Input(batch_shape=(batch_size, 256), name='more_noise')
    Z = vaegan_decoder.input
    Img = vaegan_disc.input
    G_train = vaegan_decoder(Z)
    E_mean, E_logsigma = vaegan_encoder(Img)
    G_dec = vaegan_decoder(E_mean + Z2 * E_logsigma)
    D_fake, F_fake = vaegan_disc(G_train)
    D_dec_fake, F_dec_fake = vaegan_disc(G_dec)
    D_legit, F_legit = vaegan_disc(Img)
    '''
