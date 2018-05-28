# vaegan-celeba
This is a keras implementation of the paper "Autoencoding beyond pixels using a learned similarity metric" by Larsen, A., et al.

VAEGAN Model

![](imgs/vaegan_model.png)

# Dependencies
 - keras
 - tensorflow
 - OpenCV
 - numpy

Training:
 - train-celeba.py was used to train the VAE model.
 - gan-celeba.py was used to train the GAN model.
 - add "-w" to load weights (e.g. python train-celeba.py -w vae-model.h5)
 - models_celebA.py contains models used for this project.
 - celebA_loader.py contains the data loader used to load the CelebA dataset.
 - CelebA dataset was used for training the model


# Results

VAE (Images from noise input):

![](imgs/out.jpg)

VAE Autoencoder (output-left; input-right):

![](imgs/ae_out.jpg)

GAN (Images from noise input):

![](imgs/gan_out.jpg)

VAEGAN (Images from noise input):

![](imgs/vaegan-gan_out.jpg)

VAEGAN Autoencoder (output-left; input-right):

![](imgs/vaegan-ae_out.jpg)

# References
 - "Autoencoding beyond pixels using a learned similarity metric" by Larsen, A., et al. (https://arxiv.org/abs/1512.09300)
 - CelebA Dataset (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
