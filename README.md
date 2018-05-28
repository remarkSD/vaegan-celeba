# vaegan-celeba


models_celebA.py contains models used for this project.

celebA_loader.py contains the data loader used to load the CelebA dataset.


Training:
 - train-celeba.py was used to train the VAE model.
 - gan-celeba.py was used to train the GAN model.
 - add "-w" to load weights (e.g. python train-celeba.py -w vae-model.h5)

Results:

VAE:

Images from noise:

![](img/out.jpg)

Autoencoder:

![](imgs/ae_out.jpg)

GAN

![](imgs/gan_out.jpg)
