# sound-vae

This is a variational autoencoder which has been trained on the short-term fourier transform of an assortment of sounds used in music production (drum samples, synths, vocals, etc.). The STFT is treated like an image but with translational invariance only in the time axis. Thus, a convolutional neural network is used for the encoder where the first CNN layer has a kernel where the size in the frequency dimension is the number of frequency bins.
