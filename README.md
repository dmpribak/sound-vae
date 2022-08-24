# sound-vae

This is a variational autoencoder which has been trained on the short-time fourier transform of an assortment of sounds used in music production (drum samples, synths, vocals, etc.). The STFT is treated like an image but with translational invariance only in the time axis. Thus, a convolutional neural network is used for the encoder where the first CNN layer has a kernel where the size in the frequency dimension is the number of frequency bins.

We can use the model to compare two sounds using the cosine similarity between the embedding vectors of the two sounds. This method does not work as a sound generation model because the STFT does not carry phase information. Other techniques would have to be incorporated to "guess" the phase so this could be used as a generative model.

An example of the VAE reconstructing the STFT of a 440 hz sine wave from an embedding vector:

![image](https://github.com/dmpribak/sound-vae/blob/f5ce186fca37a6eb485937cac9b127cc3ebf3b32/chart.png)
