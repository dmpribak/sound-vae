from ast import Break
import glob, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile # get the api
from scipy.fft import fft, rfft, rfftfreq
from scipy.signal import stft, istft
from pylab import savefig
import pandas as pd
import torch
import math
from PIL import Image

FFTBINS = 128 #number of bins between 0 and MAXFREQ hz
MAXFREQ = 22050#20480
MAXFRAMES = 1024
DEBUG = False

def norm_real_fft(filename):
    SAMPLE_RATE, data = wavfile.read(filename)
    if DEBUG:
        print(f"sample rate = {SAMPLE_RATE}")
        print(f"number of channels = {len(data.shape)}")
    if len(data.shape) == 2:
        mono = data.T[0] #just grab first channel for now; later should take average of channels or something
    else:
        mono = data
    
    N = len(mono)
    freq, times, Zxx = stft(mono, SAMPLE_RATE, nperseg=256) #consider abs more closely? could also consider using normal fft for phase data
    

    if DEBUG:
        plt.pcolormesh(times, freq[:FFTBINS], np.abs(Zxx), vmin=0, shading='gouraud')
        plt.ylim([0, MAXFREQ])
        savefig(filename+'_raw_fft.png')
    
    Zxx = Zxx[:FFTBINS]
    Zxx = np.abs(Zxx.T)
    

    # take average of frequencies in bins of size MAXFREQ/FFTBINS, then normalize
    #bins = np.zeros((MAXFRAMES, FFTBINS))
    """
    for frame in range(len(times)):
        if frame >= MAXFRAMES:
            break
        rfftOut = Zxx[frame]
        
        currBin = 0
        numFreqsInBin = 0
        maxBinValue = -1.0
        for i in range(0, len(freq)):
            if(freq[i] > MAXFREQ):
                break
            
            while(True):
                binMax = (MAXFREQ/FFTBINS)*(currBin+1)
                if freq[i] <= binMax:
                    bins[frame][currBin] += rfftOut[i]
                    numFreqsInBin += 1
                    break
                    
                else:
                    if numFreqsInBin != 0:
                        bins[frame][currBin] = bins[frame][currBin]/numFreqsInBin #turn sum into average
                    if bins[frame][currBin] > maxBinValue:
                        maxBinValue = bins[frame][currBin]
                    currBin += 1
                    numFreqsInBin = 0

        if maxBinValue != -1:
            bins /= maxBinValue #normalize
    """
    bins = Zxx[:MAXFRAMES]
    if bins.shape[0] != MAXFRAMES:
        bins = np.concatenate((bins, np.zeros((MAXFRAMES-bins.shape[0], FFTBINS))), axis=0)
    
    bins /= np.max(bins)

    if DEBUG:
        plt.clf()
        plt.pcolormesh(times[:MAXFRAMES], freq, bins.T, vmin=0, vmax=1, shading='gouraud')
        plt.ylim([0, MAXFREQ])
        savefig(filename+'_binned_fft.png')
        
    return bins

def load_data(directory, randomize=False, refreshCsv=False):
    #norm_real_fft('test.wav')
    #wavFiles = glob.glob("X:/Datasets/sounds/sounds/*/*.wav")
    if refreshCsv:
        wavFiles = glob.glob(directory+"/*.wav")
        data = np.zeros(shape=(len(wavFiles), MAXFRAMES, FFTBINS))
        for i, file in enumerate(wavFiles):
            data[i] = norm_real_fft(file)

        np.save("./data/stfts.npy", data)
        wavFiles = np.array(wavFiles)
        np.save("./data/names.npy", wavFiles)

    data = np.load("./data/stfts.npy")
    wavFiles = np.load('./data/names.npy')
    
    if randomize:
        np.random.shuffle(data)

    data = torch.tensor(data, dtype=torch.float32).cuda()

    return data, wavFiles

