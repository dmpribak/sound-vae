import glob, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile # get the api
from scipy.fft import rfft, rfftfreq
from pylab import savefig
import pandas as pd
import torch
import math

FFTBINS = 2048 #number of bins between 0 and MAXFREQ hz
MAXFREQ = 20480
DEBUG = False

def norm_real_fft(filename):
    SAMPLE_RATE, data = wavfile.read(filename)
    if DEBUG:
        print(f"sample rate = {SAMPLE_RATE}")
        print(f"number of channels = {data.shape[1]}")
    if len(data.shape) == 2:
        mono = data.T[0] #just grab first channel for now; later should take average of channels or something
    else:
        mono = data
    
    N = len(mono)
    rfftOut = abs(rfft(x = mono)) #consider abs more closely? could also consider using normal fft for phase data
    freq = rfftfreq(N, 1 / SAMPLE_RATE)

    if DEBUG:
        plt.plot(freq, rfftOut, 'r')
        plt.xlim([0, MAXFREQ])
        savefig(filename+'_raw_fft.png')

    # take average of frequencies in bins of size MAXFREQ/FFTBINS, then normalize
    bins = np.zeros(FFTBINS)
    currBin = 0
    numFreqsInBin = 0
    maxBinValue = -1.0
    for i in range(0, len(freq)):
        if(freq[i] > MAXFREQ):
            break
        binMax = (MAXFREQ/FFTBINS)*(currBin+1)
        if freq[i] < binMax:
            bins[currBin] += rfftOut[i]
            numFreqsInBin += 1
        else:
            if numFreqsInBin != 0:
                bins[currBin] = bins[currBin]/numFreqsInBin #turn sum into average
            if bins[currBin] > maxBinValue:
                maxBinValue = bins[currBin]
            currBin += 1
            numFreqsInBin = 0
    if maxBinValue != -1:
        bins /= maxBinValue #normalize

    if DEBUG:
        plt.clf()
        plt.plot(bins)
        plt.xlim([0, FFTBINS])
        savefig(filename+'_binned_fft.png')

    return bins

def load_data(directory, randomize=False, refreshCsv=False):
    #norm_real_fft('test.wav')
    #wavFiles = glob.glob("X:/Datasets/sounds/sounds/*/*.wav")
    df = pd.DataFrame()
    if refreshCsv:
        wavFiles = glob.glob(directory+"/*.wav")
        data = np.zeros(shape=(len(wavFiles), FFTBINS))
        for i, file in enumerate(wavFiles):
            data[i] = norm_real_fft(file)

        df = pd.DataFrame(data)
        df.insert(0, "path", wavFiles)
        df.to_csv("data.csv")
    else:
        df = pd.read_csv("data.csv")
        wavFiles = df.iloc[:, 1].values
        data = df.iloc[:, 2:].values
    
    if randomize:
        np.random.shuffle(data)

    data = torch.tensor(data, dtype=torch.float32).cuda()
    return data, wavFiles