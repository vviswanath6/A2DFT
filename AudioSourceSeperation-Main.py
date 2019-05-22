import numpy as np
import imageio
import audio_utilities
from scipy.io import wavfile
import scipy
import scipy.signal
from pylab import *
import matplotlib.pyplot as plot
from imageio import imwrite as saveimage
from imageio import imread as openimage
import cv2
from skimage import color
from skimage import io
import math
import sys
import time
import wave
import array
import os
from os.path import expanduser
import pyprind
import sys
import time
import scipy.ndimage.filters as filters

def GetAudio(InputFile, expectedFs=44100):
    print("Getting Audio...")
    fs, y = scipy.io.wavfile.read(InputFile)
    num_type = y[0].dtype
    if num_type == 'int16':
        y = y*(1.0/32768)
    elif num_type == 'int32':
        y = y*(1.0/2147483648)
    elif num_type == 'float32':
        # do nothing
        pass
    elif num_type == 'uint8':
        raise Exception('8-bit PCM is not supported.')
    else:
        raise Exception('Unknown format.')
    if fs != expectedFs:
        raise Exception('Invalid sample rate.')
    if y.ndim == 1:
        return y
    else:
        return y.mean(axis=1)

def SpectrogramByStft(x, FftSize, hopsamp):
    print("Generating Spectrogram...")
    window = np.hanning(FftSize)
    FftSize = int(FftSize)
    hopsamp = int(hopsamp)
    return np.array([np.fft.rfft(window*x[i:i+FftSize])
                     for i in range(0, len(x)-FftSize, hopsamp)])

def IstftForReconstruction(X, fft_size, hopsamp):
    print("Calculating Inverse STFT...")
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    window = np.hanning(fft_size)
    time_slices = X.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    x = np.zeros(len_samples)
    for n,i in enumerate(range(0, len(x)-fft_size, hopsamp)):
        x[i:i+fft_size] += window*np.real(np.fft.irfft(X[n]))
    return x

def SaveAudiotoFile(x, sample_rate, outfile):
    print('Saving Audio as {}'.format(outfile))
    x_max = np.max(abs(x))
    assert x_max <= 1.0, 'Input audio value is out of range. Should be in the range [-1.0, 1.0].'
    x = x*32767.0
    data = array.array('h')
    for i in range(len(x)):
        cur_samp = int(round(x[i]))
        data.append(cur_samp)
    f = wave.open(outfile, 'w')
    f.setparams((1, 2, sample_rate, 0, "NONE", "Uncompressed"))
    f.writeframes(data.tobytes())
    f.close()

def FourierTransform(input):
    print("Finding Fourier Transform...")
    FT = np.fft.fft2(input)
    return FT

def InvFourierTransform(input):
    print("Finding Inverse Fourier Transform...")
    IFT = np.fft.ifft2(input)
    return IFT

def MakeCrudeMask(input):
    print("Making crude mask(Global maxima)...")
    max1 = np.max(input)
    min1 = np.min(input)
    print('{},{}'.format(max1, min1))
    CrudeMask = np.where(np.logical_and(-1000 <= input, input <= max1), 1, 0)
    return CrudeMask

def MakeMask(input):
    print("Making mask(Local maxima)...")
    data = 20*np.log(np.abs(input))
    dataarray = np.array(data, dtype=np.float64)
    StdDeviation = np.std(data)
    print(StdDeviation)
    threshold = StdDeviation   #10
    neighborhood_size = (1, WindowLengthAlongRateAxis)
    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    mask = np.where((diff == 1), 1, 0)
    return mask

FftSize = 1024           #1024, 2048, 4096, 8192, 16384, 32768
#iterations = 10
SampleRateinHz = 44100
input1 = GetAudio("mixture1.wav", expectedFs=44100)
hopsamp = FftSize // 8
WindowLengthAlongRateAxis = 70

StftFull = SpectrogramByStft(input1,FftSize, hopsamp)
StftMagnitude = 20*np.log(np.abs(StftFull))
StftTranspose = StftMagnitude.T**0.125

figure(1)
plot.axis("off")
plot.title('StftMagnitude')
plot.imshow(StftMagnitude.T) #**0.125
fig = plot.gcf()
ax = plot.gca()
extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plot.savefig('original_spectrogram.png',
             bbox_inches=extent,
             pad_inches=0,
             transparent=True)

TwoDFourierT = FourierTransform(StftFull)
saveimage('2dfft.png', 20*np.log(np.abs(TwoDFourierT.T)))
SCBackgroundMask = MakeMask(TwoDFourierT)
SCBackgroundMask1 = MakeCrudeMask(TwoDFourierT)
saveimage('SCBackgroundMask.png',abs(SCBackgroundMask.T))
SCForegroundMask = 1 - SCBackgroundMask
saveimage('SCForegroundMask.png', abs(SCForegroundMask.T))
input6 = TwoDFourierT.copy()
input7 = TwoDFourierT.copy()
input6 = np.multiply(SCBackgroundMask, input6)
input7 = np.multiply(SCForegroundMask, input7)
TwoDFourierTMag = 20*np.log(np.abs(input6))

figure(2)
plot.axis("off")
plot.title('TwoDFourierTMag')
plt.imshow(TwoDFourierTMag.T)

BackgroundSpectrogram = abs(InvFourierTransform(input6))
ForegroundSpectrogram = abs(InvFourierTransform(input7))
BackgroundSpectrogramMag = 20*np.log(np.abs(BackgroundSpectrogram))
ForegroundSpectrogramMag = 20*np.log(np.abs(ForegroundSpectrogram))
BackgroundSpectrogramMagT = BackgroundSpectrogramMag.T
ForegroundSpectrogramMagT = ForegroundSpectrogramMag.T

figure(3)
plot.axis("off")
plot.title('BackgroundSpectrogramMag')
plt.imshow(BackgroundSpectrogramMagT)
fig = plt.gcf()
ax = plt.gca()
extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plot.savefig("BackgroundSpectrogramMag.png", bbox_inches=extent, pad_inches=0, transparent=True)

figure(4)
plot.axis("off")
plot.title('ForegroundSpectrogramMag')
plt.imshow(ForegroundSpectrogramMagT)
fig = plt.gcf()
ax = plt.gca()
extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plot.savefig("ForegroundSpectrogramMag.png", bbox_inches=extent, pad_inches=0, transparent=True)

TFBackgroundMask = np.where((BackgroundSpectrogram > ForegroundSpectrogram), 1, 0)
TFForegroundMask = 1 - TFBackgroundMask
saveimage('TFBackgroundMask.png', TFBackgroundMask.T)
saveimage('TFForegroundMask.png', TFForegroundMask.T)
BGSpectrogramToBeRecon = np.multiply(TFBackgroundMask, StftFull)
FGSpectrogramToBeRecon = np.multiply(TFForegroundMask, StftFull)
saveimage('StftFull.png', abs(StftFull))
saveimage('BGSpectrogramToBeRecon.png', abs(BGSpectrogramToBeRecon.T))
saveimage('FGSpectrogramToBeRecon.png', abs(FGSpectrogramToBeRecon.T))
BGAudioDataRecovered = IstftForReconstruction(BGSpectrogramToBeRecon, FftSize, hopsamp)
FGAudioDataRecovered = IstftForReconstruction(FGSpectrogramToBeRecon, FftSize, hopsamp)

figure(5)
plot.axis("off")
plot.title('TFBackgroundMask')
plot.imshow(20*np.log(np.abs(TFBackgroundMask.T)))

figure(6)
plot.axis("off")
plot.title('TFForegroundMask')
plot.imshow(20*np.log(np.abs(TFForegroundMask.T)))

figure(7)
plot.axis("off")
plot.title('BGSpectrogramToBeRecon')
plot.imshow(20*np.log(np.abs(BGSpectrogramToBeRecon.T)))

figure(8)
plot.axis("off")
plot.title('FGSpectrogramToBeRecon')
plot.imshow(20*np.log(np.abs(FGSpectrogramToBeRecon.T)))

max_sample = np.max(abs(BGAudioDataRecovered))
if max_sample > 1.0:
    BGAudioDataRecovered = BGAudioDataRecovered / max_sample

max_sample = np.max(abs(FGAudioDataRecovered))
if max_sample > 1.0:
    FGAudioDataRecovered = FGAudioDataRecovered / max_sample

SaveAudiotoFile(BGAudioDataRecovered, SampleRateinHz, 'BG.wav')
SaveAudiotoFile(FGAudioDataRecovered, SampleRateinHz, 'FG.wav')
plot.show()
