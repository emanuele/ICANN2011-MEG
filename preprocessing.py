"""Preprocessing steps.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.mlab as mlab


def halve_channels(data):
    """for each even channel i and odd channel i+1 derive a new
    channel=sqrt(ch_i^2 + ch_{i+1}^2 so that the number of channels
    goes from 204 to 102.
    """
    return np.sqrt(data[:,::2,:]*data[:,::2,:] + data[:,1::2,:]*data[:,1::2,:])
    

def compute_psd(x, t_start=0, t_end=-1, NFFT=128, Fs=200, noverlap=None, f_min=0, f_max=-1):
    """Compute power spectral densities of a dataset 'x' which is
    organized as: trials X sensors X time.

    t_start, t_end = define a time window. Default: all the trial.
    NFFT = size of the FFT window.
    noverlap = amount of overlapping between windows.
    f_min, f_max = return a specific frequency window of PSD. Default: full range.

    Returns:

    x_psd = dataset organized as: trials X channels X PSD.
    freq = the actual frequencies of the PSD.
    """
    print "Computing PSD of each trial (%s) and for each channel (%s):" % (x.shape[0], x.shape[1])
    if noverlap is None:
        noverlap = NFFT - Fs * 0.100 # See van gerven and jensen 2009
    size = NFFT / 2 + 1
    f_idx = range(size)
    if f_min!=0 and f_max!=-1: # compute actual frequency interval size
        tmp, freq = mlab.psd(x[0, 0, t_start:t_end], NFFT=NFFT, Fs=Fs, noverlap=noverlap)
        f_idx = np.where(((freq>=f_min) & (freq<f_max)))[0]
        size = f_idx.size
    shape = (x.shape[0], x.shape[1], size)
    x_psd = np.zeros(shape)
    for trial in range(x.shape[0]):
        print "T",trial,
        sys.stdout.flush()
        for sensor in range(x.shape[1]):
            tmp, freq = mlab.psd(x[trial, sensor, t_start:t_end], NFFT=NFFT, Fs=Fs, noverlap=noverlap)
            x_psd[trial, sensor, :] = tmp.squeeze()[f_idx]
    print
    return x_psd, freq


if __name__=='__main__':

    day = 1
    filter_bank = None

    x_filename = 'megicann_train_v2_day'+str(day)+'_'+str(filter_bank)+'.npy'
    x = np.load(x_filename) # trials X channels X time ; 200Hz.
    y_filename = 'megicann_train_v2_class_day'+str(day)+'.npy'
    y = np.load(y_filename)

    xh = halve_channels(x)
    xh_psd, freq = compute_psd(xh)

    channel = 50
    plt.figure()
    for c in [1,2,3,4,5]:
        plt.semilogy(freq, xh_psd[y==c,channel,:].mean(0), label='class '+str(c))
    plt.legend()
    plt.title('PSD of channel %s during day %s' % (channel, day))
    plt.xlabel('freq (Hz)')
    plt.ylabel('PSD')
