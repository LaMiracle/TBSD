# -*- coding: utf-8 -*-

import numpy as np
from math import inf
import matplotlib.pyplot as plt
import os
import time

Y = np.load(r'../simulation/imgs/[10, 20, 50, 100]/BTAN.npy')
nx, ny = Y.shape[0], Y.shape[1]

t1 = time.time()

#%% plt heatmap
def array2heatmap(array, folderName, imgName, colorMode = 'jet'):
    with plt.ioff():
        # plt.figure(figsize=(array.shape[0], array.shape[1]))
        plt.figure()
        plt.axis('off') # 关闭坐标轴显示
        # plt.imshow(array, cmap='gray') 
        plt.imshow(array, cmap=colorMode)
        path = './imgs/'
        if not os.path.exists(path):
            os.mkdir(path)
        path += ('fre'+folderName+',')
        plt.savefig(path+imgName+'.png', bbox_inches = 'tight', pad_inches = 0)
        plt.close()

#%% FFT func
def FFT(mixture):
    f = np.fft.fft2(mixture)
    # decentration
    fShift = np.fft.fftshift(f)
    return fShift

def FFTfilter(fShift, rangeBound, freBound):
    '''

    Parameters
    ----------
    fShift : complex array
        the FFT result after decentration.
    rangeBound : float
        the control limit of range.
    freBound : float
        the control limit of frequency.

    Returns
    -------
    filFShift : complex array
        the result after filteration, allowing the signal with low range and high frequency.

    '''
    filFShift = np.zeros((nx,ny),dtype=complex)
    for i in range(nx):
        for j in range(ny):
            # remove the noise component with high range and low frequency
            if np.log(np.abs(fShift[i][j])) < rangeBound and abs(fShift[i][j].imag) < freBound:
                filFShift[i][j] = complex(1)
            else:
                filFShift[i][j] = fShift[i][j]
    return filFShift

def invFFT(fShift):
    iShift = np.fft.ifftshift(fShift)
    invRes = np.fft.ifft2(iShift)
    return invRes

for freQuantile in [5,25,50,75,95]:
    fShift = FFT(Y)
    filFShift = FFTfilter(fShift, rangeBound=inf, freBound=np.percentile(abs(fShift[:][:].imag), (freQuantile)))
    invRes = invFFT(filFShift)
    
    fImg = np.log(np.abs(fShift)) # transform to img
    filFImg = np.log(np.abs(filFShift)) # transform to img
    filterImg = np.log(np.abs(invRes))
    
    array2heatmap(fImg, str(freQuantile), 'anomaly_FFT', colorMode='gray')
    array2heatmap(filFImg, str(freQuantile), 'anomaly_filterFFT', colorMode='gray')
    array2heatmap(filterImg, str(freQuantile), 'anomaly_filtered')

t2 = time.time()
print((t2-t1)/5)