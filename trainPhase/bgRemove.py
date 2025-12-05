# -*- coding: utf-8 -*-
"""
Revised according to Hao Yan, August 2015
Coded by Song Ji
"""

import numpy as np
from numpy.linalg import svd, solve, inv, norm
from scipy.interpolate import splev
from math import inf
import cv2
from PIL import Image
import os
from copy import deepcopy

import sys
sys.path.append('../visualization')
from pngPlot import Plotter

#%% executable classes and funcs
class FileReader:
    def __init__(self, root_path):
        self.root = root_path
        self.fileCount()
        
    def fileCount(self):
        self.file_names = [f for f in os.listdir(self.root) \
                           if os.path.isfile(os.path.join(self.root, f))]
        self.file_num = len(self.file_names)
        if self.file_num > 0:
            self.start = 0
            self.end = self.file_num - 1
        else:
            self.start = -1
            self.end = -1
    
    def fileFilter(self, start, end, prefix='', suffix='', file_type='*'):
        '''
        filter files with names like "[prefix] + [id] + [suffix] + .[file_type]"
        from start to end in the list of self.file_names
        '''
        start = int(max(self.start, min(self.end, start)))
        end = int(max(start, min(self.end, end)))
        fFile_names = []
        if self.start < 0:
            # no file can be read in the root path
            print("Couldn't find required files")
            return fFile_names
        else:
            if file_type == '*':
                for iFile in range(start, end):
                    fFile_names.append(self.file_names[iFile])
            else:
                for iFile in range(start, end):
                    if self.file_names[iFile].split('.')[1] == file_type:
                        fFile_names.append(self.file_names[iFile])
            return fFile_names
    
    def imgRead(self, start=None, end=None, prefix='', suffix='', file_type='png'):
        if start == None:
            start = self.start
        if end == None:
            end = self.end
        fFile_names = self.fileFilter(start, end, prefix, suffix, file_type)
        imgs = []
        for img_name in fFile_names:
            img = cv2.imread(os.path.join(self.root, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgs.append(img)
        return imgs
    
class ImgFilter:
    def __init__(self):
        return
    
    def meanFilter(self, img_array, directions, stepLen=1, percent = 0.8, quantile = 0.95):
        '''
        Parameters
        ----------
        array : numpy array
            array to proceed.
        directions : list of binary tuple
            which directions to be averaged,
            consider the extension directions of the texture,
            need further improvement to connect the discontinuities.
    
        Returns
        -------
        newarray : numpy array
            array proceeded.
    
        '''
        self.Y = deepcopy(img_array)
        self.sizey = self.Y.shape
        # self.Y -= np.ones(self.sizey) * np.min(self.Y)
        
        newarray = np.zeros(self.sizey)
        bound = np.percentile(self.Y, quantile)
        for x in range(self.sizey[0]):
            for y in range(self.sizey[1]):
                values = [self.Y[x][y]]
                for Dir in directions:
                    Dirx, Diry = Dir[0], Dir[1]
                    for iStep in range(1, stepLen+1):
                        for isOpposite in [-1,1]:
                            if 0 <= x+Dirx*iStep*isOpposite < self.sizey[0] and 0 <= y+Diry*iStep*isOpposite < self.sizey[1]:
                                value = self.Y[x+Dirx*iStep*isOpposite][y+Diry*iStep*isOpposite]
                                values.append(value)
                    newarray[x][y] = np.mean(values)
        return newarray
    
    def perFilter(self, img_array, percent):
        '''
        Parameters
        ----------
        imgArray : array
            img data after FFT, filter and invFFT.
        percent: int
            determine the quantile to remove
    
        Returns
        -------
        imgArray : array
            img data with all negative values set to zero, only texture kept.
    
        '''
        self.Y = deepcopy(img_array)
        self.sizey = self.Y.shape
        
        q_value = np.percentile(self.Y, (percent))
        for i in range(self.sizey[0]):
            for j in range(self.sizey[1]):
                if self.Y[i][j] < q_value:
                    self.Y[i][j] = 0
                else:
                    self.Y[i][j] -= q_value
        return self.Y

class ImgDecomposite:
    def __init__(self, img, img_array):
        self.img = img
        self.Y = img_array
        self.sizey = self.Y.shape
        
    def bsplineBasis(self, n, k, sd):
        '''
        bsplineBasis: Construct k Bspline Basis with n gridded with spline degree sd
    
        Parameters
        ----------
        n : int
            length of signal or number of pixels, 1-dimension.
        k : int
            number of knots, representing the smooth degree of the background.
            if k == n, sd must be zero, leading to an identity matrix as return.
        sd : int
            spline degree. representing the scale of the anomaly.
            if sd == 0, we use a constant function to do interpolation, usually we have sd == 3
        bd : int
            how many basis in boundary.
    
        Returns
        -------
        B : a (bd)*n matrix.
    
        '''
        if n == k:
            B = np.eye(n) # generate an identity matrix with size of n
        else:
            knots = np.r_[np.ones(sd), np.linspace(1, n, k), (n)*np.ones(sd)] # to generate the dot matrix for input of B-spline
            x = np.linspace(1, n, n) # the control points, ranging from 1 to n
            nKnots = len(knots) - sd - 1
            v = np.zeros((nKnots, n))
            d = np.eye(nKnots, n) # the cofficients matrix for the following B-spline func generation
            for i in range(nKnots):
                v[i] = splev(x, (knots, d[i], sd))
            B = v.T
        return B
        
    def bgFitting(self, bparamsxy, lambdaxy, allgamma, quantile, maxIter):
        n_dim = len(self.sizey)
        n_gamma = len(allgamma)
        '''
        both gray image, for dynamic image data, ndim = 3, 
        while ndim = 2 for single image data
        ndim is the num of dimensions, nT is the num of images
        '''
        
        # define additional functions
        softthreshold = lambda residual,gamma : np.sign(residual)*np.maximum(np.abs(residual) - gamma, 0)
        softthreshold_outlier = lambda residual,gamma : residual*(np.ones(residual.shape) - np.sign(np.abs(residual) - gamma))
        constructD =lambda n: np.diff(np.eye(n),1,axis=0);
        '''
        softthreshold(residual, gamma): Set the positive value gamma as the threshold, 
                                        record all values in residual exceeding gamma 
                                        and the corresponding error
        constructD(n): create an identity mat with size == n, 
                        each column do the diff operation for one time
                        the outcome is an (n-1)*n order difference mat
        '''
        
        (nx,kx,sdx), (ny,ky,sdy) = bparamsxy[0], bparamsxy[1]
        self.B = [self.bsplineBasis(nx,kx,sdx),  self.bsplineBasis(ny,ky,sdy)]
        '''
        the B-spline basis are designed for single image processing, 2-dimensional
        while the dynamic data is (t,x,y), so complete the dimension
        '''
    
        D = [[] for i in range(n_dim)]
        H = [[] for i in range(n_dim)]
    
        for iDim in range(n_dim):
            if self.B[iDim] is None:
                self.B[iDim] = np.eye(self.sizey[iDim])
                # if the basis are accidentally empty, 
                # follow the n==k situation to set the basis as an identity mat
            D[iDim] = constructD(self.B[iDim].shape[1])
            H[iDim] = self.B[iDim] @ solve(self.B[iDim].T @ self.B[iDim] + lambdaxy[iDim] * (D[iDim].T @ D[iDim]), self.B[iDim].T)
            '''
            "solve(A, B)" == "inv(B) @ A"
            '''
        
        Bgs = [[] for i in range(n_gamma)]
        Mixtures = [[] for i in range(n_gamma)]
        
        for i in range(n_gamma):
            # y = maxnorm(Y)
            '''
            对Y正则化会极大地削弱gamma对结果的影响，迭代次数超过2会使bg崩坏，texture中大量噪声
            '''
            y = self.Y
            Mixture = np.zeros(self.sizey)
            for iIter in range(maxIter):
                if iIter == 0:
                    Bg = H[0]@(y-Mixture)@H[1]
                    # Bg = maxnorm(Bg)
                
                residual = y - Bg
                if iIter < maxIter - 1:
                    Mixture = softthreshold(residual, allgamma[i])
                else:
                    Mixture = softthreshold(residual, np.percentile(residual, quantile))
                    # Mixture = softthreshold_outlier(residual, np.percentile(residual, 75))
    
            Bgs[i] = Bg
            Mixtures[i] = Mixture
    
        return Bgs, Mixtures


#%% main func
if __name__ == "__main__":
    img_path = '../../dataset/3Dprint/DATASET/45/Phase_I'
    plot_save_path = '../../result/trainPhase'
    
    try:
        os.makedirs(plot_save_path)
    except FileExistsError:
        pass
    
    reader = FileReader(img_path)
    train_imgs = reader.imgRead()
    
    # global params
    lambdaxy = [0.1,0.1]
    # allgamma = [1e-2,5e-2,1e-1,2e-1,5e-1,1]
    # allgamma = [1e-2,5e-2,1e-1,1.2e-1,1.4e-1,1.6e-1,1.8e-1,2e-1]
    allgamma = [2e-1]
    quantile = 75
    maxIter = 2
    extDirs = [(1,-1)] # output from perioCheck.py
    percent = 50
    
    plotter = Plotter()
    
    for img in train_imgs:
        img_array = np.array(img)
        # local params
        bparamsxy = [(img.shape[0], 2, 3), (img.shape[1], 2, 3)]
        
        decompositer = ImgDecomposite(img, img_array)
        Bgs, Mixtures = decompositer.bgFitting(bparamsxy, lambdaxy, allgamma, quantile, maxIter)
        
        imgFilter = ImgFilter()
        MFmixes = [imgFilter.perFilter(imgFilter.meanFilter(Mixtures[iGamma], extDirs), percent=75) for iGamma in range(len(allgamma))]
        # perMixes = [imgFilter.perFilter(MFmixes[iGamma], percent) for iGamma in range(len(allgamma))]
        
        # plotter.heatmapPlot(Bgs[0])
        # plotter.heatmapPlot(Mixtures[0])
        
        plotter.heatmapSave(img_array, os.path.join(plot_save_path, '0'), '@Y')
        plotter.heatmapSave(Bgs[0], os.path.join(plot_save_path, '0'), 'gamma'+str(allgamma[0])+'_Bg')
        plotter.heatmapSave(Mixtures[0], os.path.join(plot_save_path, '0'), 'gamma'+str(allgamma[0])+'_Mix')
        plotter.heatmapSave(MFmixes[0], os.path.join(plot_save_path, '0'), 'gamma'+str(allgamma[0])+'_MFmix')
        # plotter.heatmapSave(perMixes[0], os.path.join(plot_save_path, '0'), 'gamma'+str(allgamma[0])+'_perMix')
        
        break
    
