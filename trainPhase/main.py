# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:43:57 2023

@author: Song Ji
"""

import numpy as np
import os
import time
from bgRemove import FileReader, ImgDecomposite, ImgFilter
from perioCheck import PeriodicityCheck
from texBasisCreate import BtCreate

import sys
sys.path.append('../visualization')
from pngPlot import Plotter


#%% main func
if __name__ == "__main__":
    
    input_img_path = '../../dataset/3Dprint/DATASET/45/Phase_I'
    savePath_root = '../../result/trainPhase'
    savePath_LRD = os.path.join(savePath_root, 'main')
    savePath_perio = os.path.join(savePath_LRD, 'perioInfo')
    savePath_Bt = os.path.join(savePath_LRD, 'texBasis')
    savePaths = [savePath_root, savePath_LRD, savePath_perio, savePath_Bt]
    
    for path in savePaths:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
    
    reader = FileReader(input_img_path)
    train_imgs = reader.imgRead()
    
    plotter = Plotter()
    Bt = []
    for img_id in range(5):
        
        img = train_imgs[img_id]
        img_array = np.array(img)
        print("====== Processing Img " + str(img_id) + " =====")
        
        ''' Low-rank decomposition phase '''
        # global params
        lambdaxy = [0.1,0.1]
        # allgamma = [1e-2,5e-2,1e-1,2e-1,5e-1,1]
        # allgamma = [1e-2,5e-2,1e-1,1.2e-1,1.4e-1,1.6e-1,1.8e-1,2e-1]
        allgamma = [2e-1]
        quantile = 75
        maxIter = 2

        # local params
        bparamsxy = [(img.shape[0], 2, 3), (img.shape[1], 2, 3)]
        
        decompositer = ImgDecomposite(img, img_array)
        Bgs, Mixtures = decompositer.bgFitting(bparamsxy, lambdaxy, allgamma, quantile, maxIter)
        Bg, Mix = Bgs[0], Mixtures[0]
        
        # plotter.heatmapPlot(Bgs[0])
        # plotter.heatmapPlot(Mixtures[0])
        
        plotter.heatmapSave(img_array, savePath_LRD, '@Y_'+str(img_id))
        plotter.heatmapSave(Bg, savePath_LRD, 'Bg_'+str(img_id))
        plotter.heatmapSave(Mix, savePath_LRD, 'Mix_'+str(img_id))
        
        ''' Periodicity check phase '''
        if img_id == 0:
            # params
            maxRotate = 36
            sampleWidth = 3
            sampleGap = 5
            
            perioChecker = PeriodicityCheck(None, Mix)
            perioChecker.rotate(maxRotate, sampleWidth, sampleGap)
            extAngles, extDirs = perioChecker.resultPlot(maxRotate, savePath_perio)
        
        ''' texture basis learning phase '''
        imgFilter = ImgFilter()
        MFmix = imgFilter.perFilter(imgFilter.meanFilter(Mix, extDirs), percent=75)
        plotter.heatmapSave(MFmix, savePath_LRD, 'MFmix_'+str(img_id))
        
        savePath_Bt_unit = os.path.join(savePath_Bt, str(img_id))
        try:
            os.makedirs(savePath_Bt_unit)
        except FileExistsError:
            pass
        Btcreator = BtCreate(MFmix, extDirs)
        Bt_unit = Btcreator.create('TP', savePath_Bt_unit)
        Bt.extend(Bt_unit)
    
    np.save(os.path.join(savePath_Bt, 'texBasis.npy'), Bt)
