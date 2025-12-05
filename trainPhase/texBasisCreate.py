# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:12:23 2023

@author: Song Ji
"""

import numpy as np
import os
import cv2
from copy import deepcopy
from math import atan2, degrees

import sys
sys.path.append('../visualization')
from pngPlot import Plotter

#%% executable classes and funcs
class KNBN:
    def __init__(self, img_array, sizey):
        self.Y = img_array
        self.sizey = sizey
    
    def dimReduce(self, tex):
        DRtex = []
        for i in range(self.sizey[0]):
            for j in range(self.sizey[1]):
                if tex[i][j] != 0:
                    DRtex.append((i, j, tex[i][j]))
                else:
                    continue
        return DRtex
    
    def ifCanExtCheck(self, point, points, stepLen, direction):
        '''
    
        Parameters
        ----------
        point : list of [x, y, value]
            the center point to be extended.
        points : list of N*[x, y, value]
            the dimReduced version of a texture.
        stepLen: positive int
            the maximum extend distance for each dimension of the center point
    
        Returns
        -------
        isExtendable : boolean
            if the point can be further extended, return True.
    
        '''
        isExtendable = False
        extSeeds = []
        x, y = point[0], point[1]
        Dirx, Diry = direction[0], direction[1]
        pointsList = (np.array(points)[:, 0:2]).tolist()
    
        for iStep in range(1, stepLen+1):
            for isOpposite in [-1,1]:
                nextX = x + iStep*Dirx*isOpposite
                nextY = y + iStep*Diry*isOpposite
                if [nextX, nextY] in pointsList:
                    isExtendable = True
                    extSeeds.append(points[pointsList.index([nextX, nextY])])
        
        return isExtendable, extSeeds
    
    def mainLoop(self, K, ratio, Dir, stepLen=1):
        self.DRtex = self.dimReduce(self.Y)
        texBasis = []
        tex = deepcopy(self.DRtex)
        avaiSeeds = [tex[0]]
        while len(tex) != 0:
            base = []
            if len(avaiSeeds) == 0:
                avaiSeeds = [tex[0]]
            while len(avaiSeeds) > 0:
                # print(len(base))
                seed = avaiSeeds[0]
                isExtendable, extSeeds = self.ifCanExtCheck(seed, tex, stepLen, Dir)
                if isExtendable:
                    base.append(seed)
                    avaiSeeds.remove(seed)
                    avaiSeeds.extend(extSeeds)
                    avaiSeeds = list(set([tuple(t) for t in avaiSeeds]))
                    try:
                        tex.remove(seed)
                    except ValueError:
                        print(seed)
                else:
                    base.append(seed)
                    avaiSeeds.remove(seed)
                    tex.remove(seed)
                if len(tex) == 0:
                    break
            if len(base) >= K/ratio:
                texBasis.append(base)
                print("One basis recorded")
            # else:
            #     tex.extend(base)
            print(len(tex))
        print('Searching on Dir '+ str(Dir) + ' completed')
        return texBasis
        
def base2array(base, sizey):
    base_array = np.zeros(sizey)
    for point in base:
        i, j, value = [point[k] for k in range(3)]
        base_array[i][j] = value
    return base_array
    
class TwoPass:
    def __init__(self, img_array, sizey):
        self.Y = img_array
        self.sizey = sizey
        self.labels = np.zeros(self.sizey)
        self.cur_label = 1
        self.texBasis = []
        
    def merge_labels(self, labels, label1, label2):
        # 合并两个连通区域的标记
        for i in range(len(labels)):
            for j in range(len(labels)):
                if labels[i][j] == label2:
                    labels[i][j] = label1
        
    def mainLoop(self):
        labels = self.labels
        # pass 1: label all pixels
        for i in range(self.sizey[0]):
            for j in range(self.sizey[1]):
                if self.Y[i][j] > 0:
                    if i > 0 and labels[i-1][j] != 0:
                        labels[i][j] = labels[i-1][j]
                    elif j > 0 and labels[i][j-1] != 0:
                        labels[i][j] = labels[i][j-1]
                    else:  
                        labels[i][j] = self.cur_label
                        self.cur_label += 1
        # pass 2: merge connected pixels
        for i in range(self.sizey[0]):
            for j in range(self.sizey[1]):
                if self.labels[i][j] != 0:
                    if i > 0 and labels[i-1][j] != 0 and labels[i-1][j] != labels[i][j]:
                       self.merge_labels(labels, labels[i][j], labels[i-1][j])
                    if j > 0 and labels[i][j-1] != 0 and labels[i][j-1] != labels[i][j]:
                       self.merge_labels(labels, labels[i][j], labels[i][j-1])
        return labels
    
    def shapeAnalyse(self, texBasis, texCovArray, direction, degree_error=5):
        '''
        check whether the base extend on the perio_direction
        by calculating eigenvalues and eigenvectors
        '''
        texBasis_dir = []
        for iBase in range(len(texBasis)):
            base = texBasis[iBase]
            cov_mat = texCovArray[iBase]
            eValues, eVectors = np.linalg.eig(cov_mat)
            max_index = np.argmax(eValues)
            max_eVector = eVectors[:, max_index]
            # print(max_eVector)
            extAngle = degrees(atan2(-direction[1], direction[0]))
            pcaAngle = degrees(atan2(-max_eVector[1], max_eVector[0]))
            if abs(pcaAngle - extAngle) <= degree_error:
                texBasis_dir.append(base)
        return texBasis_dir
    
    def create(self, K, ratio, extDirs):
        self.labels = self.mainLoop()
        label_values = set()
        for row in self.labels:
            for pixel in row:
                label_values.add(pixel)
        self.label_values = list(label_values)
        
        texBasis = []
        texCovArray =[]
        for value in self.label_values:
            base = []
            base_loc = []
            base_valueSS = 0
            for i in range(self.sizey[0]):
                for j in range(self.sizey[1]):
                    if self.labels[i][j] == value and self.Y[i][j] != 0:
                        base.append([i, j, self.Y[i][j]])
                        base_valueSS += (self.Y[i][j])**2
                        base_loc.append([i,j])
            if K/ratio <= len(base):
                base_loc = np.array(base_loc)
                texCovArray.append(np.cov(base_loc.T))
                # for point in base:
                #     point[2] /= base_valueSS
                texBasis.append(base)
        
        for direction in extDirs:
            texBasis_dir = self.shapeAnalyse(texBasis, texCovArray, direction)
            self.texBasis.extend(texBasis_dir)
        
        return self.texBasis
        
class SeedFilling:
    def __init__(self, img_array, sizey):
        self.Y = img_array
        self.sizey = sizey
        

class BtCreate:
    def __init__(self, img_array, extDirs):
        self.Y = img_array
        self.sizey = self.Y.shape
        self.extDirs = extDirs
        self.texBasis = []
        
    def BtPlot(self, save_folder_path, texBasis, direction=None):
        plotter = Plotter()
        ttBasis = np.zeros(self.sizey)
        if direction != None:
            for base_id in range(len(texBasis)):
                base = texBasis[base_id]
                save_name = str(direction) + '_' + str(base_id)
                base_array = base2array(base, self.sizey)
                plotter.heatmapSave(base_array, save_folder_path, save_name)
                ttBasis += base_array
            save_name = str(direction) + '_SUM'
            plotter.heatmapSave(ttBasis, save_folder_path, save_name)
            npy_save_path = os.path.join(save_folder_path, 'texBasis_'+str(direction))
            np.save(npy_save_path, texBasis)
        else:
            for base_id in range(len(texBasis)):
                base = texBasis[base_id]
                save_name = str(base_id)
                base_array = base2array(base, self.sizey)
                plotter.heatmapSave(base_array, save_folder_path, save_name)
                ttBasis += base_array
            save_name = 'SUM'
            plotter.heatmapSave(ttBasis, save_folder_path, save_name)
            npy_save_path = os.path.join(save_folder_path, 'texBasis')
            np.save(npy_save_path, texBasis)
    
    def create(self, method, plot_save_path):
        if method == 'KNBN':
            for direction in self.extDirs:
                ''' KNBN method '''
                KNBN_creator = KNBN(self.Y, self.sizey)
                K, ratio = 500, 20
                texBasis = KNBN_creator.mainLoop(K, ratio, direction)
                self.BtPlot(plot_save_path, texBasis, direction)
                self.texBasis.extend(texBasis)
        elif method == 'TP':
            ''' Two Pass method '''
            tp_creator = TwoPass(self.Y, self.sizey)
            K, ratio = 500, 20
            texBasis = tp_creator.create(K, ratio, self.extDirs)
            self.BtPlot(plot_save_path, texBasis)
            self.texBasis.extend(texBasis)
        return self.texBasis

#%% main funcs
if __name__ == "__main__":
    mfImg_path = '../../result/trainPhase/0'
    plot_save_path = '../../result/trainPhase/0/texBasis'
    
    try:
        os.makedirs(plot_save_path)
    except FileExistsError:
        pass
    
    mfImg_name = 'gamma0.2_MFmix.png'
    
    mfImg = cv2.imread(os.path.join(mfImg_path, mfImg_name))
    mfImg = cv2.cvtColor(mfImg, cv2.COLOR_BGR2GRAY)
    mfImg_array = np.array(mfImg, dtype=np.float64)
    mfImg_array -= np.ones(mfImg_array.shape) * np.min(mfImg_array)
    
    # params
    extDirs = [(1,-1)]
    
    creator = BtCreate(mfImg_array, extDirs)
    Bt = creator.create('TP', plot_save_path)