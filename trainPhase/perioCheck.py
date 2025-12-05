# -*- coding: utf-8 -*-
"""
Revised according to Hao Yan, August 2015
Coded by Song Ji
"""

import numpy as np
from pandas import value_counts
import cv2
from math import sin, cos, pi
import os

import sys
sys.path.append('../visualization')
from pngPlot import Plotter

#%% executable classes and funcs
class PeriodicityCheck:
    def __init__(self, img, img_array):
        self.img = img
        self.Y = img_array
        self.sizey = self.Y.shape
    
    def flatten(self, sample):
        '''
        Parameters
        ----------
        sample : [[basis pixel], [samples on direction 1], [samples on direction 2]] list
            a sample under specific translation distance under specific rotation angle.
    
        Returns
        -------
        flattenSample : [pixels]
            the flatten version of the origin sample from one end to the other end.
    
        '''
        flattenSample = []
        
        basBranch = sample[0]
        negBranch = sample[1]
        posBranch = sample[2]
        
        # calculate the mean value of each sample unit
        for i in range(len(negBranch)):
            flattenSample.append(np.mean(negBranch[len(negBranch)-1-i]))
        for i in range(len(basBranch)):
            flattenSample.append(np.mean(basBranch[i]))
        for i in range(len(posBranch)):
            flattenSample.append(np.mean(posBranch[i]))
        
        return flattenSample
    
    def translate(self, coreX, coreY, transDist, width, angle, samples, stats, multiplier):
        for transDir in [-1, 1]:
            '''
            transDir: the direction for translation
            extDir: the direction for extension
            neighDir: the direction for neighbor finding
            
            use the symmetry, extend to the two different side of the bench curve
            calculate the furthest distance among vertex to the curve
            keep the transDist in the ranglee
            
            if basic pixel in the img, extend to two ends until
            both ends touch the img edge, then break the loop
            '''
            curSample = [[], [], []]
            curStats = 0
            # get the basis pixel for current sample curve
            curX = coreX - round(transDir * transDist * cos(angle))
            curY = coreY - round(transDir * transDist * sin(angle))
            
            # extend to the two ends of the curve, until the img edges 
            if angle == 0 or angle == pi:
                maxStep = self.sizey[1]
            elif angle == pi/2 or angle == pi/2*3:
                maxStep = self.sizey[0]
            else:
                maxStep = max(round(abs(self.sizey[0]/sin(angle))), round(abs(self.sizey[1]/cos(angle))))
                
            endCollision = [False, False]    
            for step in range(maxStep):
                for extDir in [-1, 1]:
                    if not endCollision[int((extDir+1)/2)]:
                        '''
                        if the end in current direction haven't be out of the img,
                        then continue extension
                        '''
                        neighbors = round((width - 1) / 2)
                        curX = curX - round(extDir*step*sin(angle))
                        curY = curY + round(extDir*step*cos(angle))
                        if 0 <= curX < self.sizey[0] and 0 <= curY < self.sizey[1]:
                            sampleUnit = []
                            sampleUnit.append(self.Y[curX][curY])
                            for iNe in range(neighbors):
                                for neighDir in [-1,1]:
                                    neighborX = curX - round(neighDir * transDir * (iNe+1) * cos(angle))
                                    neighborY = curY - round(neighDir * transDir * (iNe+1) * sin(angle))
                                    if 0 <= neighborX < self.sizey[0] and 0 <= neighborY < self.sizey[1]:
                                        sampleUnit.append(self.Y[neighborX][neighborY])
                            if step == 0:
                                curSample[0].append(sampleUnit)
                                break
                            else:
                                curSample[int((extDir+1)/2)+1].append(sampleUnit)
                        else:
                            if step == 0:
                                endCollision[0] = True
                                endCollision[1] = True
                                break
                            else:
                                endCollision[int((extDir+1)/2)] = True
                if all(endCollision) == True:
                    flattenSample = self.flatten(curSample)
                    res = value_counts(flattenSample, normalize=True)
                    for key in res.index:
                        curStats += key * res[key]
                    break              
            
            if curSample[0] != []:
                samples.append(curSample)
                stats.append(curStats)
            
            if multiplier == 0:
                break
    
    def linearSampling(self, angle, method, width, gap, params=[], bench='center'):
        '''
    
        Parameters
        ----------
        angle : float, in radian
            the given rotation angle for the standard sample curve.
        method : str
            the sample curve's code name. 'Linear' means a straight sampling line
        width : int
            determine how many pixels would be taken into account on the orthogonal direction
            when sampling along the curve.
        gap : int
            determine how long would be jumped between sample curves.
            rounded in pixels
        params : a list
            necessary params for specific sample curve. [] for Linear method
        bench : str, optional
            the static pixel among different angles
            all curves start from it or pixels translated from it
            The default is 'center', thus the center pixel in the img data
    
        Returns
        -------
        samples : nRotate*(nSampling*[[basis pixel], [samples on direction 1], [samples on direction 2]]) list
            contain all sample lists with different position and length under current angle.
            each sample contain a sample unit with all possible neighboring pixels
        stats : nRotate*(nSampling*1) list
            give an overall statistical description on the periodicity under current angle.
    
        '''
        samples = []
        stats = []
        
        if bench == 'center':
            # find the center pixel
            coreX, coreY = round(self.sizey[0]/2), round(self.sizey[1]/2)
            
            # start sampling
            multiplier = 0
            sample_allPossible = True
            
            while sample_allPossible:
                # calculate the translation distance
                transDist = gap*multiplier
                
                if angle == 0 or angle == pi:
                    maxDist = self.sizey[0]/2
                elif angle == pi/2 or angle == pi/2*3:
                    maxDist = self.sizey[1]/2
                else:
                    maxDist = (self.sizey[0]*abs(cos(angle)) + self.sizey[1]*abs(sin(angle))) / 2
                
                if transDist <= maxDist:
                    self.translate(coreX, coreY, transDist, width, angle, samples, stats, multiplier)
                else:
                    break
                
                multiplier += 1

        return samples, stats
    
    def rotate(self, maxRotate, width, gap, method='Linear'):
        ttSamples = []
        ttStats = []
        for iRotate in range(maxRotate):
            angle = pi/maxRotate*iRotate
            samples, stats = self.linearSampling(angle, method, width, gap)
            ttSamples.append(samples)
            ttStats.append(stats)
        self.ttSamples = ttSamples
        self.ttStats = ttStats
        
    def resultPlot(self, maxRotate, plot_save_path, quantile=95):
        plotter = Plotter()
        
        for iRotate in range(maxRotate):
            plot_id = str(180/maxRotate*iRotate)
            save_name = 'Degree ' + plot_id + '.png'
            label_x, label_y = 'Sample Code', 'Weighted Gray Value'
            title = 'Rotation Degree = ' + plot_id
            x, y = [i for i in range(len(self.ttStats[iRotate]))], self.ttStats[iRotate]
            plotter.graphSave(plot_save_path, save_name, label_x, label_y, title, x, y, None)
        
        std = []
        for iRotate in range(maxRotate):
            std.append(np.std(self.ttStats[iRotate]))
        
        save_name = 'STD_GVs.png'
        label_x, label_y = 'Sample Rotation Degree', 'Weighted Gray Value STD'
        title = 'Std of GVs'
        x, y = [180/maxRotate*iRotate for iRotate in range(maxRotate)], [std[iRotate] for iRotate in range(maxRotate)]
        quantile_line = np.percentile(y, (quantile))
        plotter.graphSave(plot_save_path, save_name, label_x, label_y, title, x, y, quantile_line)
        
        save_name = 'STD_Dif_GVs.png'
        label_x, label_y = 'Sample Rotation Degree', 'Weighted GV STD deviation'
        title = 'Std Deviation of GVs'
        x = [180/maxRotate*iRotate for iRotate in range(maxRotate)]
        y = []
        ortho_rotates = int(maxRotate/2)
        for iRotate in range(ortho_rotates):
            y.append(std[iRotate]-std[maxRotate-ortho_rotates+iRotate])
        for iRotate in range(ortho_rotates, maxRotate):
            y.append(std[iRotate]-std[iRotate-ortho_rotates])
        quantile_line = np.percentile(np.abs(y), (quantile))
        plotter.graphSave(plot_save_path, save_name, label_x, label_y, title, x, y, quantile_line)
        
        self.angle_translation = x[y.index(np.max(y))]
        if self.angle_translation < 90:
            self.angle_extension = self.angle_translation + 90
        else:
            self.angle_extension = self.angle_translation - 90
            
        self.extAngles = [self.angle_extension]
        self.extDirs = self.extension_ang2dir(self.extAngles)
        
        return self.extAngles, self.extDirs
    
    def extension_ang2dir(self, extAngles):
        extDirs = []
        for angle in extAngles:
            angle = np.deg2rad(angle)
            multiplier = 1
            while True:
                delta_x, delta_y = round(np.cos(angle) * multiplier), -round(np.sin(angle) * multiplier)
                if delta_x * delta_y != 0:
                    extDirs.append((delta_x, delta_y))
                    break
                else:
                    multiplier += 1
        return extDirs

#%% main func
if __name__ == '__main__':
    mixImg_path = '../../result/trainPhase/0'
    plot_save_path = '../../result/trainPhase/0/perioInfo'
    
    try:
        os.makedirs(plot_save_path)
    except FileExistsError:
        pass
    
    mixImg_name = 'gamma0.2_Mix.png'
    
    mixImg = cv2.imread(os.path.join(mixImg_path, mixImg_name))
    mixImg = cv2.cvtColor(mixImg, cv2.COLOR_BGR2GRAY)
    mixImg_array = np.array(mixImg)
    
    # params
    maxRotate = 36
    sampleWidth = 3
    sampleGap = 5
    
    perioChecker = PeriodicityCheck(mixImg, mixImg_array)
    perioChecker.rotate(maxRotate, sampleWidth, sampleGap)
    extAngles, extDirs = perioChecker.resultPlot(maxRotate, plot_save_path)