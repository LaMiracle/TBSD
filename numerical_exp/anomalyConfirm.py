# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

#%% plt heatmap
def array2heatmap(array, folderName, imgName, colorMode = 'jet'):
    with plt.ioff():
        # plt.figure(figsize=(array.shape[0], array.shape[1]))
        plt.figure()
        plt.axis('off') # 关闭坐标轴显示
        # plt.imshow(array, cmap='gray') 
        plt.imshow(array, cmap=colorMode)
        path = './test_SD/'+imgId+'/'+patchId+'/aConfirm/'
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(path+imgName+'.png', bbox_inches = 'tight', pad_inches = 0)
        plt.close()
        
def simpleheatmap(array, colorMode = 'jet'):
    # plt.figure(figsize=(array.shape[0], array.shape[1]))
    plt.figure()
    plt.axis('off') # 关闭坐标轴显示
    # plt.imshow(array, cmap='gray') 
    plt.imshow(array, cmap=colorMode)

#%%  
def deepcopy(originArray):
    copyArray = np.zeros(originArray.shape)
    copyArray[:][:] = originArray[:][:]
    return copyArray

def points2array(points):
    array = np.zeros(sizey)
    for point in points:
        x, y = int(point[0]), int(point[1])
        array[x][y] = 1
    return array

def ExtendExam(point, points, stepLen):
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
    pointsList = (np.array(points)[:, 0:2]).tolist()
    
    for i in range(-stepLen, stepLen+1):
        for j in range(-stepLen, stepLen+1):
            if i == 0 and j == 0:
                continue
            else:
                nextX = x + i
                nextY = y + j
                if [nextX, nextY] in pointsList:
                    isExtendable = True
                    extSeeds.append(points[pointsList.index([nextX, nextY])])
    return isExtendable, extSeeds

def EVfilter(anomaly, sigma, alpha, mode):
    # lb, ub = np.percentile(anomaly, (alpha[0])), np.percentile(anomaly, (100-alpha[1]))
    # sigma = np.std(anomaly)
    # print(sigma)
    mu = np.mean(anomaly)
    lb, ub = mu - alpha*sigma, mu + alpha*sigma
    
    extremeList = []
    for i in range(anomaly.shape[0]):
        for j in range(anomaly.shape[1]):
            if anomaly[i][j] < lb or anomaly[i][j] > ub:
                extremeList.append([i,j,anomaly[i][j]])
    EVarray = np.array(extremeList)
    np.save('./aConfirm/extremeList.npy', EVarray)
    return extremeList

def anomalyGenerate(anomaly, sigma, alpha, K, ratio, mode='two-sides'):
    Ahat = np.zeros(anomaly.shape)
    EVarray = EVfilter(anomaly, sigma, alpha, mode)
    EVbasis = []
    
    avaiSeeds = [EVarray[0]]
    while len(EVarray) != 0:
        base = []
        if len(avaiSeeds) == 0:
            avaiSeeds = [EVarray[0]]
        while len(avaiSeeds) > 0:
            # print(len(base))
            seed = avaiSeeds[0]
            isExtendable, extSeeds = ExtendExam(seed, EVarray, stepLen=1)
            if isExtendable:
                base.append(seed)
                avaiSeeds.remove(seed)
                avaiSeeds.extend(extSeeds)
                avaiSeeds = list(set([tuple(t) for t in avaiSeeds]))
                EVarray.remove(list(seed))
                if len(base) >= K:
                    EVbasis.append(base)
                    break
            else:
                avaiSeeds.remove(seed)
                EVarray.remove(list(seed))
                if len(base) >= K/ratio:
                    EVbasis.append(base)
                    break
            if len(EVarray) == 0:
                if len(base) >= K/ratio:
                    EVbasis.append(base)
                break
        print(len(EVarray))
    
    print('Neighbor Searching Completed')
    for basis in EVbasis:
        for point in basis:
            x, y = point[0], point[1]
            Ahat[x][y] = 1
    
    return Ahat, EVbasis

def anomalyGenerate_v2(anomaly, sigma, alpha, K, maxDist, mode='two-sides'):
    Ahat = np.zeros(anomaly.shape)
    EVlist = EVfilter(anomaly, sigma, alpha, mode)
    # EVarray = np.array(EVlist)
    EVgroups = []
    
    iPoint1 = 0
    while True:
        if iPoint1 >= len(EVlist):
            break
        p1 = EVlist[iPoint1]
        isAbsorded = False
        for group in EVgroups:
            groupArray = np.array(group)
            x_bar, y_bar = int(np.mean(groupArray[:,0])), int(np.mean(groupArray[:,1]))
            if np.sqrt((p1[0]-x_bar)**2+(p1[1]-y_bar)**2) <= maxDist:
                group.append(p1)
                EVlist.remove(p1)
                isAbsorded = True
                break
        if len(EVlist) == 0:
            break
        if not isAbsorded:
            EVgroup = []
            EVgroup.append(p1)
            EVlist.remove(p1)
            if len(EVlist) == 0:
                break
            
            iPoint2 = iPoint1
            while True:
                if iPoint2 >= len(EVlist):
                    break
                p2 = EVlist[iPoint2]
                if np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2) <= maxDist:
                    EVgroup.append(p2)
                    EVlist.remove(p2)
                else:
                    iPoint2 += 1
                    if iPoint2 >= len(EVlist):
                        EVgroups.append(EVgroup)
                        break
                if len(EVlist) == 0:
                    EVgroups.append(EVgroup)
                    break
            
    
    print('Neighbor Searching Completed')
    
    iGroup = 0
    while True:
        if iGroup >= len(EVgroups):
            break
        group = EVgroups[iGroup]
        if len(group) <= K:
            EVgroups.remove(group)
        else:
            iGroup += 1
        if len(EVgroups) <= 0:
            break
    print('Noise Groups Deleted')
    
    for group in EVgroups:
        for point in group:
            x, y = int(point[0]), int(point[1])
            Ahat[x][y] = 1
    
    return Ahat, EVgroups

def groupDistRes(group1, group2):
    array1, array2 = np.array(group1), np.array(group2)
    x1_bar, y1_bar = np.mean(array1[:,0]), np.mean(array1[:,1])
    x2_bar, y2_bar = np.mean(array2[:,0]), np.mean(array2[:,1])
    dist = np.sqrt((x1_bar-x2_bar)**2+(y1_bar-y2_bar)**2)
    return dist
 
def anomalyGroup(EVgroups, maxGroupDist):
    anomalyGroups = []
    for group in EVgroups:
        anomalyGroups.append(deepcopy(np.array(group)).tolist())
    iGroup1 = 0
    while True:
        if iGroup1 >= len(anomalyGroups):
            break
        group1 = anomalyGroups[iGroup1]
        iGroup2 = 0
        isGroupFinish = False
        while True:
            if iGroup2 == iGroup1:
                if iGroup1 == len(anomalyGroups)-1:
                    isGroupFinish = True
                    break
                else:
                    iGroup2 += 1
            else:
                group2 = anomalyGroups[iGroup2]
                if groupDistRes(group1, group2) <= maxGroupDist:
                    group1.extend(group2)
                    anomalyGroups.remove(group2)
                    break
                else:
                    iGroup2 += 1
                    if iGroup2 >= len(anomalyGroups):
                        iGroup1 += 1
                        break
        if isGroupFinish:
            break
    
    # for iGroup in range(len(anomalyGroups)):
    #     group = anomalyGroups[iGroup]
    #     anomalyArray = points2array(group)
    #     array2heatmap(anomalyArray, None, imgName='maxGroupDist'+str(maxDist)+',group'+str(iGroup), colorMode='gray')
    
    return anomalyGroups


#%%
# def anomalyIdle(anomaly):
#     Aidle = np.zeros(anomaly.shape)
#     for i in range(anomaly.shape[0]):
#         for j in range(anomaly.shape[1]):
#             if anomaly[i][j] == 0:
#                 Aidle[i][j] = 1
#     return Aidle
    
# def imgFilter(array, alpha, mode='two-sides'):
#     filterA = np.zeros(array.shape)
#     signFA = np.zeros(array.shape)
#     lb, ub = np.percentile(array, (alpha)), np.percentile(array, (100-alpha))
#     for i in range(array.shape[0]):
#         for j in range(array.shape[1]):
#             if array[i][j] > lb or array[i][j] < ub:
#                 filterA[i][j] = array[i][j]
#                 signFA[i][j] = 1
#     return filterA, signFA

#%% main func
def main(imgId, patchId):
    path = './test_SD/'+imgId+'/'+patchId+'/'
    As = np.load(path + 'fAnomalies.npy')
    
    # img = cv2.imread("./plt_figures/known, combined gap(10,20,50,100)/perfect basis/0.2/anomaly.png", flags=2) # read as gray img
    # anomaly = np.array(img)
    anomaly = As[0]
    # Aidle = anomalyIdle(anomaly)
    
    # simpleheatmap(anomaly)
    # reviseA = np.sign(anomaly) * (10 * np.power(np.abs(anomaly), 1/2))
    # filterA, signFA = imgFilter(anomaly, alpha=10)
    # simpleheatmap(filterA)
    # simpleheatmap(signFA)
    
    K = 15
    sigma = 20
    alpha = 3
    maxDist = 10
    maxGroupDist = 20
    Ahat, EVgroups = anomalyGenerate_v2(anomaly, sigma, alpha, K, maxDist)
    anomalyGroups = anomalyGroup(EVgroups, maxGroupDist)
    
    anomalyWarningRatio = 0.01
    positiveNum = np.sum(Ahat)  
    ratio = positiveNum/(sizey[0]*sizey[1])
    if ratio >= anomalyWarningRatio:
        print("Anomaly detected, exception ratio = "+str(ratio))
    else:
        print("Exception under control, exception ratio = "+str(ratio))
    
    array2heatmap(Ahat, None, imgName='K'+str(K)+',sigma'+str(sigma)+',alpha'+str(alpha)+',maxDist'+str(maxDist), colorMode='gray')
    np.save(path+'/aConfirm/K'+str(K)+',sigma'+str(sigma)+',alpha'+str(alpha)+',maxDist'+str(maxDist)+'.npy', Ahat)
    np.save(path+'/aConfirm/K'+str(K)+',sigma'+str(sigma)+',alpha'+str(alpha)+',maxDist'+str(maxDist)+'_groups.npy', EVgroups)
    np.save(path+'/aConfirm/K'+str(K)+',sigma'+str(sigma)+',alpha'+str(alpha)+',maxDist'+str(maxDist)+'_basis.npy', anomalyGroups)
    
    return ratio

sizey = [256,256]
# imgIds = ['wood_hole00'+str(i) for i in range(0,1)]
imgIds = ['wood_hole00'+str(i) for i in range(10)]
# patchIds = [str([i,j]) for i in range(4) for j in range(4)]
patchIds = [str([2,2])]
positiveRatios = np.zeros((10,4,4))
i, j, k = 0, 0, 0
for imgId in imgIds:
    for patchId in patchIds:
        # if imgId not in ['wood_hole00'+str(i) for i in [0,3,4,5,6,9]]:
        if imgId in ['wood_hole00'+str(i) for i in [0]]:
            positiveRatios[i, j, k] = main(imgId, patchId)
        else:
            positiveRatios[i, j, k] = 0
        j += 1
        if j == 4:
            j = 0
            k += 1
    j = 0
    k = 0
    i += 1
