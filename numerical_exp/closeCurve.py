# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from math import pi,sqrt,sin,cos,floor, atan2
from scipy.interpolate import splprep, splev, interp1d
from scipy.spatial import ConvexHull
import cv2

#%% plt heatmap
def array2heatmap(array, path, imgName, colorMode = 'jet'):
    with plt.ioff():
        # plt.figure(figsize=(array.shape[0], array.shape[1]))
        plt.figure()
        plt.axis('off') # 关闭坐标轴显示
        # plt.imshow(array, cmap='gray') 
        plt.imshow(array, cmap=colorMode)
        # path = './aConfirm/closedAnomaly/'
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

#%% rotated edge detection
def positionArray(centerP, basis):
    '''
    create an array contain all (distance,angle) of each point to the center point in the same basis
    '''
    positionArray = []
    for point in basis:
        difX = point[0] - centerP[0]
        difY = point[1] - centerP[1]
        distance = sqrt(difX**2 + difY**2)
        # cosAng = difX/distance
        # sinAng = difY/distance
        
        # ang = np.arcsin(sinAng)
        # if cosAng < 0:
        #     if sinAng >= 0:
        #         ang = pi/2 + ang
        #     else:
        #         ang = pi - ang
        # else:
        #     if sinAng < 0:
        #         ang = 2*pi + ang
        
        ang = atan2(difY, difX)
        ang = -int(ang * 180 / pi)
        if ang < 0:
            ang = 360 + ang
        ang = ang / 180 * pi
        
        positionArray.append((point[0], point[1], distance, ang))
    positionArray = np.array(positionArray)
    return positionArray

def centerPointFind(basis):
    basisArray = np.array(basis)
    basisX, basisY = basisArray[:,0], basisArray[:,1]
    centerPoint = (np.mean(basisX), np.mean(basisY))
    return centerPoint

def edgePointsFind(posArray, maxRotate):
    edgePoints = []
    pointGroup = [[] for i in range(round(maxRotate))]
    for point in posArray:
        [x, y, dist, angle] = point
        groupId = floor(angle / (2*pi/maxRotate))
        # print(angle/pi*180, groupId)
        pointGroup[groupId].append((x,y,dist))
    for group in pointGroup:
        if len(group) == 0:
            continue
        else:
            groupArray = np.array(group)
            pointIds = np.where(groupArray[:,2]==np.max(groupArray[:,2]))
            for pointId in pointIds:
                # print(pointId[0])
                edgeP = groupArray[pointId[0]]
                edgePoints.append(edgeP)
    return pointGroup, edgePoints

def edgeDetect(A_basis, maxRotate):
    posArrays = []
    datumPoints = []
    for basis in A_basis:
        centerP = centerPointFind(basis)
        posArray = positionArray(centerP, basis)
        pointGroup, edgePoints = edgePointsFind(posArray, maxRotate)
        posArrays.append(posArray)
        datumPoints.append(edgePoints)
    return posArrays, datumPoints

def sparseDecode(denseArray, adoptDat = False):
    sparseMat = np.zeros(sizey)
    for group in denseArray:
        for point in group:
            [x, y, data] = point
            if not adoptDat:
                sparseMat[int(x)][int(y)] = 1
    return sparseMat

def closeCurveFit(datumPoints, splevNum):
    curves = []
    for group in datumPoints:
        pointsArray = np.array(group)
        # # order = np.argsort(pointsArray[:,0])
        # # pointsArray = pointsArray[order]
        
        # version 1
        x, y = pointsArray[:,0], pointsArray[:,1]
        x, y = np.r_[x, x[0]], np.r_[y, y[0]]
        tck, u = splprep([x, y], s=0, per=1)
        out = splev(np.linspace(u.min(),u.max(),splevNum), tck, der=0)
        
        # version 2
        # x, y = pointsArray[:,0], pointsArray[:,1]
        # dat = np.array([x, y]).T
        # dat = np.vstack([dat, dat[0]])
        # pad = 3
        # dat = np.pad(dat, [(pad, pad), (0,0)], mode='wrap')
        # i = np.arange(0, len(dat))
        # interp_i = np.linspace(0, i.max()-pad+1, 1000*(i.size-2*pad))
        # xi, yi = interp1d(i, dat[:,0], kind='cubic')(interp_i), interp1d(i, dat[:,1], kind='cubic')(interp_i)
        # out = [xi, yi]
        
        # version 3
        # x, y = pointsArray[:,0], pointsArray[:,1]
        # xy = np.hstack((x[:,np.newaxis], y[:,np.newaxis]))
        # hull = ConvexHull(xy)
        # xi, yi= x[hull.vertices], y[hull.vertices]
        # out = [xi, yi]
        
        curves.append(out)
    return curves

def closeCurvePlot(curves, path, colorMode='jet'):
    with plt.ioff():
        plt.figure()
        plt.axis('off')
        curveMat = np.zeros(sizey)
        for curve in curves:
            x_list, y_list = curve[0], curve[1]
            for iPoint in range(len(x_list)):
                x = int(x_list[iPoint])
                y = int(y_list[iPoint])
                if x >= sizey[0]:
                    x = sizey[0] - 1
                if y >= sizey[1]:
                    y = sizey[1] - 1
                curveMat[x][y] = 1
        plt.imshow(curveMat, cmap = colorMode)
        plt.savefig(path + 'aConfirm/closeCurve_maxRotate'+str(maxRotate)+'.png', bbox_inches = 'tight', pad_inches = 0)
        plt.close()
        
def isInCurve(anomalyGroups, datumPoints, x, y):
    isInsideCurve = False
    crossNum = [0, 0]
    for iGroup in range(len(anomalyGroups)):
        group = anomalyGroups[iGroup]
        centerP = centerPointFind(group)
        # centerP = [0, 0]
        # while(centerP[0] == x):
        #     centerP = [random.randint(0, A.shape[0]), random.randint(0, A.shape[1])]
        k1 = (y-centerP[1])/(x-centerP[0])
        b1 = y - k1*x
        
        # edgePoints = datumPoints[iGroup]
        # for iPoint in range(len(edgePoints)):
        #     edgeX1, edgeY1 = edgePoints[iPoint][0], edgePoints[iPoint][1]
        #     if iPoint < len(edgePoints)-1:
        #         edgeX2, edgeY2 = edgePoints[iPoint+1][0], edgePoints[iPoint+1][1]
        #     else:
        #         edgeX2, edgeY2 = edgePoints[0][0], edgePoints[0][1]
        #     k2 = (edgeY2-edgeY1)/(edgeX2-edgeX1)
        #     b2 = edgeY2 - k2*edgeX2
            
        #     crossX = (b2-b1)/(k1-k2)
        #     direction = np.sign(x-crossX)
        #     if min(edgeX1, edgeX2) <= crossX <= max(edgeX1, edgeX2):
        #         crossNum[int((direction+1)/2)] += 1
        
        edgePoints = datumPoints[iGroup]
        for iPoint in range(len(edgePoints[0])):
            edgeX1, edgeY1 = edgePoints[0][iPoint], edgePoints[1][iPoint]
            if iPoint < len(edgePoints[0])-1:
                edgeX2, edgeY2 = edgePoints[0][iPoint+1], edgePoints[1][iPoint+1]
            else:
                edgeX2, edgeY2 = edgePoints[0][0], edgePoints[1][0]
            if edgeX2 != edgeX1:
                k2 = (edgeY2-edgeY1)/(edgeX2-edgeX1)
                b2 = edgeY2 - k2*edgeX2
    
                crossX = (b2-b1)/(k1-k2)
                direction = np.sign(x-crossX)
                if min(edgeX1, edgeX2) <= crossX <= max(edgeX1, edgeX2):
                    crossNum[int((direction+1)/2)] += 1
            else:
                direction = np.sign(x-edgeX1)
                if min(edgeY1, edgeY2) <= k1*edgeX1+b1 <= max(edgeY1, edgeY2):
                    crossNum[int((direction+1)/2)] += 1
        
        if np.max(crossNum) % 2 == 1:
            isInsideCurve = True
            break            
    return isInsideCurve

def fillCurve(anomalyGroups, curves, datumPoints):
    anomalyMat = np.zeros(sizey)
    for curve in curves:
        x_list, y_list = curve[0], curve[1]
        for iPoint in range(len(x_list)):
            x = int(x_list[iPoint])
            y = int(y_list[iPoint])
            if x >= sizey[0]:
                x = sizey[0] - 1
            if y >= sizey[1]:
                y = sizey[1] - 1
            anomalyMat[x][y] = 1
    for x in range(sizey[0]):
        for y in range(sizey[1]):
            if isInCurve(anomalyGroups, datumPoints, x, y):
                anomalyMat[x][y] = 1
    return anomalyMat

def measure(anomalyMat, GT):
    TP = 0 # the number of defect pixels correctly detected
    FN = 0 # the number of defect pixels mis-detected
    FP = 0 # the number of non-defect pixels wrongly detected
    TN = 0 # the number of non-defect pixels correctly detected
    for x in range(sizey[0]):
        for y in range(sizey[1]):
            if anomalyMat[x][y] > 0: # detected as defects
                if GT[x][y] > 0:
                    TP += 1
                else:
                    FP += 1
            else: # detected as non-defects
                if GT[x][y] > 0:
                    FN += 1
                else:
                    TN += 1             
    TPR = TP / (TP+FN) # the ratio of successfully detection in detected defects
    FPR = FP / (FP+TN) # the ratio of wrongly detection in detected non-defects
    return TPR, FPR

def GetAnomalyResidual(Y0, anomalyMat, GT):
    rAnomalyBasis = []
    rAnomalyMat = np.zeros(sizey)
    basis = []
    for i in range(sizey[0]):
        for j in range(sizey[1]):
            if GT[i][j] > 0 and anomalyMat[i][j] == 0:
                basis.append([i,j,Y0[i][j]])
                rAnomalyMat[i][j] = 1
    rAnomalyBasis.append(basis)
    return rAnomalyBasis, rAnomalyMat

#%% main func
def main(imgId, patchId, npyInput, Y0):  
    # Y0 = np.load('./test_SD/test_Y0.npy')
    # sizey = Y0.shape
    path = './test_SD/'+imgId+'/'+patchId+'/'
    TPR, FPR = 0, 0
    
    A = np.load(path+'fAnomalies.npy')[0]
    anomalyGroups = np.load(path+'aConfirm/K15,sigma20,alpha3,maxDist10_basis.npy',allow_pickle=True)
    GT_img = cv2.imread('./splitImgs/'+imgId+'/'+patchId+'_mask.png')
    GT_img = cv2.cvtColor(GT_img, cv2.COLOR_BGR2GRAY)
    GT = np.array(GT_img)
    
    posArrays, datumPoints = edgeDetect(anomalyGroups, maxRotate)
    edgeMat = sparseDecode(datumPoints)
    array2heatmap(edgeMat, path+'aConfirm/', 'edgePoints_maxRotate'+str(maxRotate)+'.png', colorMode='gray')
    print("Edge points Found.")
    
    closeCurves = closeCurveFit(datumPoints,splevNum=1000)
    closeCurvePlot(closeCurves, path, colorMode='gray')
    print("Close curve Plotted.")
    np.save(path+'/aConfirm/closeCurves.npy', closeCurves)
    
    balanced_datumPoints = closeCurveFit(datumPoints,splevNum=100)
    anomalyMat = fillCurve(anomalyGroups, closeCurves, balanced_datumPoints)
    array2heatmap(anomalyMat, path+'aConfirm/', 'anomalyConfirmed_maxRotate'+str(maxRotate)+'.png', colorMode='gray')
    print('Anomaly area Filled.')
    np.save(path+'/aConfirm/anomaly_confirmed.npy', anomalyMat)
    
    TPR, FPR = measure(anomalyMat, GT)
    print('TPR: '+str(TPR)+', FPR: '+str(FPR))
    
    return (TPR, FPR)

sizey = (256,256)
maxRotate=36*1
indicators = []

# imgIds = ['wood_hole00'+str(i) for i in range(10)]
imgIds = ['wood_hole00'+str(i) for i in [0,3,9]]
# patchIds = [str([i,j]) for i in range(4) for j in range(4)]
# patchIds = [str([0,1]), str([1,1]), str([2,2]), str([2,3]), str([3,2])]
patchIds = [str([2,2])]
for imgId in imgIds:
    # if imgId not in ['wood_hole00'+str(i) for i in [0,3,4,5,6,9]]:
    if imgId in ['wood_hole00'+str(i) for i in [0]]:
        for patchId in patchIds:
            indicators.append(main(imgId, patchId, False, None))
