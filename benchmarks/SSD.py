# -*- coding: utf-8 -*-
"""
Revised according to Hao Yan, August 2015
Coded by Song Ji
"""

import numpy as np
from numpy import eye, ones, zeros, linspace, mat
from numpy import concatenate, expand_dims, transpose, sqrt, diff, diag, r_, divide, dot
from scipy.interpolate import splev
from numpy.linalg import svd, solve, inv, norm
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import os
import time

t1 = time.time()

#%% 读入数据
imgId = "SSD"
patchId = "[10, 20, 50, 100]"
patchId = "[15, 53, 32, 16, 43]"
patchId = "crossing [38, 28, 51, 24, 34]"
# img = cv2.imread('./splitImgs/'+imgId+'/'+patchId+'_test.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Y = np.array(img)
Y = np.load('../simulation/imgs/'+patchId+'/BTAN.npy')
nx, ny = Y.shape[0], Y.shape[1]

# GT_img = cv2.imread('./splitImgs/'+imgId+'/'+patchId+'_mask.png')
# GT_img = cv2.cvtColor(GT_img, cv2.COLOR_BGR2GRAY)
# GT = np.array(GT_img)
GT = np.load('../simulation/anomaly.npy')

#%% smooth basis construction
def bsplineBasis(n, k, sd):
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
        B = eye(n) # generate an identity matrix with size of n
    else:
        knots = r_[ones(sd), linspace(1, n, k), (n)*ones(sd)] # to generate the dot matrix for input of B-spline
        x = linspace(1, n, n) # the control points, ranging from 1 to n
        nKnots = len(knots) - sd - 1
        v = zeros((nKnots, n))
        d = eye(nKnots, n) # the cofficients matrix for the following B-spline func generation
        for i in range(nKnots):
            v[i] = splev(x, (knots, d[i], sd))
        B = v.T
    return B

def bsplineSmoothDecompauto(y,B,Ba,lambdaxy,gamma,maxIter=2,errtol=1e-6):
    
    softthreshold = lambda residual,gamma : np.sign(residual)*np.maximum(np.abs(residual) - gamma, 0)
    
    Lbs = 2*norm(Ba[0])**2 * norm(Ba[1])**2
    X = zeros((Ba[0].shape[1], Ba[1].shape[1]))
    BetaA = zeros((Ba[0].shape[1], Ba[1].shape[1]))
    
    SChange = 1e10
    H, C, Z = [[], []], [[], []], [[], []] # list containing 2 elements
    a = zeros((nx, ny))
    
    for idim in range(2):
        Li = np.sqrt(B[idim].T @ B[idim])
        Li = Li + 1e-8 * eye(Li.shape[0])
        Di = diff(eye(B[idim].shape[1]), 0)
        tmp = solve(Li.T, (Di.T @ Di)) @ inv(Li)
        Ui, C[idim], vt = svd(tmp)
        Z[idim] = B[idim] @ inv(Li.T) @ Ui
    
    iIter = 0
    t = 1
    while SChange > errtol and iIter < maxIter:
        iIter = iIter + 1
        Sold = a
        BetaSold = BetaA
        told = t
        
        for idim in range(2):
            H[idim] = B[idim] @ solve(B[idim].T@B[idim] + lambdaxy[idim]*(Di[idim].T@Di[idim]), B[idim].T)
        
        yhat = H[0] @ (y - a) @ H[1]
        BetaSe = X + 2/Lbs * Ba[0].T @ (y - Ba[0]@X@Ba[1].T) @ Ba[1]
        
        # maxYe = np.max(abs(BetaSe))
        BetaA = softthreshold(BetaSe, gamma/Lbs)
        a = Ba[0] @ BetaA @ Ba[1]
        
        t = (1 + sqrt(1+4*told**2))/2
        if iIter == 1:
            X = BetaA
        else:
            X = BetaA + (told - 1)/t*(BetaA-BetaSold)
        
        SChange = 0
        for i in range(nx):
            for j in range(ny):
                SChange += (a[i][j] - Sold[i][j])**2
    
    # print(iIter, SChange)
    a = y - yhat
        
    return yhat, a

#%% measure
def imgBinary(img, boundPer):
    biImg = np.zeros((nx,ny))
    lb, ub = np.percentile(img,(boundPer[0])), np.percentile(img,(boundPer[1]))
    for i in range(nx):
        for j in range(ny):
            if img[i][j] >= lb:
                biImg[i][j] = 0
            else:
                biImg[i][j] = 1
    return biImg

def measure(anomalyMat, GT):
    TP = 0 # the number of defect pixels correctly detected
    FN = 0 # the number of defect pixels mis-detected
    FP = 0 # the number of non-defect pixels wrongly detected
    TN = 0 # the number of non-defect pixels correctly detected
    for x in range(nx):
        for y in range(ny):
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

def array2heatmap(array, folderName1, folderName2, imgname, colorMode = 'jet'):
    # plt.figure(figsize=(array.shape[0], array.shape[1]))
    plt.figure()
    plt.axis('off') # 关闭坐标轴显示
    # plt.imshow(array, cmap='gray') 
    plt.imshow(array, cmap=colorMode) 
    path = '../simulation/benchmarks/'+folderName1+'/'
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + folderName2 +'/'
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(path + imgname + '.png', bbox_inches = 'tight', pad_inches = 0)

#%% main func
def main(Y, kx, ky, sd, snk, lambdaxy, gamma):
    yhat = zeros((nx, ny))
    A = zeros((nx, ny))
    
    B, Bs = [], []
    
    B.append(bsplineBasis(nx,kx,3)) #用三次样条函数插值
    B.append(bsplineBasis(ny,ky,3))

    skx, sky = round(nx/snk), round(ny/snk)
    Bs.append(bsplineBasis(nx,skx,2))
    Bs.append(bsplineBasis(ny,sky,2))

    [yhat,A] = bsplineSmoothDecompauto(Y,B,Bs,lambdaxy,gamma)
    
    return yhat, A

#%% params setting
kx, ky = 2, 2
sd = 3
snk = 1
lambdaxy = [0.1, 0.1]
gamma = 2e-1

#%%

yhat, A = main(Y, kx, ky, sd, snk, lambdaxy, gamma)

biImg = imgBinary(A, boundPer=[25, 100])

array2heatmap(yhat, imgId, patchId, 'Bg_estimation')
array2heatmap(A, imgId, patchId, 'Anomaly_estimation')
array2heatmap(Y, imgId, patchId, 'origin_img')
array2heatmap(biImg, imgId, patchId, 'anomaly_binary', 'gray')

TPR, FPR = 0, 0
TPR, FPR = measure(biImg, GT)
print('TPR: '+str(TPR)+', FPR: '+str(FPR))

t2 = time.time()
print(t2-t1)