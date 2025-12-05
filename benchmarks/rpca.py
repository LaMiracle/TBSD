# -*- coding: utf-8 -*-"
"""
Revised according to Hao Yan, October 2014
Coded by Song Ji
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import os
import time

t1 = time.time()

#%% 读入数据
# imgId = "wood_hole009"
# patchId = "[2, 1]"
# img = cv2.imread('./splitImgs/'+imgId+'/'+patchId+'_test.png')
# # 拷贝源数据矩阵
# # Y = np.zeros((Y0.shape[0], Y0.shape[1]))
# # Y = np.copy(Y0)
# # 转化成灰度图
# gray_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
# # gray_img.save(r'./RES_gray.png')
# Y = np.array(gray_img)
# GT_img = cv2.imread('./splitImgs/'+imgId+'/'+patchId+'_mask.png')
# GT_img = cv2.cvtColor(GT_img, cv2.COLOR_BGR2GRAY)
# GT = np.array(GT_img)

imgId = "RPCA"
patchId = "[10, 20, 50, 100]"
patchId = "[15, 53, 32, 16, 43]"
patchId = "crossing [38, 28, 51, 24, 34]"
Y = np.load('../simulation/imgs/'+patchId+'/BTAN.npy')
GT = np.load('../simulation/anomaly.npy')

rownum, colnum = Y.shape[0], Y.shape[1]

#%% 光滑度使用相邻像素灰度值差的极值作为阈值，如果是rgb图像考虑曼哈顿距离
# rowdiv = np.zeros((Y.shape[0], Y.shape[1]))
# coldiv = np.zeros((Y.shape[0], Y.shape[1]))
# for i in range(Y.shape[0]-1):
#     for j in range(Y.shape[1]):
#         rowdiv[i,j] = int(Y[i,j]) - int(Y[i+1, j])
# for i in range(Y.shape[0]):
#     for j in range(Y.shape[1]-1):
#         coldiv[i,j] = int(Y[i,j]) - int(Y[i, j+1])
# np.max(rowdiv)
# np.max(coldiv)

#%% 变量初始化
A_hat = np.zeros((rownum, colnum))
E_hat = np.zeros((rownum, colnum))

#%% 迭代
def pca_iter(tol, d_norm, tt_svd, lam, rho, mu, mu_bar, maxiter, A_hat, E_hat, Y):
    S_tmp = []
    for iteration in range(maxiter):
        # 矩阵分解
        T = Y - A_hat + (1/mu)*Y # 将包含异常和噪声的排除阵、与被mu稀释的原矩阵加合
        E_hat = np.clip(T - lam/mu, 0, None) # 这里是想用类似减去噪声均值的方法把异常大致筛选出来？
        # mu在膨胀而lamda不变，T减值在不断减小，
        # 将矩阵内所有负数修改为0
        [U, S, V] = np.linalg.svd(Y - E_hat + (1/mu)*Y, full_matrices=False)
        diagS = S.flatten()
        svp = len([x for x in diagS if x > 1/mu]) # 找出大于阈值的特征值长度，完成主成分提取
        # 迭代次数越靠后，大于1的mu值在rho作用下逐渐膨胀，纳入更多主成分，但由于Y使用噪声阵Z做了修正，所以提取的特征值逐渐趋向均一化且逐渐减小
        # for i in range(1, len(diagS)):
        #     svp = i
        #     if sum(diagS[0:i]) > 0.9*sum(diagS):
        #         break
        # svp = 280
        A_hat = np.dot(np.dot(np.matrix(U[:, 0:svp]), np.transpose(np.matrix(np.diag(diagS[0:svp] - 1/mu)))), np.matrix(V[0:svp, :]))
        # 最小二乘求解
        tt_svd = tt_svd + 1
        Z = Y - A_hat - E_hat  # 包含噪声的error
        Y = Y + mu*Z
        mu = min(mu*rho, mu_bar)
        # 检查中止条件
        S_tmp.append(T)
        # S_tmp.append(diagS)
        stopCriterion = np.linalg.norm(Z, 'fro') / d_norm # 求Z的二范数并与截止条件对比，含义是像素水平上的平均拟合误差到一定程度就停止迭代
        
        if stopCriterion < tol:
            print("Iter", iteration+1, "times\n")
            break
        # 打印迭代进度
        if (iteration+1) % 10 == 0:
            print("Iter", iteration+1, "times\n")

        # break
    return [A_hat, E_hat, S_tmp]

#%% 参数初始化
tol = 1e-7
d_norm = np.linalg.norm(Y, 'fro')
tt_svd = 0
# lam = 1.85e-1
# rho = 1.1 # 迭代过程中mu的膨胀系数
# norm_two = np.linalg.norm(Y)
# mu = 1e5/norm_two  # 控制迭代过程中error向Y进行成分累加的规模乘子,1/mu是控制主成分提取的阈值参数,mu越大提取的成分越多
# mu_bar = mu*1e7 # 防止mu过度膨胀所设置的阈值
# maxiter = 1
'''调参调mu就行了'''
lam = 1e-2
rho = 1.5 # 迭代过程中mu的膨胀系数
norm_two = np.linalg.norm(Y)
mu = 1e5
mu = mu/norm_two  # 控制迭代过程中error向Y进行成分累加的规模乘子,1/mu是控制主成分提取的阈值参数,mu越大提取的成分越多
mu_bar = mu*1e7 # 防止mu过度膨胀所设置的阈值
maxiter = 2

# norm_inf = np.max(np.sum(abs(Y), axis=1)) / lam # 求矩阵的无穷范数
# Y = Y / norm_inf

[L, S, EV] = pca_iter(tol, d_norm, tt_svd, lam, rho, mu, mu_bar, maxiter, A_hat, E_hat, Y)

#%% 生成图片
def imgBinary(img, boundPer):
    biImg = np.zeros((rownum,colnum))
    lb, ub = np.percentile(img,(boundPer[0])), np.percentile(img,(boundPer[1]))
    for i in range(rownum):
        for j in range(colnum):
            if img[i][j] <= lb:
                biImg[i][j] = 0
            else:
                biImg[i][j] = 1
    return biImg

biImg = imgBinary(np.array(S), boundPer=[75, 100])

def measure(anomalyMat, GT):
    TP = 0 # the number of defect pixels correctly detected
    FN = 0 # the number of defect pixels mis-detected
    FP = 0 # the number of non-defect pixels wrongly detected
    TN = 0 # the number of non-defect pixels correctly detected
    for x in range(rownum):
        for y in range(colnum):
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
    # path = './RPCA/'+folderName1+'/'
    path = '../simulation/benchmarks/'+folderName1+'/'
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + folderName2 +'/'
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(path + imgname + '.png', bbox_inches = 'tight', pad_inches = 0)
    
array2heatmap(L, imgId, patchId, 'RPCA_LowRankFigure', 'gray')
array2heatmap(S, imgId, patchId, 'RPCA_SparseAnomaly', 'gray')
array2heatmap(Y, imgId, patchId, 'origin_img')
array2heatmap(biImg, imgId, patchId,'anomaly_binary', colorMode='gray')

TPR, FPR = 0, 0
TPR, FPR = measure(biImg, GT)
print('TPR: '+str(TPR)+', FPR: '+str(FPR))

t2 = time.time()
print(t2-t1)