# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:53:18 2023

@author: Song Ji
"""
from matplotlib import pyplot as plt
import os

class Plotter:
    def __init__(self):
        return
    
    def heatmapSave(self, img_array, save_folder_path, save_name, colorMode = 'jet'):
        if len(save_name.split('.')) == 1:
            save_name = save_name + '.png'
        else:
            try:
                if save_name.split('.')[1] not in ['png']:
                    save_name = save_name + '.png'
            except IndexError:
                print('Invalid image name given')
                save_name = 'default.png'
        with plt.ioff():
            plt.figure()
            plt.axis('off') # 关闭坐标轴显示
            plt.imshow(img_array, cmap=colorMode)
            if not os.path.exists(save_folder_path):
               os.mkdir(save_folder_path) 
            plt.savefig(os.path.join(save_folder_path, save_name), bbox_inches = 'tight', pad_inches = 0)
            plt.close()
            
    def heatmapPlot(self, img_array, colorMode = 'jet'):
        plt.figure()
        plt.axis('off') # 关闭坐标轴显示
        plt.imshow(img_array, cmap=colorMode)
        
    def graphSave(self, save_folder_path, save_name, label_x, label_y, title, x, y, quantile_line, line_style='--*b'):
        with plt.ioff():
            plt.figure()
            plt.xlabel(label_x)
            plt.ylabel(label_y)
            plt.title(title)
            plt.plot(x, y, line_style)
            if quantile_line != None:
                plt.axhline(quantile_line, color='r')
            plt.savefig(os.path.join(save_folder_path, save_name), bbox_inches = 'tight', pad_inches = 0)
            plt.close()
            

