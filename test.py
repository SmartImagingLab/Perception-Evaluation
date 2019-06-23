#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 10:40:39 2018

@author: Yi Huang
"""


from MeasureFunction import  Measure
from MeasureFunction import Load_img
import numpy as np
'''---------------------------Perceptual evaluation-------------------------------------'''
if __name__ == '__main__':
     measure = Measure()
     load_img = Load_img()
     ref_img = load_img.read_fits('./PeMeasureData/ref_x1.fits') 
     ref_img = load_img.norm(ref_img)
     observe_img = load_img.read_fits('./PeMeasureData/H_000000.fits')
     size=300
     temp = []
     for i in range(0,10):    
        for j in range(0,10): 
            patch = observe_img[i*50:i*50+size,j*50:j*50+size]
            patch = load_img.norm(patch)
            l = measure.Preception(ref_img,patch)
            temp.append(l)
     PE = np.mean(temp)
     print(PE)
            
