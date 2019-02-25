#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 10:40:39 2018

@author: yellow
"""

import net
import numpy as np
import torch
import load_img
from torch.autograd import Variable

'''---------------------------Perceptual evaluation-------------------------------------'''

def pre_data(img):
    image = load_img.norm(img)
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    image = image.unsqueeze(0)
    image = torch.cat((image,image,image),1)
    return image
# Gram Loss  
def gram_matrix(y):
	"""
	for each channel  in C of the feature map:
		element-wise multiply and sum together 
	return a C*C matrix
	"""
	(b, ch, h, w) = y.size()
	features = y.view(b, ch, w*h)
	features_t = features.transpose(1,2)
	gram = features.bmm(features_t) /(ch*h*w)
	return gram

def LOSS(or_img,de_img):
    or_img3 = pre_data(or_img)
    de_img3 = pre_data(de_img)
    vgg_model = net.Vgg16() # fill pretrained vgg
    if torch.cuda.is_available():
        print('=> Use CUDA')
        vgg_model.cuda()
    if torch.cuda.is_available():
        or_img3 = Variable(or_img3).cuda()
        de_img3 = Variable(de_img3).cuda()
    feature_or = vgg_model(or_img3)
    feature_de = vgg_model(de_img3)
    if torch.cuda.is_available():
        or_img3 = torch.Tensor.cpu(or_img3)
        de_img3 = torch.Tensor.cpu(de_img3)
    L = 0
    x1 = gram_matrix(feature_or[3].detach())
    x1 = x1.reshape(1,-1)[0]
    x2 = gram_matrix(feature_de[3].detach())
    x2 = x2.reshape(1,-1)[0]
    L = torch.dot(x1,x2)/(torch.norm(x1)*torch.norm(x2))
    print(L)
    L = np.float32(L)
    return L

if __name__ == '__main__':
     img1 = load_img.read_fits('.fits') #input single channel image
     img2 = load_img.read_fits('.fits') #input single channel image
     l = LOSS(img1,img2)
     
     
    










    

            
