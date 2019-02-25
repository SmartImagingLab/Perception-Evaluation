import PIL.Image as Image
from astropy.io import fits
import torchvision.transforms as transforms
import numpy as np
import torch
import os

img_size = 4000

def norm(img):
    img = (img - np.min(img))/(np.max(img) - np.min(img)) #归一化
    img -= np.mean(img)  #去均值
    img /= np.std(img)  #标准化
    img = np.array(img,dtype='float32')
    return img

def load_imgfit_channel3(img_path):
    hdu = fits.open(img_path)
    image = hdu[0].data
    image = norm(image)
    image = Image.fromarray(image)
    image = image.convert('RGB')
    image = transforms.CenterCrop(img_size)(image)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    return image
 

def load_imgfit_channel1(img_path):
    hdu = fits.open(img_path)
    image = hdu[0].data
    image = norm(image)
    image = Image.fromarray(image)
    image = transforms.RandomCrop(img_size)(image)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    return image

def load_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img

def load_img_1(img_path):
    img = Image.open(img_path)
    img = img.resize((img_size, img_size))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img


def show_img(img):
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    img = transforms.Grayscale()(img)
    img.show()
    
def save_fit(img,name,path):
    if torch.cuda.is_available(): 
        img = torch.Tensor.cpu(img)
        img = img.data.numpy()
    else:
        img = np.array(img)
    if os.path.exists(path + name+'.fits'):
        os.remove(path + name+'.fits')
    grey=fits.PrimaryHDU(img)
    greyHDU=fits.HDUList([grey])
    greyHDU.writeto(path + name+'.fits')
    
def save_fit_cpu(img,name,path):
    if os.path.exists(path + name+'.fits'):
        os.remove(path + name+'.fits')
    grey=fits.PrimaryHDU(img)
    greyHDU=fits.HDUList([grey])
    greyHDU.writeto(path + name+'.fits')
    

def read_fits(path):
    hdu = fits.open(path)
    img = hdu[0].data
    img =norm(img) 
    img = np.array(img,dtype = np.float32)
    hdu.close()
    return img



class Load_img:
    def __init__(self,path,img,name):
        self.path = path
        self.img = img
        self.name = name
        
    def norm(img):
        img = (img - np.min(img))/(np.max(img) - np.min(img)) #normalization
        img -= np.mean(img)  # take the mean
        img /= np.std(img)  #standardization
        img = np.array(img,dtype='float32')
        return img
    
    def norm_0_1(img):
        img = (img - np.min(img))/(np.max(img) - np.min(img)) #normalization
        return img
        
    
    def read_fits(path):
        hdu = fits.open(path)
        img = hdu[0].data
        img = np.array(img,dtype = np.float32)
        hdu.close()
        return img
    
    def save_fit_cpu(img,name,path):
        if os.path.exists(path + name+'.fits'):
            os.remove(path + name+'.fits')
        grey=fits.PrimaryHDU(img)
        greyHDU=fits.HDUList([grey])
        greyHDU.writeto(path + name+'.fits')
        
    def save_fit(img,name,path):
        if torch.cuda.is_available(): 
            img = torch.Tensor.cpu(img)
            img = img.data.numpy()
        else:
            img = np.array(img)
        if os.path.exists(path + name+'.fits'):
            os.remove(path + name+'.fits')
        grey=fits.PrimaryHDU(img)
        greyHDU=fits.HDUList([grey])
        greyHDU.writeto(path + name+'.fits')



    




    
