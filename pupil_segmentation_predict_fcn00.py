
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   LEARN FCN00
#

from __future__ import print_function
import argparse
import os

import numpy as np
import pickle
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, AveragePooling2D
from keras.layers import merge
from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.image import list_pictures, array_to_img

from image_ext import list_pictures_in_multidir, load_imgs_asarray, img_dice_coeff, get_center
from fname_func import load_fnames, make_fnames

# MAXPOOLING
from create_fcn import create_fcn01, create_fcn00
# AVERAGE POOLING
#from create_fcn_avpool import create_fcn01,create_fcn00

np.random.seed(2016)


# In[ ]:


#
#  MAIN STARTS FROM HERE
#
if __name__ == '__main__':
    
    target_size = (224, 224)
    dpath_this = './'
    # dname_checkpoints = 'checkpoints_fcn00_avpool.augumented'
    dname_checkpoints = 'checkpoints_fcn00.augumented_2.alldata'
    # dname_checkpoints_fcn01 = 'checkpoints_fcn01_avpool'
    dname_checkpoints_fcn01 = 'checkpoints_fcn01'
    dname_outputs = 'outputs'
    fname_architecture = 'architecture.json'
    fname_weights = "model_weights_{epoch:02d}.h5"
    fname_stats = 'stats01.npz'
    dim_ordering = 'channels_first'
    fname_history = "history.pkl"

    # モデルを作成
    print('creating model fcn00...')
    model_fcn00 = create_fcn00(target_size)


# In[4]:


#
# Test UBIRIS Data
#
fnames = load_fnames('data/list_test_01.txt')
[fpaths_xs_test,fpaths_ys_test] = make_fnames(fnames,'data/img','data/mask','OperatorA_')

X_test = load_imgs_asarray(fpaths_xs_test, grayscale=False, target_size=target_size,
                            dim_ordering=dim_ordering)
Y_test = load_imgs_asarray(fpaths_ys_test, grayscale=True, target_size=target_size,
                            dim_ordering=dim_ordering)

# Yを初期化
center_test = []
for i in range(Y_test.shape[0]):
    center_test.append(get_center(Y_test[i,0,:,:]))
center_test = np.array(center_test)

# トレーニング時に計算した平均・標準偏差をロード    
print('loading mean and standard deviation from ' + fname_stats + '...')
stats = np.load(dname_checkpoints + '/' + fname_stats)
mean = stats['mean']
std = stats['std']
print('==> mean: ' + str(mean))
print('==> std : ' + str(std))

for i in range(3):
    X_test[:, i] = (X_test[:, i] - mean[i]) / std[i]
print('==> done')

from PIL import Image

# test epoch 50, 100, 150, 200
for epoch in range(50,201,50):
    # 学習済みの重みをロード
    fname_weights = 'model_weights_%02d.h5'%(epoch)
    fpath_weights = os.path.join(dname_checkpoints, fname_weights)
    model_fcn00.load_weights(fpath_weights)
    print('==> done loading data %d'%(epoch))
    
    # テストを開始
    outputs = model_fcn00.predict(X_test)
    
    # 出力を画像として保存
    dname_outputs = './outputs/'
    if not os.path.isdir(dname_outputs):
        print('create directory: %s'%(dname_outputs))
        os.mkdir(dname_outputs)
        
    print('saving outputs as images...')
    for i, array in enumerate(outputs):
        array = np.where(array > 0.1, 1, 0) # 二値に変換
        array = array.astype(np.float32)
        img_out = array_to_img(array, dim_ordering)
        # fpath_out = os.path.join(dname_outputs, fnames[i])
        fpath_out = os.path.join(dname_outputs, "%05d.png"%(i))
        img_out.save(fpath_out)

    print('==> done')
    
    dice_eval = []
    center_gt = []
    center_estimated = []
    
    for i in range(len(fpaths_xs_test)):
        # 出力結果
        im2 = Image.open(os.path.join(dname_outputs, "%05d.png"%(i)))
        im2 = im2.resize(target_size)
        # Grond Truth
        im3 = Image.open(fpaths_ys_test[i])
        im3 = im3.resize(target_size)

        im2a = np.array(im2)
        im2a[im2a > 0] = 1
        im3a = np.array(im3)
        im3a[im3a > 0] = 1

        overlap_a = np.array(im2a) * np.array(im3a)
        overlap_b = np.array(im2a) + np.array(im3a)
        dice_eval.append(2*sum(sum(overlap_a))/sum(sum(overlap_b)))

        c1 = get_center(im2a)
        c2 = get_center(im3a)
        
        if c1[0] >= 0 and c2[0] >= 0:
            center_gt.append(c2)
            center_estimated.append(c1)
    
    diff = np.array(center_gt) - np.array(center_estimated)

    print('%d: Dice eval av. : %f'%(epoch,np.mean(np.array(dice_eval))))
    print('Estimated %d / %d'%(len(center_gt),len(fpaths_xs_test)))
    print('Av. diff = %f, %f'%(np.sum(diff[:,0])/diff.shape[0],np.sum(diff[:,1])/diff.shape[0]))


# In[6]:


#
# Test NNLAB Data
#
fnames = load_fnames('data.nnlab/list_test_01.txt')
[fpaths_xs_test,fpaths_ys_test] = make_fnames(fnames,'data.nnlab/image','data.nnlab/gt','')

X_test = load_imgs_asarray(fpaths_xs_test, grayscale=False, target_size=target_size,
                            dim_ordering=dim_ordering)
Y_test = load_imgs_asarray(fpaths_ys_test, grayscale=True, target_size=target_size,
                            dim_ordering=dim_ordering)

# Yを初期化
center_test = []
for i in range(Y_test.shape[0]):
    center_test.append(get_center(Y_test[i,0,:,:]))
center_test = np.array(center_test)

# トレーニング時に計算した平均・標準偏差をロード    
print('loading mean and standard deviation from ' + fname_stats + '...')
stats = np.load(dname_checkpoints + '/' + fname_stats)
mean = stats['mean']
std = stats['std']
print('==> mean: ' + str(mean))
print('==> std : ' + str(std))

for i in range(3):
    X_test[:, i] = (X_test[:, i] - mean[i]) / std[i]
print('==> done')


# In[5]:


from PIL import Image

# test epoch 50, 100, 150, 200
for epoch in range(50,201,50):
    # 学習済みの重みをロード
    fname_weights = 'model_weights_%02d.h5'%(epoch)
    fpath_weights = os.path.join(dname_checkpoints, fname_weights)
    model_fcn00.load_weights(fpath_weights)
    print('==> done loading data %d'%(epoch))
    
    # テストを開始
    outputs = model_fcn00.predict(X_test)
    
    # 出力を画像として保存
    dname_outputs = './outputs/'
    if not os.path.isdir(dname_outputs):
        print('create directory: %s'%(dname_outputs))
        os.mkdir(dname_outputs)
        
    print('saving outputs as images...')
    for i, array in enumerate(outputs):
        array = np.where(array > 0.1, 1, 0) # 二値に変換
        array = array.astype(np.float32)
        img_out = array_to_img(array, dim_ordering)
        # fpath_out = os.path.join(dname_outputs, fnames[i])
        fpath_out = os.path.join(dname_outputs, "%05d.png"%(i))
        img_out.save(fpath_out)

    print('==> done')
    
    dice_eval = []
    center_gt = []
    center_estimated = []
    
    for i in range(len(fpaths_xs_test)):
        # 出力結果
        im2 = Image.open(os.path.join(dname_outputs, "%05d.png"%(i)))
        im2 = im2.resize(target_size)
        # Grond Truth
        im3 = Image.open(fpaths_ys_test[i])
        im3 = im3.resize(target_size)
        im3 = im3.convert('L')

        im2a = np.array(im2)
        im2a[im2a > 0] = 1
        im3a = np.array(im3)
        im3a[im3a > 0] = 1

        overlap_a = np.array(im2a) * np.array(im3a)
        overlap_b = np.array(im2a) + np.array(im3a)
        dice_eval.append(2*sum(sum(overlap_a))/sum(sum(overlap_b)))

        c1 = get_center(im2a)
        c2 = get_center(im3a)

        if c1[0] >= 0 and c2[0] >= 0:
            center_gt.append(c2)
            center_estimated.append(c1)
    
    diff = np.array(center_gt) - np.array(center_estimated)
    #print(diff)

    print('%d: Dice eval av. : %f'%(epoch,np.mean(np.array(dice_eval))))
    print('Av. diff = %f, %f'%(np.sum(diff[:,0])/diff.shape[0],np.sum(diff[:,1])/diff.shape[0]))
    print('number of estimated: %d / %d'%(diff.shape[0],len(fpaths_xs_test)))


# In[ ]:


from PIL import Image
import matplotlib.pyplot as plt

# 学習済みの重みをロード
epoch = 200
fname_weights = 'model_weights_%02d.h5'%(epoch)
fpath_weights = os.path.join(dname_checkpoints, fname_weights)
model_fcn00.load_weights(fpath_weights)
print('==> done')

# テストを開始
outputs = model_fcn00.predict(X_test)

# 出力を画像として保存
dname_outputs = './outputs/'
if not os.path.isdir(dname_outputs):
    print('create directory: %s'%(dname_outputs))
    os.mkdir(dname_outputs)

print('saving outputs as images...')
n = 0
for i, array in enumerate(outputs):
    array = np.where(array > 0.1, 1, 0) # 二値に変換
    array = array.astype(np.float32)
    img_out = array_to_img(array, dim_ordering)
    # fpath_out = os.path.join(dname_outputs, fnames[i])
    fpath_out = os.path.join(dname_outputs, "%05d.png"%(n))
    img_out.save(fpath_out)
    n = n + 1

print('==> done')


# In[10]:


n = 0
dice_eval = []

for i in range(len(fpaths_xs_test)):
    # テスト画像
    im1 = Image.open(fpaths_xs_test[i])
    im1 = im1.resize((320,240)) 
    # 出力結果
    im2 = Image.open(os.path.join(dname_outputs, "%05d.png"%(n)))
    im2 = im2.resize((320,240))
    # Grond Truth
    im3 = Image.open(fpaths_ys_test[i])
    im3 = im3.resize((320,240))
    im3 = im3.convert('L')
    
    im2_d = np.zeros((240,320,3), 'uint8')
    im2_d[:,:,0] = np.array(im2)
    im2_d[:,:,1] = np.array(im3)
    im2_d[:,:,2] = 0

    # Compute dice coeff
    im2a = np.array(im2)
    im2a[im2a > 0] = 1
    im3a = np.array(im3)
    im3a[im3a > 0] = 1

    overlap_a = np.array(im2a) * np.array(im3a)
    overlap_b = np.array(im2a) + np.array(im3a)
    dice_eval.append(2*sum(sum(overlap_a))/sum(sum(overlap_b)))

    print('%d: Dice eval : %f'%(n,2*sum(sum(overlap_a))/sum(sum(overlap_b))))  
    
    print(fpaths_xs_test[i])
    print(np.array(im1).shape)
    print(np.array(im2_d).shape)
    
    plt.imshow(np.hstack((np.array(im1)[:,:,:3],np.array(im2_d)[:,:,:3])))
#        plt.imshow(np.hstack((np.array(im1),np.array(im2))))
    plt.show()

    n = n + 1

print('%d: Dice eval av. : %f'%(epoch,np.mean(np.array(dice_eval))))

