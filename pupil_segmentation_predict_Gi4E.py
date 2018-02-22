
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   LEARN FCN00
#

from __future__ import print_function
import argparse
import os, sys

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


# In[2]:


#
#  MAIN STARTS FROM HERE
#
if __name__ == '__main__':
    
    argv = sys.argv
    argc = len(argv)
    if argc < 3:
        print('usage: %s [TEST_SEQ] [EPOCH]'%(argv[0]))
        sys.exit(0)
    
    TEST_SEQ = argv[1]
    epoch = argv[2]
    
    target_size = (224, 224)
    dpath_this = './'
    dname_checkpoints = 'checkpoints_fcn00.Gi4E.' + TEST_SEQ
    dname_outputs = 'outputs.learn_by_Gi4E'
    fname_architecture = 'architecture.json'
    fname_weights = "model_weights_{epoch:02d}.h5"
    fname_stats = 'stats01.npz'
    dim_ordering = 'channels_first'
    fname_history = "history.pkl"

    # モデルを作成
    print('creating model fcn00...')
    model_fcn00 = create_fcn00(target_size)


# In[7]:


#
# Test Gi4e data
#
fnames = load_fnames('data.gi4e/list_test' + argv[1] + '.txt')
[fpaths_xs_test,fpaths_ys_test] = make_fnames(fnames,'data.gi4e/left/img','','')

X_test = load_imgs_asarray(fpaths_xs_test, grayscale=False, target_size=target_size,
                            dim_ordering=dim_ordering)

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


# In[4]:


from PIL import Image
import matplotlib.pyplot as plt

# 学習済みの重みをロード
fname_weights = 'model_weights_%02d.h5'%(int(epoch))
fpath_weights = os.path.join(dname_checkpoints, fname_weights)
model_fcn00.load_weights(fpath_weights)
print('==> done')

# テストを開始
outputs = model_fcn00.predict(X_test)


# In[8]:


# 出力を画像として保存
dname_outputs = './outputs.gi4e-left.learnedbygig4e/'
if not os.path.isdir(dname_outputs):
    print('create directory: %s'%(dname_outputs))
    os.mkdir(dname_outputs)

print('saving outputs as images...')
for i, array in enumerate(outputs):
    #array = np.where(array > 0.1, 1, 0) # 二値に変換
    #array = array.astype(np.float)
    formatted = (array[0]*255.0/np.max(array[0])).astype('uint8')
    #img_out = array_to_img(array, dim_ordering)
    img_out = Image.fromarray(formatted)
    fpath_out = os.path.join(dname_outputs, "%s"%(fnames[n]))
    img_out.save(fpath_out)

print('==> done')


# In[9]:


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
    plt.imshow(np.hstack((np.array(im1),np.array(im2))))
    plt.show()
    n = n + 1

print('%d: Dice eval av. : %f'%(epoch,np.mean(np.array(dice_eval))))

