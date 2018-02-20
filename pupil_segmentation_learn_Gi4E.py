
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

from image_ext import list_pictures_in_multidir, load_imgs_asarray, img_dice_coeff
from fname_func import load_fnames, make_fnames

# MAXPOOLING
from create_fcn import create_fcn00

np.random.seed(2016)


# In[2]:


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2.*intersection + 1) / (K.sum(y_true) + K.sum(y_pred) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# In[10]:


#
#  MAIN STARTS FROM HERE (python xxx [test_seq])
#
if __name__ == '__main__':
    
    argv = sys.argv
    argc = len(argv)
    if argc < 2:
        print('usage: %s [TEST_SEQ]'%(argv[0]))
        sys.exit(0)
    
    TEST_SEQ = argv[1]
    
    target_size = (224, 224)
    dpath_this = './'
    # dname_checkpoints = 'checkpoints_fcn00_avpool.augumented'
    dname_checkpoints = 'checkpoints_fcn00.Gi4E.' + TEST_SEQ
    dname_checkpoints_fcn00 = 'checkpoints_fcn00.augumented_2'    
    fname_architecture = 'architecture.json'
    fname_weights = "model_weights_{epoch:02d}.h5"
    fname_stats = 'stats01.npz'
    dim_ordering = 'channels_first'
    fname_history = 'history.pkl'

    # モデルを作成
    print('creating model fcn00...')
    model_fcn00 = create_fcn00(target_size)
    
    # 学習済みの重みをロード(UBIRIS + Augumented2)
    print('loading weight..')
    epoch = 200
    fname_weights = 'model_weights_%02d.h5'%(epoch)
    fpath_weights = os.path.join(dname_checkpoints_fcn00, fname_weights)
    model_fcn00.load_weights(fpath_weights)
    print('==> done')
    
    if os.path.exists(dname_checkpoints) == 0:
        os.mkdir(dname_checkpoints)
        print('making directory:' + dname_checkpoints)


# In[11]:


#
#   LEARNING MODE
#
# Read Learning Data
fnames = load_fnames('data.gi4e/list_train' + TEST_SEQ + '.txt')
[fpaths_xs_train,fpaths_ys_train] = make_fnames(fnames,'data.gi4e/left/img','data.gi4e/left/mask','')
[fpaths_xs_train_R,fpaths_ys_train_R] = make_fnames(fnames,'data.gi4e/right/img','data.gi4e/right/mask','')
fpaths_xs_train.extend(fpaths_xs_train_R)
fpaths_ys_train.extend(fpaths_ys_train_R)

print('reading training data')
X_train = load_imgs_asarray(fpaths_xs_train, grayscale=False, target_size=target_size,
                            dim_ordering=dim_ordering)
print('reading training gt data')
Y_train = load_imgs_asarray(fpaths_ys_train, grayscale=True, target_size=target_size,
                            dim_ordering=dim_ordering) 

# Read Validation Data
#    fnames = load_fnames('data/list_valid_01.txt')
#    [fpaths_xs_valid,fpaths_ys_valid] = make_fnames(fnames,'data/img','data/mask','OperatorA_')
VALIDATION = 0

if VALIDATION == 1:
    fnames = load_fnames('data.gi4e/list_valid.txt')
    [fpaths_xs_valid,fpaths_ys_valid] = make_fnames(fnames,'data.gi4e/left/img','data.gi4e/left/mask','')
    [fpaths_xs_valid_R,fpaths_ys_valid_R] = make_fnames(fnames,'data.gi4e/right/img','data.gi4e/right/mask','')
    fpaths_xs_valid.extend(fpaths_xs_valid_R)
    fpaths_ys_valid.extend(fpaths_ys_valid_R)

    print('reading validation data')
    X_valid = load_imgs_asarray(fpaths_xs_valid, grayscale=False, target_size=target_size,
                                dim_ordering=dim_ordering)
    Y_valid = load_imgs_asarray(fpaths_ys_valid, grayscale=True, target_size=target_size,
                                dim_ordering=dim_ordering)     
else:
    X_valid = []
    Y_valid = []

print('==> ' + str(len(X_train)) + ' training images loaded')
print('==> ' + str(len(Y_train)) + ' training masks loaded')
print('==> ' + str(len(X_valid)) + ' validation images loaded')
print('==> ' + str(len(Y_valid)) + ' validation masks loaded')

# 前処理
print('computing mean and standard deviation...')
mean = np.mean(X_train, axis=(0, 2, 3))
std = np.std(X_train, axis=(0, 2, 3))
print('==> mean: ' + str(mean))
print('==> std : ' + str(std))


# In[12]:


print('saving mean and standard deviation to ' + fname_stats + '...')
stats = {'mean': mean, 'std': std}
np.savez(dname_checkpoints + '/' + fname_stats, **stats)
print('==> done')

print('globally normalizing data...')
for i in range(3):
    X_train[:, i] = (X_train[:, i] - mean[i]) / std[i]
    if VALIDATION == 1:
        X_valid[:, i] = (X_valid[:, i] - mean[i]) / std[i]

Y_train /= 255

if VALIDATION == 1:
    Y_valid /= 255
print('==> done')


# In[13]:


# 損失関数，最適化手法を定義
adam = Adam(lr=1e-5)
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.95, nesterov=True)
#rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model_fcn00.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])

# 構造・重みを保存するディレクトリーの有無を確認
dpath_checkpoints = os.path.join(dpath_this, dname_checkpoints)
if not os.path.isdir(dpath_checkpoints):
    os.mkdir(dpath_checkpoints)

# 重みを保存するためのオブジェクトを用意
fname_weights = "model_weights_{epoch:02d}.h5"
fpath_weights = os.path.join(dpath_checkpoints, fname_weights)
checkpointer = ModelCheckpoint(filepath=fpath_weights, save_best_only=True)      


# In[14]:


# トレーニングを開始
print('start training...')
epochs = 100
if VALIDATION == 1:
    history = model_fcn00.fit(X_train[:,:,:,:], Y_train[:,:,:,:], batch_size=32, epochs=epochs, verbose=1,
                  shuffle=True, validation_data=(X_valid, Y_valid), callbacks=[checkpointer])
else:
    history = model_fcn00.fit(X_train[:,:,:,:], Y_train[:,:,:,:], batch_size=32, epochs=epochs, verbose=1,
                  shuffle=True, validation_data=(X_train, Y_train), validation_split=0.1, 
                callbacks=[checkpointer])        


# In[8]:


# Save History
f = open(dname_checkpoints + '/' + fname_history,'wb')
pickle.dump(history.history,f)
f.close


# In[10]:


#
#   Show History
#
SHOW_HISTORY = 0

if SHOW_HISTORY == 1:
    import matplotlib.pyplot as plt

    # load pickle
    print(dname_checkpoints + '/' + fname_history)
    history = pickle.load(open(dname_checkpoints + '/' + fname_history, 'rb'))

    for k in history.keys():
        plt.plot(history[k])
        plt.title(k)
        plt.show()

