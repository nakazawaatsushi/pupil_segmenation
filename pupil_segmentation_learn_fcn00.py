
# coding: utf-8

# In[1]:


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

from image_ext import list_pictures_in_multidir, load_imgs_asarray, img_dice_coeff
from fname_func import load_fnames, make_fnames

# MAXPOOLING
from create_fcn import create_fcn01, create_fcn00
# AVERAGE POOLING
#from create_fcn_avpool import create_fcn01,create_fcn00

np.random.seed(2016)


# In[2]:


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2.*intersection + 1) / (K.sum(y_true) + K.sum(y_pred) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# In[3]:


#
#  MAIN STARTS FROM HERE
#
if __name__ == '__main__':
    
    target_size = (224, 224)
    dpath_this = './'
    dname_checkpoints = 'checkpoints_fcn00.augumented_2.alldata'
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
    
    if os.path.exists(dname_checkpoints) == 0:
        os.mkdir(dname_checkpoints)


# In[4]:


#
#   LEARNING MODE
#
# Read Learning Data
#    fnames = load_fnames('data/list_train_01.txt')
#    [fpaths_xs_train,fpaths_ys_train] = make_fnames(fnames,'data/img','data/mask','OperatorA_')
#    fnames = load_fnames('data.nnlab/list_train_01.txt')
#    fnames = load_fnames('data/list_train_01.txt')
fnames = load_fnames('data_augumented_2/list_all.txt')
#    [fpaths_xs_train,fpaths_ys_train] = make_fnames(fnames,'data.nnlab/image','data.nnlab/gt','')
[fpaths_xs_train,fpaths_ys_train] = make_fnames(fnames,'data_augumented_2/img','data_augumented_2/mask','')

print('reading training data')
X_train = load_imgs_asarray(fpaths_xs_train, grayscale=False, target_size=target_size,
                            dim_ordering=dim_ordering)
print('reading traking gt data')
Y_train = load_imgs_asarray(fpaths_ys_train, grayscale=True, target_size=target_size,
                            dim_ordering=dim_ordering) 


# In[6]:


# Read Validation Data
#    fnames = load_fnames('data/list_valid_01.txt')
#    [fpaths_xs_valid,fpaths_ys_valid] = make_fnames(fnames,'data/img','data/mask','OperatorA_')
#fnames = load_fnames('data_augumented_2/list_valid_01.txt')
# [fpaths_xs_valid,fpaths_ys_valid] = make_fnames(fnames,'data_augumented_2/img','data_augumented_2/mask','')
VALIDATION = 0

if VALIDATION == 1:
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


# In[7]:


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
    Y_train_valid /= 255
print('==> done')


# In[8]:


init_from_fcn01 = 1

if init_from_fcn01 == 1:
    # モデルに学習済のfcn01 Weightをロードする
    model_fcn01 = create_fcn01(target_size)        
    epoch = 100
    fname_weights = 'model_weights_%02d.h5'%(epoch)
    fpath_weights_fcn01 = os.path.join(dname_checkpoints_fcn01, fname_weights)
    model_fcn01.load_weights(fpath_weights_fcn01)
    #print('==> done')

    # load weights from Learned U-NET
    layer_names = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2',
                   'conv4_1','conv4_2','conv5_1', 'conv5_2',
                'up1_1', 'up1_2', 'up2_1', 'up2_2', 'up3_1', 'up3_2', 'up4_1', 
                   'up4_2', 'conv_fin']
    layer_names = ['conv1_1','conv1_2','conv2_1','conv2_2',
                'up1_1', 'up1_2', 'up2_1', 'up2_2', 'conv_fin']

    print('copying layer weights')
    for name in layer_names:
        print(name)
        model_fcn00.get_layer(name).set_weights(model_fcn01.get_layer(name).get_weights())
        model_fcn00.get_layer(name).trainable = True


# In[9]:


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


# In[12]:


# トレーニングを開始
epochs = 200

if VALIDATION == 1:
    print('start training...')
    history = model_fcn00.fit(X_train[:,:,:,:], Y_train[:,:,:,:], batch_size=32, epochs=epochs, verbose=1,
                  shuffle=True, validation_data=(X_valid, Y_valid), callbacks=[checkpointer])
else:
    print('start training...')
    history = model_fcn00.fit(X_train[:,:,:,:], Y_train[:,:,:,:], batch_size=32, epochs=epochs, verbose=1,
                shuffle=True, validation_split=0.1, callbacks=[checkpointer])


# In[9]:


# Save History
f = open(dname_checkpoints + '/' + fname_history,'wb')
pickle.dump(history.history,f)
f.close


# In[10]:


#
#   Show History
#

# load pickle
print(dname_checkpoints + '/' + fname_history)
history = pickle.load(open(dname_checkpoints + '/' + fname_history, 'rb'))

for k in history.keys():
    plt.plot(history[k])
    plt.title(k)
    plt.show()

