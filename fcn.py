#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os

import numpy as np
np.random.seed(2016)
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import merge
from keras.optimizers import Adam
from keras.preprocessing.image import list_pictures, array_to_img

from image_ext import list_pictures_in_multidir, load_imgs_asarray


def create_fcn(input_size):
    inputs = Input((3, input_size[1], input_size[0]))

    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool5)
    conv6 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv5], mode='concat', concat_axis=1)
    conv7 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv4], mode='concat', concat_axis=1)
    conv8 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv3], mode='concat', concat_axis=1)
    conv9 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv9)

    up10 = merge([UpSampling2D(size=(2, 2))(conv9), conv2], mode='concat', concat_axis=1)
    conv10 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up10)
    conv10 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv10)

    up11 = merge([UpSampling2D(size=(2, 2))(conv10), conv1], mode='concat', concat_axis=1)
    conv11 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up11)
    conv11 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv11)

    conv12 = Convolution2D(1, 1, 1, activation='sigmoid')(conv11)

    fcn = Model(input=inputs, output=conv12)

    return fcn

def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2.*intersection + 1) / (K.sum(y_true) + K.sum(y_pred) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

if __name__ == '__main__':
    # コマンドライン引数を解析
    parser = argparse.ArgumentParser('Train/Test FCN with Keras.')
    parser.add_argument('mode', choices=['train', 'test'], help='run mode',
                        metavar='MODE')
    parser.add_argument('--weights', default='', help='path to a weights file')
    args = parser.parse_args()

    # オプション
    target_size = (224, 224)
    dname_checkpoints = 'checkpoints'
    dname_outputs = 'outputs'
    fname_architecture = 'architecture.json'
    fname_weights = "model_weights_{epoch:02d}.h5"
    fname_stats = 'stats.npz'
    dim_ordering = 'th'

    # データディレクトリーのパスを取得
    fpath_this = os.path.realpath(__file__)
    dpath_this = os.path.dirname(fpath_this)
    dpath_data = os.path.join(dpath_this, 'data')

    if args.mode == 'train': # トレーニング
        # データを配列として取得
        print('loading data...')
        dpaths_xs_train = [os.path.join(dpath_data, 'train'),
                           os.path.join(dpath_data, 'train-aug')]
        dpaths_ys_train = [os.path.join(dpath_data, 'train_mask'),
                           os.path.join(dpath_data, 'train_mask-aug')]
        dpaths_xs_valid = [os.path.join(dpath_data, 'valid'),
                           os.path.join(dpath_data, 'valid-aug')]
        dpaths_ys_valid = [os.path.join(dpath_data, 'valid_mask'),
                           os.path.join(dpath_data, 'valid_mask-aug')]

        fpaths_xs_train = list_pictures_in_multidir(dpaths_xs_train)
        fpaths_ys_train = list_pictures_in_multidir(dpaths_ys_train)
        fpaths_xs_valid = list_pictures_in_multidir(dpaths_xs_valid)
        fpaths_ys_valid = list_pictures_in_multidir(dpaths_ys_valid)

        fpaths_xs_train = sorted(fpaths_xs_train)
        fpaths_ys_train = sorted(fpaths_ys_train)
        fpaths_xs_valid = sorted(fpaths_xs_valid)
        fpaths_ys_valid = sorted(fpaths_ys_valid)

        X_train = load_imgs_asarray(fpaths_xs_train, grayscale=False,
                                    target_size=target_size,
                                    dim_ordering=dim_ordering)
        Y_train = load_imgs_asarray(fpaths_ys_train, grayscale=True,
                                    target_size=target_size,
                                    dim_ordering=dim_ordering)
        X_valid = load_imgs_asarray(fpaths_xs_valid, grayscale=False,
                                    target_size=target_size,
                                    dim_ordering=dim_ordering)
        Y_valid = load_imgs_asarray(fpaths_ys_valid, grayscale=True,
                                    target_size=target_size,
                                    dim_ordering=dim_ordering)

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

        print('saving mean and standard deviation to ' + fname_stats + '...')
        stats = {'mean': mean, 'std': std}
        np.savez(fname_stats, **stats)
        print('==> done')

        print('globally normalizing data...')
        for i in range(3):
            X_train[:, i] = (X_train[:, i] - mean[i]) / std[i]
            X_valid[:, i] = (X_valid[:, i] - mean[i]) / std[i]
        Y_train /= 255
        Y_valid /= 255
        print('==> done')

        # モデルを作成
        print('creating model...')
        model = create_fcn(target_size)
        model.summary()

        # 損失関数，最適化手法を定義
        adam = Adam(lr=1e-5)
        model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])

        # 構造・重みを保存するディレクトリーの有無を確認
        dpath_checkpoints = os.path.join(dpath_this, dname_checkpoints)
        if not os.path.isdir(dpath_checkpoints):
            os.mkdir(dpath_checkpoints)

        # モデルの構造を保存
        json_string = model.to_json()
        fpath_architecture = os.path.join(dpath_checkpoints, fname_architecture)
        with open(fpath_architecture, 'wb') as f:
            f.write(json_string)

        # 重みを保存するためのオブジェクトを用意
        fpath_weights = os.path.join(dpath_checkpoints, fname_weights)
        checkpointer = ModelCheckpoint(filepath=fpath_weights,
                                       save_best_only=False)

        # トレーニングを開始
        print('start training...')
        model.fit(X_train, Y_train, batch_size=32, nb_epoch=20, verbose=1,
                  shuffle=True, validation_data=(X_valid, Y_valid),
                  callbacks=[checkpointer])

    else: # test
        # コマンドライン引数の正否をチェック
        assert(os.path.isfile(args.weights))

        # データを配列として取得
        print('loading data...')
        dpath_xs_test = os.path.join(dpath_data, 'test')
        fpaths_xs_test = list_pictures(dpath_xs_test)
        fnames_xs_test = [os.path.basename(fpath) for fpath in fpaths_xs_test]
        X_test = load_imgs_asarray(fpaths_xs_test, grayscale=False,
                                   target_size=target_size,
                                   dim_ordering=dim_ordering)
        print('==> ' + str(len(X_test)) +  ' test images loaded')

        # トレーニング時に計算した平均・標準偏差をロード
        print('loading mean and standard deviation from ' + fname_stats + '...')
        stats = np.load(fname_stats)
        mean = stats['mean']
        std = stats['std']
        print('==> mean: ' + str(mean))
        print('==> std : ' + str(std))

        print('globally normalizing data...')
        for i in range(3):
            X_test[:, i] = (X_test[:, i] - mean[i]) / std[i]
        print('==> done')

        # モデルを作成
        # （model_from_json()を使って保存してある構造を読み込むことも可能）
        print('creating model...')
        model = create_fcn(target_size)
        model.summary()

        # 学習済みの重みをロード
        fpath_weights = os.path.realpath(args.weights)
        print('loading weights from ' + fpath_weights)
        model.load_weights(fpath_weights)
        print('==> done')

        # テストを開始
        outputs = model.predict(X_test)

        # 出力を画像として保存
        dpath_outputs = os.path.join(dpath_this, dname_outputs)
        if not os.path.isdir(dpath_outputs):
            os.mkdir(dpath_outputs)

        print('saving outputs as images...')
        for i, array in enumerate(outputs):
            array = np.where(array > 0.5, 1, 0) # 二値に変換
            array = array.astype(np.float32)
            img_out = array_to_img(array, dim_ordering)
            fpath_out = os.path.join(dpath_outputs, fnames_xs_test[i])
            img_out.save(fpath_out)
        print('==> done')

