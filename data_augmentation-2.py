#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os

import numpy as np
np.random.seed(2016)
from keras.preprocessing.image import transform_matrix_offset_center, \
                                      apply_transform, \
                                      flip_axis, \
                                      array_to_img, \
                                      list_pictures, \
                                      ImageDataGenerator, \
                                      Iterator

from image_ext import load_imgs_asarray


class ImagePairDataGenerator(ImageDataGenerator):

    def __init__(self, *args, **kwargs):
        super(ImagePairDataGenerator, self).__init__(*args, **kwargs)

    def flow(self, X, Y, batch_size=32, shuffle=True, seed=None,
             save_to_dir_x=None, save_to_dir_y=None,
             save_prefix_x='', save_prefix_y='',
             save_prefixes_x=None, save_prefixes_y=None,
             save_format='jpeg'):
        return NumpyArrayIterator(
            X, Y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir_x=save_to_dir_x, save_to_dir_y=save_to_dir_y,
            save_prefixes_x=save_prefixes_x, save_prefixes_y=save_prefixes_y,
            save_format=save_format)

    def flow_from_directory(self):
        raise NotImplementedError

    # 2画像x, yを引数に
    def random_transform(self, x, y):
        # バックエンドがTheano/TensorFlowのどちらでも同じ処理で変換を行えるよう
        # インデックスの位置を取得する
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        # 変換のための行列を作る
        # use composition of homographies to generate final transform that
        # needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)

        # yにも同様の変換を施す
        y = apply_transform(y, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)

        # 実装されているチャンネルシフトは2画像に適用できないのでコメントアウト
        # if self.channel_shift_range != 0:
        #     x = random_channel_shift(x, self.channel_shift_range,
        #                                img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                # yも反転
                y = flip_axis(y, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                # yも反転
                y = flip_axis(y, img_row_index)

        return x, y

class NumpyArrayIterator(Iterator):

    def __init__(self, X, Y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir_x=None, save_to_dir_y=None,
                 save_prefix_x='', save_prefix_y='',
                 save_prefixes_x=None, save_prefixes_y=None,
                 save_format='jpeg'):
        if Y is not None and len(X) != len(Y):
            raise Exception('X (images tensor) and y (images tensor) '
                            'should have the same length. '
                            'Found: X.shape = %s, Y.shape = %s' % (np.asarray(X).shape, np.asarray(Y).shape))
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.X = X
        self.Y = Y
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir_x = save_to_dir_x
        self.save_to_dir_y = save_to_dir_y
        self.save_prefix_x = save_prefix_x
        self.save_prefix_y = save_prefix_y
        self.save_prefixes_x = save_prefixes_x
        self.save_prefixes_y = save_prefixes_y
        self.save_format = save_format
        super(NumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        batch_y = np.zeros(tuple([current_batch_size] + list(self.Y.shape)[1:]))
        if self.save_prefixes_x:
            batch_prefixes_x = ['' for i in range(current_batch_size)]
        if self.save_prefixes_y:
            batch_prefixes_y = ['' for i in range(current_batch_size)]
        for i, j in enumerate(index_array):
            x = self.X[j]
            y = self.Y[j]
            x, y = self.image_data_generator.random_transform(x.astype('float32'),
                                                              y.astype('float32'))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y[i] = y
            if self.save_prefixes_x is not None:
                batch_prefixes_x[i] = self.save_prefixes_x[j]
            if self.save_prefixes_y is not None:
                batch_prefixes_y[i] = self.save_prefixes_y[j]
        hash_val = np.random.randint(1e4)

        if self.save_to_dir_x:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                if self.save_prefixes_x is None:
                    fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix_x,
                                                                      index=current_index + i,
                                                                      hash=hash_val,
                                                                      format=self.save_format)
                else:
                    fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=batch_prefixes_x[i],
                                                                      index=current_index + i,
                                                                      hash=hash_val,
                                                                      format=self.save_format)
                img.save(os.path.join(self.save_to_dir_x, fname))
        if self.save_to_dir_y:
            for i in range(current_batch_size):
                img = array_to_img(batch_y[i], self.dim_ordering, scale=True)
                if self.save_prefixes_y is None:
                    fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix_y,
                                                                      index=current_index + i,
                                                                      hash=hash_val,
                                                                      format=self.save_format)
                else:
                    fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=batch_prefixes_y[i],
                                                                      index=current_index + i,
                                                                      hash=hash_val,
                                                                      format=self.save_format)
                img.save(os.path.join(self.save_to_dir_y, fname))
        return batch_x, batch_y

def augment_img_pairs(dpath_src_x, dpath_src_y, dpath_dst_x, dpath_dst_y,
                      target_size, grayscale_x=False, grayscale_y=False,
                      nb_times=1,
                      rotation_range=0.,
                      width_shift_range=0.,
                      height_shift_range=0.,
                      shear_range=0.,
                      zoom_range=0.,
                      dim_ordering='default'):
    print('loading images from ' + dpath_src_x)
    print('loading images from ' + dpath_src_y)

    # numpy.ndarray型で画像を取得
    fpaths_x = list_pictures(dpath_src_x)
    fpaths_y = list_pictures(dpath_src_y)

    fpaths_x = sorted(fpaths_x)
    fpaths_y = sorted(fpaths_y)

    X = load_imgs_asarray(fpaths_x, grayscale=grayscale_x,
                          target_size=target_size, dim_ordering=dim_ordering)
    Y = load_imgs_asarray(fpaths_y, grayscale=grayscale_y,
                          target_size=target_size, dim_ordering=dim_ordering)

    assert(len(X) == len(Y))
    nb_pairs = len(X)
    print('==> ' + str(nb_pairs) + ' pairs loaded')

    # データ生成器を準備
    datagen = ImagePairDataGenerator(rotation_range=rotation_range,
                                     width_shift_range=width_shift_range,
                                     height_shift_range=height_shift_range,
                                     shear_range=shear_range,
                                     zoom_range=zoom_range)

    # ファイル名（拡張子なし）を取得
    froots_x = []
    for fpath_x in fpaths_x:
        basename = os.path.basename(fpath_x)
        froot_x = os.path.splitext(basename)[0]
        froots_x.append(froot_x)

    froots_y = []
    for fpath_y in fpaths_y:
        basename = os.path.basename(fpath_y)
        froot_y = os.path.splitext(basename)[0]
        froots_y.append(froot_y)

    # データを拡張
    print('augmenting data...')
    i = 0
    for batch in datagen.flow(X, Y, batch_size=nb_pairs, shuffle=False,
                              save_to_dir_x=dpath_dst_x,
                              save_to_dir_y=dpath_dst_y,
                              save_prefixes_x=froots_x,
                              save_prefixes_y=froots_y):
        i += 1
        if i >= nb_times:
            break
    print('==> ' + str(nb_times*nb_pairs) + ' pairs created')

if __name__ == '__main__':
    # オプション
    dname_out_suffix = '-aug'
    target_size = (224, 224)
    nb_times = 25
    rotation_range = 15
    width_shift_range = 0.15
    height_shift_range = 0.15
    shear_range = 0.35
    zoom_range = 0.3
    dim_ordering = 'th'

    # プロジェクト内のデータディレクトリーのパスを取得
    fpath_this = os.path.realpath(__file__)
    dpath_this = os.path.dirname(fpath_this)
    dpath_data = os.path.join(dpath_this, 'data')

    # トレーニングデータを拡張
    print('\n# training data augmentation')
    dpath_img_in = os.path.join(dpath_data, 'train')
    dpath_mask_in = os.path.join(dpath_data, 'train_mask')

    dpath_img_out = dpath_img_in + dname_out_suffix
    dpath_mask_out = dpath_mask_in + dname_out_suffix

    if not os.path.isdir(dpath_img_out):
        os.mkdir(dpath_img_out)
    if not os.path.isdir(dpath_mask_out):
        os.mkdir(dpath_mask_out)

    augment_img_pairs(dpath_img_in, dpath_mask_in,
                      dpath_img_out, dpath_mask_out,
                      target_size,
                      grayscale_x=False, grayscale_y=True,
                      nb_times=nb_times,
                      rotation_range=rotation_range,
                      width_shift_range=width_shift_range,
                      height_shift_range=height_shift_range,
                      shear_range=shear_range,
                      zoom_range=zoom_range,
                      dim_ordering=dim_ordering)

    # バリデーションデータを拡張
    print('\n# validation data augmentation')
    dpath_img_in = os.path.join(dpath_data, 'valid')
    dpath_mask_in = os.path.join(dpath_data, 'valid_mask')

    dpath_img_out = dpath_img_in + dname_out_suffix
    dpath_mask_out = dpath_mask_in + dname_out_suffix

    if not os.path.isdir(dpath_img_out):
        os.mkdir(dpath_img_out)
    if not os.path.isdir(dpath_mask_out):
        os.mkdir(dpath_mask_out)

    augment_img_pairs(dpath_img_in, dpath_mask_in,
                      dpath_img_out, dpath_mask_out,
                      target_size,
                      grayscale_x=False, grayscale_y=True,
                      nb_times=nb_times,
                      rotation_range=rotation_range,
                      width_shift_range=width_shift_range,
                      height_shift_range=height_shift_range,
                      shear_range=shear_range,
                      zoom_range=zoom_range,
                      dim_ordering=dim_ordering)
