#!/usr/bin/env python
from __future__ import print_function
import os
from shutil import copyfile


print('copying images...')

dpath_home = os.path.expanduser('~')
fpath_this = os.path.realpath(__file__)
dpath_this = os.path.dirname(fpath_this)
dpath_src = os.path.join(dpath_home, 'data/101_ObjectCategories/airplanes')

fnames_test = ['image_0027.jpg', 'image_0028.jpg', 'image_0029.jpg',
               'image_0030.jpg', 'image_0039.jpg', 'image_0060.jpg',
               'image_0062.jpg', 'image_0063.jpg', 'image_0065.jpg',
               'image_0537.jpg']

for ds_name in ['train', 'valid', 'test']:
    dpath_dst = os.path.join(dpath_this, 'data', ds_name)

    if not os.path.isdir(dpath_dst):
        os.mkdir(dpath_dst)

    if ds_name != 'test':
        dpath = os.path.join(dpath_this, 'data', ds_name + '_mask')
        fnames = os.listdir(dpath)
    else:
        fnames = fnames_test

    for fname in fnames:
        src = os.path.join(dpath_src, fname)
        dst = os.path.join(dpath_dst, fname)
        copyfile(src, dst)

print('==> done')
