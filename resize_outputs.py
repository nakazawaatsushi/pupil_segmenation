#!/usr/bin/env python
import os

from PIL import Image
from keras.preprocessing.image import list_pictures, load_img


fpath_this = os.path.realpath(__file__)
dpath_this = os.path.dirname(fpath_this)

dpath_in = os.path.join(dpath_this, 'data/test')
dpath_out = os.path.join(dpath_this, 'outputs')

dpath_resized = os.path.join(dpath_this, 'resized')

fpaths_in = list_pictures(dpath_in)
fpaths_out = list_pictures(dpath_out)

fpaths_in = sorted(fpaths_in)
fpaths_out = sorted(fpaths_out)

if not os.path.isdir(dpath_resized):
    os.mkdir(dpath_resized)

print('resizing images...')
for (fpath_in, fpath_out) in zip(fpaths_in, fpaths_out):
    img_in = load_img(fpath_in)
    img_out = load_img(fpath_out, grayscale=True)
    img_resized = img_out.resize(img_in.size, resample=Image.LANCZOS)
    fname_resized = os.path.basename(fpath_in)
    fpath_resized = os.path.join(dpath_resized, fname_resized)
    img_resized.save(fpath_resized)
print('==> done')
