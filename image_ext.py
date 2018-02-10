import numpy
from keras.preprocessing.image import list_pictures, load_img, img_to_array


def list_pictures_in_multidir(paths):
    fpaths = []
    for path in paths:
        fpaths += list_pictures(path)
    return fpaths

def load_imgs_asarray(paths, grayscale=False, target_size=None,
                      dim_ordering='default'):
    arrays = []
    for path in paths:
        img = load_img(path, grayscale, target_size)
        array = img_to_array(img, dim_ordering)
        arrays.append(array)
    return numpy.asarray(arrays)

def img_dice_coeff(im1,im2):
    # Compute dice coeff
    im1a = numpy.array(im1)
    im1a[im1a > 0] = 1
    im2a = numpy.array(im2)
    im2a[im2a > 0] = 1
    
    overlap_a = numpy.array(im1a) * numpy.array(im2a)
    overlap_b = numpy.array(im1a) + numpy.array(im2a)
    
    return (2*sum(sum(overlap_a))/sum(sum(overlap_b)))

def get_center(im):
    im[im>0] = 1;
    xval = 0
    yval = 0
    npix = 0

    for x in range(0,im.shape[1]):
        xval += (x*sum(im[:,x]))
        npix += sum(im[:,x])
    
    for y in range(0,im.shape[0]):
        yval += (y*sum(im[y,:]))
    
    if npix == 0:
        return [-1,-1]
    else:
        return [xval/npix,yval/npix]
