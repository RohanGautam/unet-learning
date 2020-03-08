import os
import numpy as np
import pickle
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from random import randint

from constants import *
TRAIN_PATH = "data/stage1_train/"
TEST_PATH = "data/stage1_test/"

def getData():
    # [1] to get only the filenames
    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]

    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH,
                        IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    print('Resizing training images and masks')
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                    mode='constant', preserve_range=True)
        X_train[n] = img  # fill empty X train with values from image
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(
                mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
            # combining all masks - at every pixel of final val, take max of values it encounters, will be 1 if there is white there.
            mask = np.maximum(mask, mask_)

        Y_train[n] = mask

    X_test = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH,
                    IMG_CHANNELS), dtype=np.uint8)

    print('Resizing training images and masks')
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                    mode='constant', preserve_range=True)
        X_test[n] = img  # fill empty X train with values from image

    # testing: show random image and it's mask:
    # im = randint(0, len(train_ids)-1)
    # imshow(X_train[im])
    # plt.show()
    # imshow(np.squeeze(Y_train[im]))
    # plt.show()

    # to not rerun this every time we need it
    pickle.dump(X_train, open('parsed_data/X_train.dat','wb'))
    pickle.dump(Y_train, open('parsed_data/Y_train.dat','wb'))
    pickle.dump(X_test, open('parsed_data/X_test.dat','wb'))
    return(X_train, Y_train, X_test)

