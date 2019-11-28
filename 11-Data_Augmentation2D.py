import random
import csv
import cv2 as cv
import os, sys
import numpy as np

from load_data_plaque import save_gradient_data, prior_probability

import PIL as plt

def pad_image2d(image, pad_shape, pad_mode='center'):

    padded_image = np.zeros((pad_shape[0], pad_shape[1]), image.dtype)

    if pad_mode == 'center':

        pady = np.abs(pad_shape[0] - image.shape[0])
        padx = np.abs(pad_shape[1] - image.shape[1])

        padded_image = np.pad(image, ((int(pady/2), pady-int(pady/2)), (int(padx/2), padx-int(padx/2))),
                              'constant', constant_values=(0, 0))

    if (pad_mode != 'center'):
        ValueError("Error cropping mode!")
        sys.exit(0)

    return padded_image

def get_image_list(csv_file, shuffle = False):
  image_list, label_list, shape_list = [], [], []

  with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)

    for row in reader:

        image_name = row['filepath'].replace('\t','').strip()
        label_name = row['labelpath'].replace('\t','').strip()
        # print(imagename)
        image_list.append(image_name)
        label_list.append(label_name)


  if shuffle:
    combinelist = list(zip(image_list, label_list, shape_list))
    random.shuffle(combinelist)
    image_list, label_list, shape_list = zip(*combinelist)


  data_list = {}
  data_list['image_filenames'] = image_list
  data_list['label_filenames'] = label_list
  data_list['shapes'] = shape_list

  return data_list



def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def normalize(image, shape, flag_mean = False, interpolate = cv.INTER_NEAREST):

    re_image = cv.resize(image, (shape[1], shape[0]), interpolation=interpolate)
    re_image = re_image.astype('float32')
    re_image /= 255
    if flag_mean:
        re_image -= np.mean(re_image)

    return re_image


# def get_roirect_fix(label, para_pad, wim, him, para_hwratio = 0.5):
#
#
#     flag_cut = True
#
#     minxlist = list()
#     maxxlist = list()
#     minylist = list()
#     maxylist = list()
#
#
#     locy, locx = np.where(label > 0)
#     centy = np.mean(locy)
#     centx = np.mean(locx)
#
#     maxx = int(np.max(locx)) + 1
#     minx = int(np.min(locx))
#     maxy = int(np.max(locy)) + 1
#     miny = int(np.min(locy))
#
#     if flag_cut:
#         pad = int(np.round((para_pad - 1) * (maxy - miny + 1)))
#
#         wroi0 = (maxx - minx + 1) + pad
#         hroi0 = (maxy - miny + 1) + pad
#
#
#         miny = int(np.round(centy - hroi0 / 2))
#         if miny < 1:
#             miny = 1
#
#         maxy = miny + hroi0 - 1
#         if maxy > him:
#             maxy = him
#             miny = maxy - hroi0 + 1
#
#
#         minx = int(round(centx - wroi0 / 2))
#         if minx < 1:
#             minx = 1
#
#         maxx = minx + wroi0 - 1
#         if maxx > wim:
#             maxx = wim
#             minx = maxx - wroi0 + 1
#         maxx0 = maxx
#         wsubroi = int(np.round(hroi0 / para_hwratio))
#         ncut = int(np.ceil(wroi0/wsubroi))2D_shuffle_large
#         for indexn in range(ncut):
#             minx += indexn * wsubroi
#             if ncut > 1 and (maxx0 - minx + 1) < wsubroi:
#                 maxx = maxx0
#                 minx = maxx - wsubroi + 1
#             else:
#                 maxx = minx + wsubroi - 1
#
#             # if minx < 0:
#             #     print('warning')
#
#             minxlist.append(minx)
#             maxxlist.append(maxx)
#             minylist.append(miny)
#             maxylist.append(maxy)
#
#     else:
#
#         wroi = int(np.round(para_pad * (maxx - minx + 1)))
#         hroi = int(np.round(wroi * para_hwratio))
#
#
#         miny = int(np.round(centy - hroi / 2))
#         if miny < 1:
#             miny = 1
#
#         maxy = miny + hroi - 1
#         if maxy > him:
#             maxy = himget_image_list
#             miny = maxy - hroi + 1
#
#
#         minx = int(round(centx - wroi / 2))
#         if minx < 1:
#             minx = 1
#
#         maxx = minx + wroi - 1
#         if maxx > wim:
#             maxx = wim
#             minx = maxx - wroi + 1
#
#         minxlist.append(minx)
#         maxxlist.append(maxx)
#         minylist.append(miny)
#         maxylist.append(maxy)
#
#     return minxlist,maxxlist,minylist,maxylist

def get_roirect_fix(label, para_pad, wim, him, para_hwratio = 0.7, cut_hwth = 0.33, flag_cut = True):

    minxlist = list()
    maxxlist = list()
    minylist = list()
    maxylist = list()


    locy, locx = np.where(label > 0)
    centy = np.mean(locy)
    centx = np.mean(locx)

    maxx = int(np.max(locx)) + 1
    minx = int(np.min(locx))
    maxy = int(np.max(locy)) + 1
    miny = int(np.min(locy))

    wroi0 = int((maxx - minx + 1) * para_pad)
    hroi0 = int((maxy - miny + 1) * para_pad)

    if flag_cut and (float(hroi0)/wroi0) < cut_hwth:

        miny = int(np.round(centy - hroi0 / 2))
        if miny < 1:
            miny = 1

        maxy = miny + hroi0 - 1
        if maxy > him:
            maxy = him
            miny = maxy - hroi0 + 1


        minx = int(round(centx - wroi0 / 2))
        if minx < 1:
            minx = 1

        maxx = minx + wroi0 - 1
        if maxx > wim:
            maxx = wim
            minx = maxx - wroi0 + 1
        maxx0 = maxx

        wsubroi = int(np.round(hroi0 / cut_hwth))
        ncut = int(np.ceil(wroi0/wsubroi))
        for indexn in range(ncut):
            minx += indexn * wsubroi
            if ncut > 1 and (maxx0 - minx + 1) < wsubroi:
                maxx = maxx0
                minx = maxx - wsubroi + 1
            else:
                maxx = minx + wsubroi - 1


            if maxx < 1:
                print('debug')

            if minx < 1:
                print('Debug')

            hsubroi = int(np.round(wsubroi * para_hwratio))

            miny = int(np.round(centy - hsubroi / 2))
            if miny < 1:
                miny = 1

            maxy = miny + hsubroi - 1
            if maxy > him:
                maxy = him
                miny = maxy - hsubroi + 1

            minxlist.append(minx)
            maxxlist.append(maxx)
            minylist.append(miny)
            maxylist.append(maxy)

    else:

        wroi = int(np.round(para_pad * (maxx - minx + 1)))
        hroi = int(np.round(wroi * para_hwratio))

        if hroi < hroi0:
            hroi = int(np.round(para_pad * (maxy - miny + 1)))
            wroi = int(np.round(hroi / para_hwratio))

        miny = int(np.round(centy - hroi / 2))
        if miny < 1:
            miny = 1

        maxy = miny + hroi - 1
        if maxy > him:
            maxy = him
            miny = maxy - hroi + 1


        minx = int(round(centx - wroi / 2))
        if minx < 1:
            minx = 1

        maxx = minx + wroi - 1
        if maxx > wim:
            maxx = wim
            minx = maxx - wroi + 1

        minxlist.append(minx)
        maxxlist.append(maxx)
        minylist.append(miny)
        maxylist.append(maxy)

    return minxlist,maxxlist,minylist,maxylist


def get_roirect2(label, para_pad, wim, him, para_hwratio = 0.667):#0.5

    minxlist = list()
    maxxlist = list()
    minylist = list()
    maxylist = list()

    locy, locx = np.where(label > 0)
    centy = np.mean(locy)
    centx = np.mean(locx)

    maxx = int(np.max(locx))
    minx = int(np.min(locx))

    maxy = int(np.max(locy))
    miny = int(np.min(locy))

    # wroi = int(np.round(para_pad * (maxx - minx + 1)))
    # hroi = int(np.round(para_pad * (maxy - miny + 1)))
    #
    # miny = int(np.round(centy - hroi / 2))
    # if miny < 1:
    #     miny = 1
    #
    # maxy = miny + hroi - 1
    # if maxy > him:
    #     maxy = him
    #     miny = maxy - hroi + 1
    #
    # minx = int(round(centx - wroi / 2))
    # if minx < 1:
    #     minx = 1
    #
    # maxx = minx + wroi - 1
    # if maxx > wim:
    #     maxx = wim
    #     minx = maxx - wroi + 1

    minxlist.append(minx)
    maxxlist.append(maxx)
    minylist.append(miny)
    maxylist.append(maxy)

    return minxlist,maxxlist,minylist,maxylist

def get_data(datalocs, Datafilelist, norm_size, augment=True, flag_freesize = False):

    para_hwratio = float(norm_size[0])/norm_size[1]
    para_pad = 1.2 #+ np.random.rand(1) * 0.4


    imagelist = []
    labellist = []

    para_flip = (-1, 0, 1, 2)

    get_roirect = get_roirect_fix
    # if flag_freesize:
    #     get_roirect = get_roirect2
    # else:
    #     get_roirect = get_roirect_fix

    # Get train data
    for index in datalocs:
        image_name = Datafilelist['image_filenames'][index]
        label_name = Datafilelist['label_filenames'][index]
        print('Get data from %s' % (image_name))

        image = cv.imread(image_name, flags=cv.IMREAD_GRAYSCALE)
        label = cv.imread(label_name, flags=cv.IMREAD_GRAYSCALE)

        rows, cols = image.shape

        minxlist, maxxlist, minylist, maxylist = get_roirect(label, para_pad, cols, rows, para_hwratio, flag_cut = False)#, flag_cut = False
        for ind_sub in range(len(minxlist)):
            minx = minxlist[ind_sub]
            maxx = maxxlist[ind_sub] + 1
            miny = minylist[ind_sub]
            maxy = maxylist[ind_sub] + 1

            image_roi = image[miny:maxy, minx:maxx]
            label_roi = label[miny:maxy, minx:maxx]
            if flag_freesize:
                normw = int(norm_size[0] * float(image_roi.shape[1])/float(image_roi.shape[0]))
                shape = (norm_size[0], normw)
            else:
                shape = norm_size

            image_norm = normalize(image_roi, shape, flag_mean=True, interpolate=cv.INTER_LINEAR)
            label_norm = normalize(label_roi, shape)
            imagelist.append(image_norm)
            labellist.append(label_norm)

        if augment:
            for ind_flid in range(len(para_flip)):
                if para_flip[ind_flid] == 2:
                    image_flip = image
                    label_flip = label
                else:
                    #flid
                    image_flip = cv.flip(image, para_flip[ind_flid])
                    label_flip = cv.flip(label, para_flip[ind_flid])

                locy, locx = np.where(label_flip > 0)
                centy = np.mean(locy)
                centx = np.mean(locx)


                #rotate
                angle = random.randint(-20, 20)
                # print('angle%d \n'%(angle))
                M = cv.getRotationMatrix2D((centx, centy), angle, 1)
                image_rotate = cv.warpAffine(image_flip, M, (cols, rows))
                label_rotate = cv.warpAffine(label_flip, M, (cols, rows))


                minxlist, maxxlist, minylist, maxylist = get_roirect(label_flip, para_pad, cols, rows, para_hwratio, flag_cut = False)#, flag_cut = False
                for ind_sub in range(len(minxlist)):
                    minx = minxlist[ind_sub]
                    maxx = maxxlist[ind_sub] + 1
                    miny = minylist[ind_sub]
                    maxy = maxylist[ind_sub] + 1
                    image_flip_roi = image_flip[miny:maxy, minx:maxx]
                    label_flip_roi = label_flip[miny:maxy, minx:maxx]
                    if flag_freesize:
                        normw = int(norm_size[0] * float(image_flip_roi.shape[1]) / float(image_flip_roi.shape[0]))
                        shape = (norm_size[0], normw)
                    else:
                        shape = norm_size
                    image_norm = normalize(image_flip_roi, shape, flag_mean=True, interpolate=cv.INTER_LINEAR)
                    label_norm = normalize(label_flip_roi, shape)
                    imagelist.append(image_norm)
                    labellist.append(label_norm)

                minxlist, maxxlist, minylist, maxylist = get_roirect(label_rotate, para_pad, cols, rows, para_hwratio, flag_cut = False)#, flag_cut = False
                for ind_sub in range(len(minxlist)):
                    minx = minxlist[ind_sub]
                    maxx = maxxlist[ind_sub] + 1
                    miny = minylist[ind_sub]
                    maxy = maxylist[ind_sub] + 1
                    image_rotate_roi = image_rotate[miny:maxy, minx:maxx]
                    label_rotate_roi = label_rotate[miny:maxy, minx:maxx]
                    if flag_freesize:
                        normw = int(norm_size[0] * float(label_rotate_roi.shape[1]) / float(label_rotate_roi.shape[0]))
                        shape = (norm_size[0], normw)
                    else:
                        shape = norm_size
                    image_norm = normalize(image_rotate_roi, shape, flag_mean=True, interpolate=cv.INTER_LINEAR)
                    label_norm = normalize(label_rotate_roi, shape)
                    imagelist.append(image_norm)
                    labellist.append(label_norm)

    return imagelist, labellist

def load_data(data_csv, trainpart, shape):


  Datafilelist = get_image_list(data_csv)


  num_files = len(Datafilelist['image_filenames'])
  trainlocs = [i for i in range(487)]
  testlocs = [i for i in range(487, num_files)]

  trainimagelist, trainlabellist = get_data(trainlocs, Datafilelist, shape, flag_freesize=False)


  #Get test data
  testimagelist, testlabellist = get_data(testlocs, Datafilelist, shape, augment=False, flag_freesize=False)



  return trainimagelist, trainlabellist, testimagelist, testlabellist

def modify_roi(minx, maxx, miny, maxy, wim, him, para_pad = 1.2, para_hwratio = 0.667, cut_hwth = 0.33, flag_cut = True):
    centy = (miny+maxy)/2
    centx = (minx+maxx)/2

    wroi0 = int((maxx - minx + 1) * para_pad)
    hroi0 = int((maxy - miny + 1) * para_pad)

    if flag_cut and (float(hroi0)/wroi0) < cut_hwth:

        miny = int(np.round(centy - hroi0 / 2))
        if miny < 1:
            miny = 1

        maxy = miny + hroi0 - 1
        if maxy > him:
            maxy = him
            miny = maxy - hroi0 + 1


        minx = int(round(centx - wroi0 / 2))
        if minx < 1:
            minx = 1

        maxx = minx + wroi0 - 1
        if maxx > wim:
            maxx = wim
            minx = maxx - wroi0 + 1
        maxx0 = maxx

        wsubroi = int(np.round(hroi0 / cut_hwth))
        ncut = int(np.ceil(wroi0/wsubroi))
        for indexn in range(ncut):
            minx += indexn * wsubroi
            if ncut > 1 and (maxx0 - minx + 1) < wsubroi:
                maxx = maxx0
                minx = maxx - wsubroi + 1
            else:
                maxx = minx + wsubroi - 1


            if maxx < 1:
                print('debug')

            if minx < 1:
                print('Debug')

            hsubroi = int(np.round(wsubroi * para_hwratio))

            miny = int(np.round(centy - hsubroi / 2))
            if miny < 1:
                miny = 1

            maxy = miny + hsubroi - 1
            if maxy > him:
                maxy = him
                miny = maxy - hsubroi + 1


    else:

        wroi = int(np.round(para_pad * (maxx - minx + 1)))
        hroi = int(np.round(wroi * para_hwratio))

        if hroi < hroi0:
            hroi = int(np.round(para_pad * (maxy - miny + 1)))
            wroi = int(np.round(hroi / para_hwratio))

        miny = int(np.round(centy - hroi / 2))
        if miny < 1:
            miny = 1

        maxy = miny + hroi - 1
        if maxy > him:
            maxy = him
            miny = maxy - hroi + 1


        minx = int(round(centx - wroi / 2))
        if minx < 1:
            minx = 1

        maxx = minx + wroi - 1
        if maxx > wim:
            maxx = wim
            minx = maxx - wroi + 1


    return minx,maxx,miny,maxy

if __name__=='__main__':



  trainimagelist, trainlabellist, testimagelist, testlabellist = load_data('../Datalist_2DPlaque_SPARCM_timepoint_1010.csv', 1, shape=(96, 144))

  train_images = np.array(trainimagelist)
  train_images = np.expand_dims(train_images, axis=-1)
  train_labels = np.array(trainlabellist)
  train_labels = np.expand_dims(train_labels, axis=-1)

  test_images = np.array(testimagelist)
  test_images = np.expand_dims(test_images, axis=-1)
  test_labels = np.array(testlabellist)
  test_labels = np.expand_dims(test_labels, axis=-1)

  savepath = './npydata_M_test_1127'



  if not os.path.exists(savepath):
    os.mkdir(savepath)
    print("Directory ", savepath, " Created ")
  else:
    print("Directory ", savepath, " already exists")

  np.save(savepath + '/imgs_train.npy', train_images)
  np.save(savepath + '/imgs_mask_train.npy', train_labels)
  np.save(savepath + '/imgs_test.npy', test_images)
  np.save(savepath + '/test_mask.npy', test_labels)


  # save_gradient_data(savepath)
  # probmap = prior_probability('./npydata')


