import csv
import  random, os
import cv2 as cv
import pandas as pd
from model_2DUNet import my2DUNet
from keras.preprocessing.image import array_to_img
from keras.optimizers import *
import scipy.io as sio
from LossMetrics import *
from Data_Augmentation2D import normalize, get_roirect_fix, get_image_list, modify_roi

import matplotlib.pyplot as plt
import scipy.misc

def takeSecond(elem):
    return elem[1]

if __name__ == '__main__':

    data_csv = './1112/Datalist_qiong_1112.csv'
    note = './results_roi_qiong_1112_finalunet_2/'
    modelpath = './log1111_final_unet/plaque_UNet.hdf5'

    lognoteDir = './' + note
    result_csv = lognoteDir + '/result.csv'
    savepath = lognoteDir + '/images'
    save_root_dir = './' + note
    shape = (96, 144)
    para_hwratio = float(shape[0]) / shape[1]
    para_pad = 1.2  # + np.random.rand(1) * 0.4
    lognoteDir = './' + note
    if os.path.exists(lognoteDir) is True:
        print('Exist %s' % lognoteDir)
    else:
        os.mkdir(lognoteDir)

    if os.path.exists(savepath) is True:
        print('Exist %s' % savepath)
    else:
        os.mkdir(savepath)

    plaque_areas_label = list()
    error_plaque_areas = list()
    plaque_areas_pred = list()


    # mynet = VNet(shape[0], shape[1], 1)
    mynet = my2DUNet(None, None, 1)
    # mynet = FCN(img_rows, img_cols)
    #
    # model = mynet.model(classes=1, kernel_size=(5, 5))
    model = mynet.model(classes=1, stages = 4, base_filters = 64, activation_name='sigmoid', deconvolution = False)
    # model = mynet.FCN_Vgg16_8s(weight_decay=0.001, classes=1, activate='sigmoid')

    model.load_weights(modelpath)

    model.compile(optimizer=adam(lr=1e-4), loss=DSCLoss, metrics=[DSC2D, 'accuracy'])

    #get imagelist from csv file
    image_list, minx_list, miny_list, maxx_list, maxy_list, res_list, area_pre_list = [], [], [], [], [], [], []

    with open(data_csv, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_name = row['filepath'].replace('\t', '').strip()
            print(image_name)
            image_list.append(image_name)
            pos = image_name.find('.png')

            minx = row[' minx'].replace('\t', '').strip()
            #minx_list.append(minx)
            miny = row[' miny'].replace('\t', '').strip()
            #miny_list.append(miny)
            maxx = row[' maxx'].replace('\t', '').strip()
            #maxx_list.append(maxx)
            maxy = row[' maxy'].replace('\t', '').strip()
            #maxy_list.append(maxy)
            #res_list.append(res)
            # print(minx)
            # print(maxx)
            # print(miny)
            # print(maxy)


            image_name = image_name
            print('Get data from %s' % (image_name))
            image = cv.imread(image_name, flags=cv.IMREAD_GRAYSCALE)
            rows, cols = image.shape
            #modify ROI
            minx,maxx,miny,maxy = modify_roi(int(minx), int(maxx), int(miny), int(maxy), cols, rows)
            minx_list.append(minx)
            maxx_list.append(maxx)
            miny_list.append(miny)
            maxy_list.append(maxy)

            image_roi = image[int(miny):int(maxy)+1, int(minx):int(maxx)+1]
            image_norm = normalize(image_roi, (shape[0], shape[1]), flag_mean=True, interpolate=cv.INTER_LINEAR)
            #image_norm = normalize(image_roi, (int(maxy+1-miny),int(maxx+1-minx)),flag_mean=True, interpolate=cv.INTER_LINEAR)
            # roi_col, roi_row = image_norm.resize
            # print('col:', roi_col)
            # print('row:', roi_row)
            image_norm = np.expand_dims(image_norm, axis=-1)
            image_norm = np.expand_dims(image_norm, axis=0)
            # image_norm = np.expand_dims(image_roi, axis=-1)
            # image_norm = np.expand_dims(image_norm, axis=0)
            pred = model.predict(image_norm, verbose=1)

            pred_resize = cv.resize(pred[0, :, :, 0], (image_roi.shape[1], image_roi.shape[0]), 0)
            ret, pred_binary = cv.threshold(pred_resize, 0.5, 1, cv.THRESH_BINARY)
            area_pre = int(np.sum(pred_binary))
            area_pre_list.append(area_pre)
            tmpimname = image_name
            loc = tmpimname.rfind('/')
            saveimpath = os.path.join(savepath, '%s.bmp' % tmpimname[loc + 1: -4])
            savepredpath = os.path.join(savepath, '%s_pred.bmp' % tmpimname[loc + 1: -4])
            scipy.misc.imsave(saveimpath, image_roi, 'bmp')
            scipy.misc.imsave(savepredpath, pred_binary, 'bmp')

    dataframe = pd.DataFrame({'image_filenames': image_list,
                              'minx': minx_list,
                              'maxx': maxx_list,
                              'miny': miny_list,
                              'maxy': maxy_list,
                              'areas_pred': area_pre_list,})
    dataframe.to_csv(result_csv, index=False, sep=',')