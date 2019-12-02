import  random, os
import cv2 as cv
import pandas as pd
from model_2DUNet import my2DUNet
from keras.preprocessing.image import array_to_img
from keras.optimizers import *
import scipy.io as sio
from LossMetrics import *
from Data_Augmentation2D import normalize, get_roirect_fix, get_image_list,get_roirect2

import matplotlib.pyplot as plt
import scipy.misc

def takeSecond(elem):
    return elem[1]

if __name__ == '__main__':

    data_csv = '../DatalistC_1129_test.csv'
    data_root = '../local'
    note = 'results_chris_net_1129'
    modelpath = './log1125_final_chris_lr4_unet/plaque_UNet.hdf5'
    lognoteDir = './' + note
    result_csv = lognoteDir + '/result.csv'
    savepath = lognoteDir + '/images'
    save_root_dir = './' + note
    shape = (96, 144)
    Dice = 0
    DSC_full = 0
    augment = False

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

    get_roirect = get_roirect_fix

    # mynet = VNet(shape[0], shape[1], 1)
    mynet = my2DUNet(None, None, 1)
    # mynet = FCN(img_rows, img_cols)
    #
    # model = mynet.model(classes=1, kernel_size=(5, 5))
    model = mynet.model(classes=1, stages = 4, base_filters = 64, activation_name='sigmoid', deconvolution = False)
    # model = mynet.FCN_Vgg16_8s(weight_decay=0.001, classes=1, activate='sigmoid')

    model.load_weights(modelpath)


    model.compile(optimizer=adam(lr=1e-4), loss=DSCLoss, metrics=[DSC2D, 'accuracy'])

    Datafilelist = get_image_list(data_csv)
    num_files = len(Datafilelist['image_filenames'])
    #testlocs = [i for i in range(int(num_files * 0.67), num_files)]
    testlocs = [i for i in range(1, num_files)]

    para_hwratio = float(shape[0])/shape[1]
    para_pad = 1.2 #+ np.random.rand(1) * 0.4

    imagelist = []
    labellist = []

    para_flip = random.randint(-1, 2)
    angle = random.randint(-45, 45)
    # Get train data
    for index in testlocs:
        image_name = os.path.join(data_root, Datafilelist['image_filenames'][index])
        label_name = os.path.join(data_root, Datafilelist['label_filenames'][index])
        print('Get data from %s' % (image_name))

        image = cv.imread(image_name, flags=cv.IMREAD_GRAYSCALE)
        label = cv.imread(label_name, flags=cv.IMREAD_GRAYSCALE)

        rows, cols = image.shape

        minxlist, maxxlist, minylist, maxylist = get_roirect(label, para_pad, cols, rows, para_hwratio,flag_cut = False)#,flag_cut = False
        minx = minxlist[0]
        maxx = maxxlist[0] + 1
        miny = minylist[0]
        maxy = maxylist[0] + 1

        image_roi = image[miny:maxy, minx:maxx]
        label_roi = label[miny:maxy, minx:maxx]

        if augment:
            if para_flip < 2:
                image_flip = cv.flip(image, para_flip)
                label_flip = cv.flip(label, para_flip)
            else:
                image_flip = image
                label_flip = label

            locy, locx = np.where(label_flip > 0)
            centy = np.mean(locy)
            centx = np.mean(locx)
            M = cv.getRotationMatrix2D((centx, centy), angle, 1)
            image_rotate = cv.warpAffine(image_flip, M, (cols, rows), flags=cv.INTER_LINEAR)
            label_rotate = cv.warpAffine(label_flip, M, (cols, rows), flags=cv.INTER_NEAREST)
            minxlist1, maxxlist1, minylist1, maxylist1 = get_roirect(label_rotate, cols, rows, para_pad, para_hwratio, flag_cut = False)#, flag_cut = False
            minx = minxlist1[0]
            maxx = maxxlist1[0] + 1
            miny = minylist1[0]
            maxy = maxylist1[0] + 1

            image_rotate_roi = image_rotate[miny:maxy, minx:maxx]
            label_rotate_roi = label_rotate[miny:maxy, minx:maxx]
            image_norm = normalize(image_rotate_roi, (shape[1], shape[0]), flag_mean=True, interpolate=cv.INTER_LINEAR)
            label_norm = normalize(label_rotate_roi, (shape[1], shape[0]))

            image_norm = np.expand_dims(image_norm, axis=-1)
            image_norm = np.expand_dims(image_norm, axis=0)
            label_norm = np.expand_dims(label_norm, axis=-1)
            label_norm = np.expand_dims(label_norm, axis=0)

            pred = model.predict(image_norm, verbose=1)
            metrics = model.evaluate(x=image_norm, y=label_norm, verbose=1)
            Dice += metrics[1]

            pred_resize = cv.resize(pred[0,:,:,0], (image_rotate_roi.shape[1], image_rotate_roi.shape[0]), 0)
            ret, pred_binary = cv.threshold(pred_resize, 0.5, 1, cv.THRESH_BINARY)
            area_pre = np.sum(pred_binary)
            plaque_areas_pred.append(area_pre)
            area_label = np.sum(label_rotate_roi/255)
            plaque_areas_label.append(area_label)
            error_plaque_areas.append(area_label-area_pre)

            pred_fs = np.zeros(shape=image_rotate.shape)
            pred_fs[miny:maxy, minx:maxx] = pred_binary
            # M = cv.getRotationMatrix2D((centx, centy), -angle, 1)
            pred_fs = cv.warpAffine(pred_fs, M, (cols, rows),flags=cv.WARP_INVERSE_MAP | cv.INTER_NEAREST)
            pred_fs = cv.flip(pred_fs, para_flip)

        else:
            image_norm = normalize(image_roi, (shape[0], shape[1]), flag_mean=True, interpolate=cv.INTER_LINEAR)
            label_norm = normalize(label_roi, (shape[0], shape[1]))

            image_norm = np.expand_dims(image_norm, axis=-1)
            image_norm = np.expand_dims(image_norm, axis=0)
            label_norm = np.expand_dims(label_norm, axis=-1)
            label_norm = np.expand_dims(label_norm, axis=0)

            pred = model.predict(image_norm, verbose=1)
            metrics = model.evaluate(x=image_norm, y=label_norm, verbose=1)
            Dice += metrics[1]

            pred_resize = cv.resize(pred[0,:,:,0], (image_roi.shape[1], image_roi.shape[0]), 0)
            ret, pred_binary = cv.threshold(pred_resize, 0.5, 1, cv.THRESH_BINARY)
            area_pre = int(np.sum(pred_binary))*0.065*0.065
            plaque_areas_pred.append(area_pre)
            area_label = int(np.sum(label_roi/255))*0.065*0.065
            plaque_areas_label.append((area_label+area_pre)/2)
            error_plaque_areas.append(area_label-area_pre)

            pred_fs = np.zeros(shape=image.shape)
            pred_fs[miny:maxy, minx:maxx] = pred_binary


            tmpimname = Datafilelist['image_filenames'][index]
            loc = tmpimname.rfind('/')
            saveimpath = os.path.join(savepath, '%s.bmp'%tmpimname[loc + 1 : -4])
            savelabelpath = os.path.join(savepath, '%s_label.bmp'%tmpimname[loc + 1 : -4])
            savepredpath = os.path.join(savepath, '%s_pred.bmp' % tmpimname[loc + 1: -4])
            scipy.misc.imsave(saveimpath, image_roi, 'bmp')
            scipy.misc.imsave(savelabelpath, label_roi, 'bmp')
            scipy.misc.imsave(savepredpath, pred_binary, 'bmp')


        transaction = np.multiply(pred_fs, label/255)
        DSC_full += 2*np.sum(transaction)/float(np.sum(pred_fs) + np.sum(label/255))



            # tmppred = array_to_img(pred)
            # tmplabel = array_to_img(label_norm)
            # tmpimg= array_to_img(image_norm)
            #
            # tmppred.save(save_root_dir+'/pred/%d.bmp'%(index))
            # # sio.savemat('./results/mat_softmax/%d.mat'%(index), {'array': pred[index, :, :, :]})
            # tmplabel.save(save_root_dir + '/label/%d.bmp' % (index))
            # tmpimg.save(save_root_dir + '/image/%d.bmp' % (index))
    Dice /= len(testlocs)
    DSC_full /= len(testlocs)

    print('Average Dice is %f'% (Dice))
    print('Average DSC_full is %f'% (DSC_full))

    dataframe = pd.DataFrame({'image_filenames': Datafilelist['image_filenames'][1 : num_files],
                              'plaque_areas_label': plaque_areas_label,
                              'plaque_areas_pred': plaque_areas_pred,
                              'error_plaque_areas': error_plaque_areas})
    dataframe.to_csv(result_csv, index=False, sep=',')


    np.save(note + '/plaque_areas_pred.npy', plaque_areas_pred)
    np.save(note + '/plaque_areas_label.npy', plaque_areas_label)
    np.save(note + '/error_plaque_areas.npy', error_plaque_areas)

    combinelist = list(zip(plaque_areas_pred, plaque_areas_label, error_plaque_areas))
    combinelist.sort(key=takeSecond)
    plaque_areas_pred,plaque_areas_label,error_plaque_areas = zip(*combinelist)


    bin_step = 1500
    area_range = range(0,int(np.max(plaque_areas_label)), bin_step)

    # area_error_hist = np.zeros(shape=len(area_range) + 1)
    # num_hist = np.zeros(shape=len(area_range) + 1)
    #
    # for index_plaque in range(len(plaque_areas_label)):
    #     tmparea = plaque_areas_label[index_plaque]
    #     for index in range(len(area_range)):
    #         if tmparea<area_range[index] and tmparea>area_range[index] - bin_step:
    #             area_error_hist[index] += error_plaque_areas[index_plaque]
    #             num_hist[index] += 1
    #
    #     if tmparea > 10000:
    #         area_error_hist[len(area_range)] += error_plaque_areas[index_plaque]
    #         num_hist[len(area_range)] += 1
    #
    # area_hist = np.divide(area_error_hist, num_hist)


    area_error_list = list()
    area_range = list(area_range)
    area_range.append(int(np.max(plaque_areas_label)))

    areaerror_mean_hist = np.zeros(shape=len(area_range))
    areaerror_std_hist = np.zeros(shape=len(area_range))

    for index in range(len(area_range) - 1):
        area_bin = list()
        for index_plaque in range(len(plaque_areas_label)):
            tmparea = plaque_areas_label[index_plaque]
            if tmparea<area_range[index + 1] and tmparea>area_range[index]:
                area_bin.append(np.abs(error_plaque_areas[index_plaque]))
        area_error_list.append(area_bin)
        areaerror_mean_hist[index] = np.mean(np.array(area_bin))
        areaerror_std_hist[index] = np.std(np.array(area_bin))
        area_bin.clear()


    mean_error = np.mean(error_plaque_areas)
    std_error = np.std(error_plaque_areas)
    plt.figure()
    plt.scatter(plaque_areas_label, abs(error_plaque_areas), c='r', marker='o')
    plt.hlines(mean_error, xmin = 0, xmax = np.max(plaque_areas_label), colors = "c")
    plt.hlines(mean_error + 1.96 * std_error, xmin = 0, xmax = np.max(plaque_areas_label), colors="r", linestyles="dashed")
    plt.hlines(mean_error - 1.96 * std_error, xmin = 0, xmax = np.max(plaque_areas_label), colors="r", linestyles="dashed")
    plt.title('area_error')
    plt.xlabel('plaque area avg')
    plt.ylabel('area_error (pixels)')
    plt.savefig(note + "/area_error.png")
    plt.show()

    plt.figure()
    plt.bar(range(len(areaerror_mean_hist)), areaerror_mean_hist, alpha=1, width=0.8, color='blue', lw=3)
    plt.xticks([i-1 for i in range(len(areaerror_std_hist))], area_range)
    plt.title('average area error')
    plt.xlabel('plaque area')
    plt.ylabel('area_error (pixels)')
    plt.savefig(note + "/mean_error_hist.png")
    plt.show()


    plt.figure()
    plt.bar(range(len(areaerror_std_hist)), areaerror_std_hist, alpha=1, width=0.8, color='blue', lw=3)
    plt.xticks([i-1 for i in range(len(areaerror_std_hist))], area_range)
    plt.title('standard deviation area error')
    plt.xlabel('plaque area')
    plt.ylabel('area_error (pixels)')
    plt.savefig(note + "/std_error_hist.png")
    plt.show()