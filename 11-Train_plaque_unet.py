
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
from model_2DUNet import my2DUNet
from load_data_plaque import *
from LossMetrics import *


if __name__ == '__main__':

    batchsize = 64
    steps = 300
    stage = 4
    filters = 64
    datapath = './npydata_M_test_1127'

    note = 'UNet'

    lognoteDir = './log1127_m_lr4'
    if os.path.exists(lognoteDir) is True:
        print('Exist %s' % lognoteDir)
    else:
        os.mkdir(lognoteDir)

    imgs_train, imgs_mask_train, imgs_test, test_mask = load_npydata(datapath)

    train_data = imgs_train#np.concatenate((imgs_train, gradientimages), axis=-1)
    test_data = imgs_test#np.concatenate((imgs_test, gradienttests), axis=-1)

    # mynet = VNet(img_rows, img_cols, 1)
    mynet = my2DUNet(None, None, 1)
    # mynet = FCN(img_rows, img_cols)
    #
    # model = mynet.model(classes=1, kernel_size=(5, 5))
    model = mynet.model(classes=1, stages = stage, base_filters = filters, activation_name='sigmoid', deconvolution = False)
    # model = mynet.FCN_Vgg16_8s(weight_decay=0.001, classes=1, activate='sigmoid')

    model.compile(optimizer=adam(lr=1e-4), loss='binary_crossentropy', metrics=[DSC2D, 'accuracy'])


    def scheduler(epoch):
        if epoch == 200:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1) )
        if epoch == 700:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(model.optimizer.lr)


    reduce_lr = LearningRateScheduler(scheduler)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=50, epsilon=0.001, min_lr=1e-6)
    # model.load_weights("100_3DUNet.hdf5")
    model_checkpoint = ModelCheckpoint(os.path.join(lognoteDir, 'plaque_'+note+'.hdf5'), monitor='val_DSC2D', mode='max', verbose=1, save_best_only=True)

    visualization = TensorBoard(log_dir=os.path.join(lognoteDir, 'Graph_' + note), histogram_freq=0, write_graph=True, write_images=True)
    print('Fitting model...')

    # model.load_weights(os.path.join(lognoteDir, 'pre_plaque_UNet_fs64_stg4.hdf5'))

    hist = model.fit(train_data, imgs_mask_train, batch_size=batchsize, epochs=steps, verbose=1,
                     validation_data=[test_data, test_mask], shuffle=True, callbacks=[model_checkpoint, visualization])

    # hist = model.fit_generator(generate_arrays_from_file(25, './CAINdata--CCA_multi.csv', (img_rows, img_cols, img_depth)),
    #                            steps_per_epoch=25, nb_epoch=5,
    #                            validation_data=generate_arrays_from_file(25, './CAINdata--CCA_multi_validation.csv', (img_rows, img_cols, img_depth)),
    #                            validation_steps=13,
    #                            shuffle=True, callbacks=[model_checkpoint])

    with open(os.path.join(lognoteDir, 'log_'+note+'.txt'), 'w') as f:
        f.write(str(hist.history))





