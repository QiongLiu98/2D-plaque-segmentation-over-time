



#                 Write by Ran Zhou, 2019/04/11
#                 Huazhong University of Science and Technology
#                 Western University

import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, Deconvolution2D,\
    concatenate, GlobalAveragePooling2D, multiply, add, Reshape, Dropout
from keras.optimizers import Adam
from keras import regularizers
import math


class my2DUNet(object):

    def __init__(self, height, width, chs=1):

        self.height = height
        self.width = width
        self.chs = chs

    def model(self, classes=1, base_filters = 32, pool_size=(2, 2), deconvolution=False,
              stages = 4, activation_name="sigmoid", flag_activate=True, flag_drop = True):
        """
        :param depth: depth of input images
        :param height: height of input images
        :param width: width of input images
        :param pool_size: the size of pooling operator
        :param classes: the classes of output mask
        :param base_filters:
        :param deconvolution:
        :param layers:
        :param activation_name:
        :param reg:
        :return:
        """
        channel_dim = -1
        height = self.height
        width = self.width
        chs = self.chs
        input_shape = (height, width, chs)

        # If channels order is "channels first", modify the input shape and channels dimension.
        if K.image_data_format() == "channels_first":
            input_shape = (1, height, width)
            channel_dim = 1

        inputs = Input(shape=input_shape)
        concat_layers = list()
        filter_list = list()
        curr_filters = base_filters
        # Loop over the number of stages (block names).
        x = inputs
        for i in range(stages):

            x = self.myconv2D(x, curr_filters, stride=(1, 1), channel_dim=channel_dim)
            x = self.myconv2D(x, curr_filters, stride=(1, 1), channel_dim=channel_dim)
            if flag_drop and i >= 2:
                x = Dropout(0.5)(x, training=True)

            concat_layers.append(x)
            filter_list.append(curr_filters)
            curr_filters = curr_filters * 2

            # x = self.myconv2D(x, curr_filters, stride=(2, 2), channel_dim=channel_dim)

            x = MaxPooling2D(pool_size=pool_size, padding='same')(x)

        x = self.myconv2D(x, curr_filters, stride=(1, 1), channel_dim=channel_dim)
        x = self.myconv2D(x, curr_filters, stride=(1, 1), channel_dim=channel_dim)
        if flag_drop:
            x = Dropout(0.5)(x, training=True)

        #add layers with up-convolution or up-sampling
        for i in range(stages):
            curr_filters = filter_list.pop()
            y = concat_layers.pop()

            x = self.upconvolution(x, pool_size=pool_size, deconvolution=deconvolution,
                                   filters=curr_filters)
            x = concatenate([x, y], axis=channel_dim)

            x = self.myconv2D(x, curr_filters, stride=(1, 1), channel_dim=channel_dim)
            x = self.myconv2D(x, curr_filters, stride=(1, 1), channel_dim=channel_dim)

        featuremap = x

        if flag_activate:
            x = Conv2D(classes, (1, 1), activation=activation_name)(featuremap)
        else:
            x = Conv2D(classes, (1, 1))(featuremap)

        model = Model(inputs=inputs, outputs=x)

        #visualization of the architecture of the network
        model.summary()

        return model
    def myconv2D(self, x, filters, stride, channel_dim = -1, reg=1e-4, bn_eps=2e-5, bn_moment=0.9):

        x = Conv2D(int(filters), (3, 3), strides=stride, padding="same", use_bias=False,
                       kernel_regularizer=regularizers.l2(reg))(x)
        x = BatchNormalization(axis=channel_dim, epsilon=bn_eps, momentum=bn_moment)(x)
        x = Activation("relu")(x)

        return x

    def upconvolution(self, x, filters, pool_size, kernel_size=(2, 2), strides=(2, 2),
                           reg=1e-4, deconvolution=False):
        if deconvolution:
            return Deconvolution2D(filters=filters, kernel_size=kernel_size, padding='same',
                                   strides=strides)(x)
        else:
            # return UpSampling2D(size=pool_size)(x)
            return Conv2D(filters, kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',
                          kernel_regularizer=regularizers.l2(reg))(UpSampling2D(size = pool_size)(x))




