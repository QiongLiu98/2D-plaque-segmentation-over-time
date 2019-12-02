
from keras import backend as K
import tensorflow as tf
import numpy as np


def DSC2D(y_true, y_pred):

    IMT_y_true = y_true
    IMT_y_pred = y_pred
    y_true_f = K.flatten(IMT_y_true)
    y_pred_f = K.flatten(IMT_y_pred)
    # y_pred_f = K.cast(y_pred_f > 0.5, K.floatx())
    intersection = K.sum(y_true_f * y_pred_f)
    DSC = (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return DSC


def DSCLoss(y_true, y_pred):

    myloss = 1 - DSC2D(y_true, y_pred)

    return myloss