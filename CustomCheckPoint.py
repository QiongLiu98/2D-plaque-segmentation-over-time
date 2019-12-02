import numpy as np
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, Callback
from LossMetrics import *



def strrfind(str, sub):
    pos = len(str) + 1
    poslist = list()
    while True:
        pos = str.rfind(sub, 0, pos - 1)
        if 0 == pos:
            break
        poslist.append(pos)
    return poslist

class CustomModelCheckpoint(Callback):

    def __init__(self, savemodel, path, monitor='val_loss', mode='min'):
        self.monitor = monitor
        self.model = savemodel
        self.path = path
        self.best = np.inf

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        monitor_val = logs[self.monitor]
        if (epoch % 50) == 0:
            pos = strrfind(self.path, ".")
            self.model.save_weights("%s_%d_%s" % (self.path[0:pos[-1]], epoch, self.path[pos[-1]:]), overwrite=True)
        if self.monitor_op(monitor_val, self.best):
            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                  ' saving model to %s'
                  % (epoch + 1, self.monitor, self.best,
                     monitor_val, self.path))
            self.model.save_weights(self.path, overwrite=True)
            self.best = monitor_val
        else:
            print('\nEpoch %05d: %s did not improve from %0.5f' %
                  (epoch + 1, self.monitor, self.best))
