import os
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.utils import to_categorical

_ENVIRONMENT_KEY = 'ENVIRONMENT'
_DOCKER_HOME = Path('/workspace/source')
_REGULAR_HOME = Path('.').resolve()



class LayerInfo:
    # conv_index: int
    def __init__(
            self,
            conv_name: str,
            layer_index: int,
            indices: List[int],
    ):
        self.conv_name = conv_name
        self.layer_index = layer_index
        self.indices = indices


class SWA(Callback):

    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch

    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'.format(self.nb_epoch - self.swa_epoch))

    def on_epoch_end(self, epoch, logs=None):

        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()

        elif epoch > self.swa_epoch:
            for i, layer in enumerate(self.model.layers):
                self.swa_weights[i] = (self.swa_weights[i] * (epoch - self.swa_epoch) + self.model.get_weights()[i]) / (
                            (epoch - self.swa_epoch) + 1)
        else:
            pass

    def get_swa_weights(self):
        return self.swa_weights

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save(self.filepath)
        print('Final stochastic averaged model saved to file.')


class CustomCallback(Callback):

    def __init__(self, log_file_path):
        super().__init__()
        self.max_val_acc = None
        self._log_file_path = log_file_path

    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
        metrics = logs  # Log all metrics

        with open(self._log_file_path, 'a') as f:
            f.write(f"Step: {epoch};")
            for metric in metrics:
                if metric == 'val_acc':
                    self.max_val_acc = metrics[metric] if self.max_val_acc is None or metrics[metric] > self.max_val_acc \
                        else self.max_val_acc
                f.write(f"{metric}: {metrics[metric]};")
                f.write(f"max_val_acc: {self.max_val_acc};")
            f.write("\n")

        print(f" Step: {epoch}, Max Val Acc: {self.max_val_acc}")


def smooth_labels(y, smooth_factor):
    """
    Convert a matrix of one-hot row-vector labels into smoothed versions.

    # Arguments
        y: matrix of one-hot row-vector labels to be smoothed
        smooth_factor: label smoothing factor (between 0 and 1)

    # Returns
        A matrix of smoothed labels.
    """
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    return y


def load_data(should_standardize_train: bool = True, should_smooth: bool = True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    x_test = x_test.astype(np.float32)
    x_train_mean = np.mean(x_train, axis=(0, 1, 2))
    x_train_std = np.std(x_train, axis=(0, 1, 2), ddof=1)

    if should_standardize_train:
        x_train = x_train.astype(np.float32)
        x_train -= x_train_mean
        x_train /= x_train_std

    x_test -= x_train_mean
    x_test /= x_train_std

    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)

    if should_smooth:
        y_train = smooth_labels(y_train, smooth_factor=0.1)

    return (x_train, y_train), (x_test, y_test)


def get_source_home_dir():
    if _ENVIRONMENT_KEY in os.environ:
        return _DOCKER_HOME
    return _REGULAR_HOME


