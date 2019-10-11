import math
import random
from functools import partial

import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps
import numpy as np
from tensorflow.python.keras.utils import Sequence

from utils import get_source_home_dir


def float_parameter(level, maxval):
    return float(level) * maxval / 10


def int_parameter(level, maxval):
    return int(level * maxval / 10)


def cutout(image, value):
    value = int_parameter(value, 20)
    crop_size = value
    image_height, image_width = image.size
    x_start = np.random.randint(image_width - crop_size)
    y_start = np.random.randint(image_height - crop_size)
    image_copy = image.copy()
    PIL.ImageDraw.Draw(image_copy).rectangle((x_start, y_start, x_start + crop_size, y_start + crop_size), (0, 0, 0))
    return image_copy


def random_translation(image, value):
    translation_x = np.random.randint(-value, value + 1)
    translation_y = np.random.randint(-value, value + 1)
    return image.transform(image.size, PIL.Image.AFFINE, (1, 0, translation_x, 0, 1, translation_y))


def horizontal_flip(image, value):
    return PIL.ImageOps.mirror(image)


def invert(image, value):
    return PIL.ImageOps.invert(image)


def contrast(image, value):
    value = float_parameter(value, 1.8) + .1
    return PIL.ImageEnhance.Contrast(image).enhance(value)


def auto_contrast(image, value):
    return PIL.ImageOps.autocontrast(image)


def equalize(image, value):
    return PIL.ImageOps.equalize(image)


def posterize(image, value):
    value = int_parameter(value, 4)
    return PIL.ImageOps.posterize(image, value)


def rotate(image, value):
    value = int_parameter(value, 30)
    if np.random.random() <= 0.5:
        value = -value
    return image.rotate(value)


def shear_x(image, value):
    value = float_parameter(value, 0.3)
    if np.random.random() <= 0.5:
        value = -value
    image = image.transform(image.size, PIL.Image.AFFINE, (1, value, -value * image.size[1] / 2, 0, 1, 0))
    return image


def shear_y(image, value):
    value = float_parameter(value, 0.3)
    if np.random.random() <= 0.5:
        value = -value
    image = image.transform(image.size, PIL.Image.AFFINE, (1, 0, 0, value, 1, -value * image.size[1] / 2))
    return image


def translate_x(image, value):
    value = int_parameter(value, 10)
    if np.random.random() < 0.5:
        value = -value
    return image.transform(image.size, PIL.Image.AFFINE, (1, 0, value, 0, 1, 0))


def translate_y(image, value):
    value = int_parameter(value, 10)
    if np.random.random() < 0.5:
        value = -value
    return image.transform(image.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, value))


def color(image, value):
    value = float_parameter(value, 1.8) + .1
    return PIL.ImageEnhance.Color(image).enhance(value)


def brightness(image, value):
    value = float_parameter(value, 1.8) + .1
    return PIL.ImageEnhance.Brightness(image).enhance(value)


def solarize(image, value):
    value = int_parameter(value, 256)
    return PIL.ImageOps.solarize(image, value)


def sharpness(image, value):
    value = float_parameter(value, 1.8) + .1
    return PIL.ImageEnhance.Sharpness(image).enhance(value)


def standardize(image, per_channel_mean, per_channel_std):
    return (np.array(image).astype(np.float32) - per_channel_mean) / per_channel_std


def wrap_image_op_with_probability(probability, image_op):
    return lambda image: (image_op(image), 1) if np.random.uniform() <= probability else (image, 0)


def _load_pba_augmentations(load_best: bool):
    source_home_dir = get_source_home_dir()
    file_name = 'pba_augmentations_best.txt' if load_best else 'pba_augmentations.txt'
    full_path = source_home_dir / file_name
    if not full_path.exists():
        raise ValueError("Augmentations file doesn't exist. Either use best augmentations or train "
                         "new ones")
    with full_path.open() as f:
        op_strings = [eval(l) for l in f]
    schedule_ops = []
    for op_string in op_strings:
        epoch_ops = []
        for single_op in op_string:
            epoch_ops.append(
                wrap_image_op_with_probability(
                    single_op[1],
                    partial(
                        globals()[single_op[0]],
                        value=single_op[2]
                    ),
                )
            )
        schedule_ops.append(epoch_ops)

    return schedule_ops


def _scale_ops_to_num_epochs(ops, epochs):
    scale = epochs / len(ops)
    return [ops[int(math.floor(i / scale))] for i in range(epochs)]


class Augmenter(Sequence):
    def __init__(self, x, y, epochs, batch_size, use_best_augmentations, shuffle=True, standardize_images=True):
        self.__x = x
        self.__y = y
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__len = int(math.ceil(len(x) / batch_size))
        self.__indices = [i for i in range(len(x))]
        self.__augmentation_ops = _scale_ops_to_num_epochs(_load_pba_augmentations(use_best_augmentations), epochs)
        self.__baseline_augment_op = lambda image: random_translation(
            wrap_image_op_with_probability(
                0.5, partial(horizontal_flip, value=None)
            )(image)[0], value=4)
        self.__standardize_op = partial(
            standardize,
            per_channel_mean=np.mean(x, axis=(0, 1, 2)) if standardize_images else np.array((0, 0, 0)),
            per_channel_std=np.std(x, axis=(0, 1, 2), ddof=1) if standardize_images else np.array((1, 1, 1,))
        )
        self.__cutout_op = wrap_image_op_with_probability(0.5, partial(cutout, value=4))
        self.__epoch = 0

    def _apply_augmentation(self, xi, augmentations):
        xi = self.__baseline_augment_op(PIL.Image.fromarray(xi))
        count = np.random.choice((0, 1, 2), p=(0.2, 0.3, 0.5))
        for _ in range(count):
            augmentation = random.choice(augmentations)
            xi, done_augmentations = augmentation(xi)
        xi = self.__cutout_op(xi)[0]
        xi = self.__standardize_op(xi)
        return xi

    def _apply_augmentations(self, x, augmentations):
        return np.array([self._apply_augmentation(xi, augmentations) for xi in x])

    def __len__(self):
        return self.__len

    def __getitem__(self, idx):
        indices = self.__indices[idx * self.__batch_size:(idx + 1) * self.__batch_size]
        y = self.__y[indices]
        x = self.__x[indices]
        augmentations = self.__augmentation_ops[self.__epoch]
        x = self._apply_augmentations(x, augmentations)
        return x, y

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.__indices)
        self.__epoch += 1