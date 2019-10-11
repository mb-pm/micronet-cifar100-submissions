import logging

import argparse
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.layers import (
    Dense,
    MaxPooling2D,
    Flatten,
    Activation,
    BatchNormalization,
    Dropout,
    GlobalMaxPool2D
)
from tensorflow.python.keras.regularizers import l2

from augmenter import Augmenter
from custom_layers import KvantizationLayer, BinaryConv
from utils import load_data, SWA, get_source_home_dir

DECAY_STEP = 0.65
INITIAL_LR = 0.1
EPOCHS_FOR_DECAY = 30
NUM_EPOCHS = 750
BATCH = 128
WEIGHT_DECAY = 1e-3
MOMENTUM=0.9
ACTIVATION = 'elu'

K = tf.keras.backend
logger = logging.getLogger('cnn')
logger.setLevel(logging.INFO)


def simpnet_layer(
        inputs,
        num_filters,
        should_pool=False,
        should_dropout=False,
        dropout_ratio=0.2,
        kernel_initializer='glorot_normal',
        activation=ACTIVATION,
):
    x = inputs
    kvant_l = KvantizationLayer()
    conv = BinaryConv(
        filters=num_filters,
        kernel_initializer=kernel_initializer
    )
    x = kvant_l(x)
    x = conv(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    if should_pool:
        x = MaxPooling2D(
            pool_size=(2, 2),
            strides=2
        )(x)
    if should_dropout:
        x = Dropout(rate=dropout_ratio)(x)
    return x


def build_simpnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # group #1
    x = simpnet_layer(inputs=inputs, num_filters=66)
    x = simpnet_layer(inputs=x, num_filters=128)
    x = simpnet_layer(inputs=x, num_filters=128)
    x = simpnet_layer(inputs=x, num_filters=128)

    x = simpnet_layer(inputs=x, num_filters=128, should_dropout=True, should_pool=True)

    # group #2
    x = simpnet_layer(inputs=x, num_filters=192)
    x = simpnet_layer(inputs=x, num_filters=192)
    x = simpnet_layer(inputs=x, num_filters=192)
    x = simpnet_layer(inputs=x, num_filters=192)

    x = simpnet_layer(inputs=x, num_filters=288, should_dropout=True, should_pool=True, dropout_ratio=0.3)

    x = simpnet_layer(inputs=x, num_filters=288)

    x = simpnet_layer(inputs=x, num_filters=355)

    x = simpnet_layer(inputs=x, num_filters=432)

    x = GlobalMaxPool2D()(x)
    x = Dropout(rate=0.3)(x)
    y = Flatten()(x)
    outputs = Dense(
        num_classes,
        activation='softmax',
        kernel_initializer='glorot_normal',
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(y)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--use_best_augmentations',
        help='Whether the best previously trained augmentations should be loaded',
        type=int,
        default=1
    )

    args = argparser.parse_args()
    use_best_augmentations = args.use_best_augmentations == 1

    num_categories = 100
    model = build_simpnet(input_shape=(32, 32, 3), num_classes=num_categories)

    # Learning rate scheduler
    def scheduler(epoch):
        lr = INITIAL_LR
        decay = DECAY_STEP ** (epoch // EPOCHS_FOR_DECAY)
        return lr * decay

    # fetch training data
    (x_train, y_train), (x_test, y_test) = load_data(should_standardize_train=False, should_smooth=True)

    output_path = get_source_home_dir() / 'outputs/quant_bin_simpnet_plus'
    output_path.mkdir(parents=True, exist_ok=True)

    train_args = dict(
        validation_data=(x_test, y_test),
        verbose=1,
        epochs=NUM_EPOCHS,
        callbacks=[
            LearningRateScheduler(scheduler),
            SWA(str(output_path / 'model'), NUM_EPOCHS - EPOCHS_FOR_DECAY + 1)  # last 29 models
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=INITIAL_LR, momentum=MOMENTUM),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    gen = Augmenter(x=x_train, y=y_train, epochs=NUM_EPOCHS,
                    batch_size=BATCH, use_best_augmentations=use_best_augmentations,
                    standardize_images=True, shuffle=True)
    model.fit_generator(
        gen,
        steps_per_epoch=len(x_train) // BATCH,
        workers=1,
        use_multiprocessing=False,
        **train_args
    )

    score, accuracy = model.evaluate(x_test, y_test, batch_size=BATCH)
    print(f"Accuracy: {accuracy}")