import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import (
    Dense,
    MaxPooling2D,
    Conv2D,
    Flatten,
    Activation,
    BatchNormalization,
    Dropout,
    GlobalMaxPool2D
)
from tensorflow.python.keras.regularizers import l2

from custom_layers import KvantizationLayer, BinaryConv

WEIGHT_DECAY = 1e-3
ACTIVATION = 'elu'


def simpnet_layer(
        inputs,
        num_filters,
        should_pool=False,
        should_dropout=False,
        kernel_size=3,
        stride=1,
        dropout_ratio=0.2,
        kernel_initializer='glorot_normal',
        activation=ACTIVATION,
        type='binary',
):
    x = inputs
    if type == 'full':
        conv = Conv2D(
            filters=num_filters,
            kernel_size=kernel_size,
            strides=stride,
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=l2(WEIGHT_DECAY),
            use_bias=False
        )
        x = conv(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
    elif type == 'binary':
        kvant_l = KvantizationLayer()
        conv = BinaryConv(
            filters=num_filters,
            kernel_initializer=kernel_initializer
        )
        x = kvant_l(x)
        x = conv(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
    else:
        raise Exception('Unknown type {}'.format(type))
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
    from functools import partial
    layer_fn = partial(simpnet_layer, type='binary')
    x = layer_fn(inputs=inputs, num_filters=66)
    x = layer_fn(inputs=x, num_filters=128)
    x = layer_fn(inputs=x, num_filters=128)
    x = layer_fn(inputs=x, num_filters=128)

    x = layer_fn(inputs=x, num_filters=192, should_dropout=True, should_pool=True)

    # group #2
    x = layer_fn(inputs=x, num_filters=192)
    x = layer_fn(inputs=x, num_filters=192)
    x = layer_fn(inputs=x, num_filters=192)
    x = layer_fn(inputs=x, num_filters=192)

    x = layer_fn(inputs=x, num_filters=288, should_dropout=True, should_pool=True, dropout_ratio=0.3)

    x = layer_fn(inputs=x, num_filters=288)

    x = layer_fn(inputs=x, num_filters=355)

    x = layer_fn(inputs=x, num_filters=432)

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
