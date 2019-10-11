from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    return tf.layers.batch_normalization(
        inputs=inputs,
        axis=1 if data_format == 'channels_first' else 3,
        training=training,
    )


def conv2d(inputs, filters, kernel_size, strides, data_format):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='SAME',
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format,
    )


def simpnet_block(inputs, filters, should_pool=False, should_dropout=False, kernel_size=3, dropout_ratio=0.2, kernel_initializer=tf.variance_scaling_initializer, activation='elu', data_format='channels_last', training=True):
    features = conv2d(inputs, filters, kernel_size, 1, data_format)
    features = batch_norm(features, training, data_format)
    if activation == 'elu':
        features = tf.nn.relu(features)
    else:
        raise ValueError('bad activation {}'.format(activation))
    if should_pool:
        features = tf.layers.max_pooling2d(features, pool_size=2, strides=2)
    if should_dropout:
        features = tf.layers.dropout(features, rate=dropout_ratio, training=training)
    return features


def build_simpnet_model(inputs, num_classes, hparams, training):
    data_format = 'channels_last'
    
    features = inputs
    features = simpnet_block(inputs=features, filters=66, training=training)

    features = simpnet_block(inputs=features, filters=128, training=training)
    features = simpnet_block(inputs=features, filters=128, training=training)
    features = simpnet_block(inputs=features, filters=128, training=training)

    features = simpnet_block(inputs=features, filters=192, should_dropout=True, should_pool=True, training=training)

    features = simpnet_block(inputs=features, filters=192, training=training)
    features = simpnet_block(inputs=features, filters=192, training=training)
    features = simpnet_block(inputs=features, filters=192, training=training)
    features = simpnet_block(inputs=features, filters=192, training=training)

    features = simpnet_block(inputs=features, filters=288, should_dropout=True, should_pool=True, dropout_ratio=0.3, training=training)

    features = simpnet_block(inputs=features, filters=288, training=training)

    features = simpnet_block(inputs=features, filters=355, training=training)

    features = simpnet_block(inputs=features, filters=432, training=training)

    axes = [2, 3] if data_format == 'channels_first' else [1, 2]
    features = tf.reduce_max(features, axes, keepdims=False)
    features = tf.layers.dropout(features, rate=0.3)
    features = tf.layers.dense(inputs=features, units=num_classes)
    features = tf.identity(features, 'final_dense')
    return features
