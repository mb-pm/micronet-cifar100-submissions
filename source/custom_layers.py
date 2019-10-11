import tensorflow as tf
from tensorflow.python.keras.layers import (
    Layer
)
from tensorflow.python.keras.regularizers import l2


@tf.custom_gradient
def sign(x):
    return (tf.cast(x > 0, tf.float32) * 2 - 1), lambda dy: dy


def binarize_weights(weights):
    return sign(tf.nn.tanh(weights))


class KvantizationLayer(Layer):

    def build(self, input_shape):
        self.range_bottom = self.add_weight(
            name='range_min_kurwa_xo',
            shape=(),
            initializer=tf.constant_initializer(-3),
            trainable=True,
            dtype=tf.float32
        )

        self.range_top = self.add_weight(
            name='range_max_kurwa_xo',
            shape=(),
            initializer=tf.constant_initializer(3),
            trainable=True,
            dtype=tf.float32
        )

    def call(self, inputs, **kwargs):

        return tf.quantization.fake_quant_with_min_max_vars(
            inputs,
            self.range_bottom,
            self.range_top,
            num_bits=8
        )


class BinaryConv(Layer):
    def __init__(self, filters, kernel_initializer, kernel=3, *args, **kwargs):
        super(BinaryConv, self).__init__(*args, **kwargs)
        self.__filters = filters
        self.__kernel = kernel
        self.__kernel_initializer = kernel_initializer
        self._weights = None
        self._input_shape = None

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        self._input_shape = input_shape
        self._weights = binarize_weights(self.add_weight(
            shape=(self.__kernel, self.__kernel, in_channels, self.__filters),
            name='kernel',
            initializer=self.__kernel_initializer,
            trainable=True,
            regularizer=l2(1e-3)
        ))

    def call(self, inputs):
        return tf.nn.conv2d(
            input=inputs,
            filter=self._weights,
            strides=(1, 1, 1, 1),
            padding='SAME'
        )

    def get_config(self):
        return dict(
            **super().get_config(),
            filters=self.__filters,
            kernel=self.__kernel,
            kernel_initializer=self.__kernel_initializer
        )

    @property
    def kernel(self):
        return self._weights

    @property
    def strides(self):
        return (1, 1, 1, 1)

    @property
    def padding(self):
        return "same"

    @property
    def input_shape(self):
        return self._input_shape.as_list()


class QuantBinaryConv(BinaryConv):
    """
    This Layer will be used during the evaluation for "fake" quantization to 16 bits of inputs to Conv Layer.
    However, the model is trained with regular BinaryConv (see above).

    """

    def call(self, inputs):
        return super().call(inputs=tf.cast(tf.cast(inputs, tf.float16), tf.float32))
