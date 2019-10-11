# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module defines an API for counting parameters and operations.


## Defining the Operation Count API
- `input_size` is an int, since square image assumed.
- `strides` is a tuple, but assumed to have same stride in both dimensions.
- Supported `paddings` are `same' and `valid`.
- `use_bias` is boolean.
- `activation` is one of the following `relu`, `swish`, `sigmoid`, None
- kernel_shapes for `Conv2D` dimensions must be in the following order:
  `k_size, k_size, c_in, c_out`
- kernel_shapes for `FullyConnected` dimensions must be in the following order:
  `c_in, c_out`
- kernel_shapes for `DepthWiseConv2D` dimensions must be like the following:
  `k_size, k_size, c_in==c_out, 1`

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from custom_layers import QuantBinaryConv, BinaryConv, KvantizationLayer

"""Operation definition for 2D convolution.

Attributes:
  input_size: int, Dimensions of the input image (square assumed).
  kernel_shape: list, of length 4. Shape of the convolutional kernel.
  strides: list, of length 2. Stride with which the kernel is applied.
  padding: str, padding added to the input image.
  use_bias: bool, if true a bias term is added to the output.
  activation: str, type of activation applied to the output.
"""
# pylint: disable=pointless-string-statement

Input = tf.keras.layers.InputLayer
Lambda = tf.keras.layers.Lambda
Concatenate = tf.keras.layers.Concatenate

Conv2D = tf.keras.layers.Conv2D

BatchNorm = tf.keras.layers.BatchNormalization

Activation = tf.keras.layers.Activation

Add = tf.keras.layers.Add
Multiply = tf.keras.layers.Multiply

Dropout = tf.keras.layers.Dropout
Flatten = tf.keras.layers.Flatten

"""Operation definition for 2D depthwise convolution.

Only difference compared to Conv2D is the kernel_shape[3] = 1.
"""
DepthWiseConv2D = tf.keras.layers.DepthwiseConv2D

"""Operation definition for Global Max Pooling.

Attributes:
  input_size: int, Dimensions of the input image (square assumed).
  n_channels: int, Number of output dimensions.
"""
GlobalMax = tf.keras.layers.GlobalMaxPool2D

"""Operation definition for Max Pooling.

Attributes:
  input_size: int, Dimensions of the input image (square assumed).
  kernel_shape: list, of length 2. Shape of pooling kernel.
  strides: list, of length 2. Stride with which the kernel is applied.
  n_channels: int, Number of output dimensions.
"""
MaxPool = tf.keras.layers.MaxPooling2D

"""Operation definition for Global Average Pooling.

Attributes:
  input_size: int, Dimensions of the input image (square assumed).
  n_channels: int, Number of output dimensions.
"""
GlobalAvg = tf.keras.layers.GlobalAvgPool2D

"""Operation definitions for elementwise multiplication and addition.

Attributes:
  kernel_shape: list, of length 2. Shape of the weight matrix.
  use_bias: bool, if true a bias term is added to the output.
  activation: str, type of activation applied to the output.
"""
Dense = tf.keras.layers.Dense


def get_sparse_size(tensor_shape, param_bits, sparsity):
    """Given a tensor shape returns #bits required to store the tensor sparse.

    If sparsity is greater than 0, we do have to store a bit mask to represent
    sparsity.
    Args:
      tensor_shape: list<int>, shape of the tensor
      param_bits: int, number of bits the elements of the tensor represented in.
      sparsity: float, sparsity level. 0 means dense.
    Returns:
      int, number of bits required to represented the tensor in sparse format.
    """
    n_elements = np.prod(tensor_shape)
    c_size = n_elements * param_bits * (1 - sparsity)
    if sparsity > 0:
        c_size += n_elements  # 1 bit binary mask.
    return c_size


def get_conv_output_size(image_size, filter_size, padding, stride):
    """Calculates the output size of convolution.

    The input, filter and the strides are assumed to be square.
    Arguments:
      image_size: int, Dimensions of the input image (square assumed).
      filter_size: int, Dimensions of the kernel (square assumed).
      padding: str, padding added to the input image. 'same' or 'valid'
      stride: int, stride with which the kernel is applied (square assumed).
    Returns:
      int, output size.
    """
    if padding == 'same':
        pad = filter_size // 2
    elif padding == 'valid':
        pad = 0
    else:
        raise NotImplementedError('Padding: %s should be `same` or `valid`.'
                                  % padding)
    out_size = np.ceil((image_size - filter_size + 1. + 2 * pad) / stride)
    return int(out_size)


def count_ops(op, sparsity, param_bits):
    """Given a operation class returns the flop and parameter statistics.

    Args:
      op: namedtuple, operation definition.
      sparsity: float, sparsity of parameterized operations. Sparsity only effects
        Conv and FC layers; since activations are dense.
      param_bits: int, number of bits required to represent a parameter.
    Returns:
      param_count: number of bits required to store parameters
      n_mults: number of multiplications made per input sample.
      n_adds: number of multiplications made per input sample.
    """
    flop_mults = flop_adds = param_count = 0
    if isinstance(op, Input):
        pass
    elif isinstance(op, Lambda):
        pass
    elif isinstance(op, Concatenate):
        pass
    elif isinstance(op, KvantizationLayer):
        operations = np.prod(op.input_shape[1:])
        flop_mults += 10 * operations
        flop_adds += 2 * operations

        assert param_bits == 32
        param_count += 3 * param_bits  # 32bit parameters
    elif isinstance(op, QuantBinaryConv):
        # Square kernel expected.
        assert op.kernel.shape[0].value == op.kernel.shape[1].value
        k_size, _, c_in, c_out = [d.value for d in op.kernel.shape]

        # Square stride expected.
        assert op.strides[0] == op.strides[1]
        stride = op.strides[0]

        # Each application of the kernel can be thought as a dot product between
        # the flattened kernel and patches of the image.
        vector_length = (k_size * k_size * c_in)
        # Number of elements in the output is OUT_SIZE * OUT_SIZE * OUT_CHANNEL
        # print("COUNTING PADDING: ", op.padding)
        assert op.padding.upper() == "SAME"
        assert op.input_shape[1] == op.input_shape[2]
        n_output_elements = get_conv_output_size(op.input_shape[1], k_size, op.padding,
                                                 stride) ** 2 * c_out
        # Each output is the product of a one dot product. Dot product of two
        # vectors of size n needs n multiplications and n - 1 additions.
        assert param_bits == 32

        # binary weights only, multiplication is only a 1bit operation
        flop_mults += (vector_length * n_output_elements) * (1 / param_bits)
        flop_adds += (vector_length - 1) * n_output_elements / 2  # inputs are in 16 bits

        param_count += np.prod([s.value for s in op.weights[0].shape])  # 1 bit parameters

    elif isinstance(op, BinaryConv):
        # Square kernel expected.
        assert op.kernel.shape[0].value == op.kernel.shape[1].value
        k_size, _, c_in, c_out = [d.value for d in op.kernel.shape]

        # Square stride expected.
        assert op.strides[0] == op.strides[1]
        stride = op.strides[0]

        # Each application of the kernel can be thought as a dot product between
        # the flattened kernel and patches of the image.
        vector_length = (k_size * k_size * c_in)
        # Number of elements in the output is OUT_SIZE * OUT_SIZE * OUT_CHANNEL
        # print("COUNTING PADDING: ", op.padding)
        assert op.padding.upper() == "SAME"
        assert op.input_shape[1] == op.input_shape[2]
        n_output_elements = get_conv_output_size(op.input_shape[1], k_size, op.padding,
                                                 stride) ** 2 * c_out
        # Each output is the product of a one dot product. Dot product of two
        # vectors of size n needs n multiplications and n - 1 additions.
        assert param_bits == 32

        # binary weights only, multiplication is only a 1bit operation
        flop_mults += (vector_length * n_output_elements) * (1 / param_bits)
        flop_adds += (vector_length - 1) * n_output_elements * 8 / 32  # inputs are in 8 bits

        param_count += np.prod([s.value for s in op.weights[0].shape])  # 1 bit parameters

    elif isinstance(op, DepthWiseConv2D):
        assert op.depthwise_kernel.shape[0].value == op.depthwise_kernel.shape[1].value
        k_size, _, _, _ = [d.value for d in op.depthwise_kernel.shape]
        c_in = op.input_shape[-1]
        c_out = op.input_shape[-1] * op.depth_multiplier

        # Size of the possibly sparse convolutional tensor.
        # param_count += get_sparse_size(
        #     [k_size, k_size, c_in, c_out], param_bits, sparsity)

        # Square stride expected.
        assert op.strides[0] == op.strides[1]
        stride = op.strides[0]

        # Each application of the kernel can be thought as a dot product between
        # the flattened kernel and patches of the image.
        vector_length = (k_size * k_size)
        # Number of elements in the output is OUT_SIZE * OUT_SIZE * OUT_CHANNEL
        assert op.padding.upper() == "SAME"
        assert op.input_shape[1] == op.input_shape[2]
        n_output_elements = get_conv_output_size(op.input_shape[1], k_size, op.padding,
                                                 stride) ** 2 * c_out

        # Each output is the product of a one dot product. Dot product of two
        # vectors of size n needs n multiplications and n - 1 additions.

        flop_mults += vector_length * n_output_elements
        flop_adds += (vector_length - 1) * n_output_elements

        assert param_bits == 32
        param_count += np.prod([s.value for s in op.weights[0].shape]) * param_bits  # 32bit parameters

        if op.use_bias:
            # For each output channel we need a bias term.
            param_count += c_out * param_bits
            # If we have bias we need one more addition per dot product.
            flop_adds += n_output_elements

        # activation is added as a separate layer
        assert op.activation == tf.keras.activations.linear

    elif isinstance(op, Conv2D):
        # Square kernel expected.
        assert op.kernel.shape[0].value == op.kernel.shape[1].value
        k_size, _, c_in, c_out = [d.value for d in op.kernel.shape]

        # Size of the possibly sparse convolutional tensor.
        # param_count += get_sparse_size(
        #     [k_size, k_size, c_in, c_out], param_bits, sparsity)

        # Square stride expected.
        assert op.strides[0] == op.strides[1]
        stride = op.strides[0]

        # Each application of the kernel can be thought as a dot product between
        # the flattened kernel and patches of the image.
        vector_length = (k_size * k_size * c_in)
        # Number of elements in the output is OUT_SIZE * OUT_SIZE * OUT_CHANNEL
        assert op.input_shape[1] == op.input_shape[2]
        assert op.padding.upper() == "SAME"
        n_output_elements = get_conv_output_size(op.input_shape[1], k_size, op.padding,
                                                 stride) ** 2 * c_out

        # Each output is the product of a one dot product. Dot product of two
        # vectors of size n needs n multiplications and n - 1 additions.

        flop_mults += vector_length * n_output_elements
        flop_adds += (vector_length - 1) * n_output_elements

        assert param_bits == 32
        param_count += np.prod([s.value for s in op.weights[0].shape]) * param_bits  # 32bit parameters

        if op.use_bias:
            # For each output channel we need a bias term.
            param_count += c_out * param_bits
            # If we have bias we need one more addition per dot product.
            flop_adds += n_output_elements

        # activation is added as a separate layer
        assert op.activation == tf.keras.activations.linear

    elif isinstance(op, BatchNorm):
        # batchnorm linear
        # we assume BN is transfored as Ax+B
        operations = np.prod(op.input_shape[1:])
        flop_mults += operations
        flop_adds += operations

        param_size = op.weights[0].shape[0].value
        assert len(set([w.shape[0].value for w in op.weights])) == 1

        # mean, var, alpha and beta are packed as A and B
        assert param_bits == 32
        param_count += 2 * param_size * param_bits  # 32bit parameters

    elif isinstance(op, MaxPool):
        output_shape = op.output_shape[1:]
        feature_map_size = np.prod(output_shape)
        pool_size = op.pool_size[1]

        assert op.pool_size[0] == op.pool_size[1]
        flop_mults += feature_map_size * (pool_size ** 2 - 1)

        # # for each output element, we must make kernel_size ** 2 * num_in_channels - 1 comparisons
        # n_output_elements = get_conv_output_size(op.input_size, op.kernel.shape[0].value, 'valid', op.strides[0])
        # flop_mults += (op.kernel.shape[0].value ** 2 * op.n_channels - 1) * n_output_elements

    elif isinstance(op, Activation):
        if op.activation == tf.keras.activations.relu:
            # For each output channel we will make a max.
            feature_map_size = np.prod(op.input_shape[1:])
            flop_mults += 1 * feature_map_size  # 1 * ili 3 *?

        elif op.activation == tf.keras.activations.elu:
            # x < 0 or x >= 0 -> 1 op
            # x >= 0: x -> 0 extra ops
            # x < 0: alpha(exp(x)-1) ==> exp(x) - 1 -> 1 extra mul and 1 extra add
            # But alpha is equal to 1 so we can disregard it
            feature_map_size = np.prod(op.input_shape[1:])
            flop_adds += 1 * feature_map_size
            flop_mults += 2 * feature_map_size

        elif op.activation == tf.keras.activations.tanh:
            feature_map_size = np.prod(op.input_shape[1:])
            flop_mults += feature_map_size

        elif op.activation == tf.keras.activations.sigmoid:
            feature_map_size = np.prod(op.input_shape[1:])
            flop_mults += feature_map_size

        else:
            raise ValueError('Encountered unknown Activation %s.' % str(op.activation))

    elif isinstance(op, Add):
        # Number of elements many additions.
        assert op.input_shape[0] == op.input_shape[1]

        feature_map_size = np.prod(op.input_shape[0][1:])
        flop_adds += feature_map_size

    elif isinstance(op, Multiply):
        # Number of elements many additions.
        assert op.input_shape[0] == op.input_shape[1]

        feature_map_size = np.prod(op.input_shape[0][1:])
        flop_adds += feature_map_size

    elif isinstance(op, GlobalMax):
        w, h, cin = op.input_shape[1:]
        assert w == h
        # For each output channel we will make comparison (mul op
        flop_mults += cin * (w * h - 1)

    elif isinstance(op, GlobalAvg):
        # For each output channel we will make a division.
        w, h, cin = op.input_shape[1:]

        flop_mults += cin
        # We have to add values over spatial dimensions.
        flop_adds += (w * h - 1) * cin
    elif isinstance(op, Dropout):
        # weights in the following dense layer can be scaled
        pass
    elif isinstance(op, Flatten):
        # weights in the following dense layer can be scaled
        pass
    elif isinstance(op, Dense):
        c_in, c_out = [s.value for s in op.kernel.shape]

        flop_mults += c_out * c_in
        flop_adds += c_out * (c_in - 1)

        backup_p = param_count
        assert param_bits == 32
        param_count += np.prod([s.value for s in op.weights[0].shape]) * param_bits  # 32bit parameters

        added = param_count - backup_p
        assert c_in * c_out * param_bits == added

        if op.use_bias:
            assert param_bits
            param_count += c_out * param_bits
            flop_adds += c_out

        if op.activation:
            assert op.activation == tf.keras.activations.softmax
            flop_mults += (c_out - 1)
    else:
        raise ValueError('Encountered unknown operation %s.' % str(op))
    return param_count, flop_mults, flop_adds


# Info
def get_info(op):
    """Given an op extracts some common information."""
    input_size, kernel_size, in_channels, out_channels = [-1] * 4
    if isinstance(op, Input):
        input_shape = op.input_shape[0]
        in_channels = input_shape[-1]
        out_channels = input_shape[-1]
        input_size = input_shape[1]
    elif isinstance(op, Lambda):
        pass
    elif isinstance(op, Concatenate):
        pass
    elif isinstance(op, DepthWiseConv2D):
        # square kernel assumed.
        input_shape = op.input_shape
        in_channels = input_shape[-1]
        out_channels = input_shape[-1] * op.depth_multiplier
        input_size = input_shape[1]
    elif isinstance(op, Conv2D):
        # square kernel assumed.
        kernel_size, _, in_channels, out_channels = [s.value for s in op.kernel.shape]
        input_size = op.input_shape[1]
    elif isinstance(op, KvantizationLayer):
        in_channels = op.input_shape[-1]
        out_channels = op.input_shape[-1]
        input_size = op.input_shape[1]
    elif isinstance(op, BinaryConv) or isinstance(op, QuantBinaryConv):
        kernel_size, _, in_channels, out_channels = [s.value for s in op.kernel.shape]
        input_size = op.input_shape[1]
    elif isinstance(op, BatchNorm):
        in_channels = op.input_shape[-1]
        out_channels = op.input_shape[-1]
        input_size = op.input_shape[1]
    elif isinstance(op, Activation):
        in_channels = op.input_shape[-1]
        out_channels = op.input_shape[-1]
        input_size = op.input_shape[1]
    elif isinstance(op, (Add)):
        in_channels = op.input_shape[0][-1]
        out_channels = op.input_shape[0][-1]
        input_size = op.input_shape[0][1]
    elif isinstance(op, Multiply):
        assert op.input_shape[0] == op.input_shape[1]
        in_channels = op.input_shape[0][-1]
        out_channels = op.input_shape[0][-1]
        input_size = op.input_shape[0][1]
    elif isinstance(op, GlobalAvg):
        in_channels = op.input_shape[-1]
        out_channels = op.input_shape[-1]
        input_size = op.input_shape[1]
    elif isinstance(op, MaxPool):
        in_channels = op.input_shape[-1]
        out_channels = op.input_shape[-1]
        input_size = op.input_shape[1]
    elif isinstance(op, Flatten):
        in_channels = op.input_shape[-1]
        out_channels = np.prod(op.input_shape[1:])
        input_size = out_channels
    elif isinstance(op, Dropout):
        in_channels = op.input_shape[-1]
        out_channels = op.input_shape[-1]
        input_size = op.input_shape[1]
    elif isinstance(op, GlobalMax):
        in_channels = op.input_shape[-1]
        out_channels = op.input_shape[-1]
        input_size = op.input_shape[1]
    elif isinstance(op, Dense):
        in_channels, out_channels = [s.value for s in op.kernel.shape]
        input_size = 1
    else:
        # print('Encountered unknown operation %s. Skipping.' % str(op))
        raise ValueError('Encountered unknown operation %s.' % str(op))
    return input_size, kernel_size, in_channels, out_channels


class MicroNetCounter(object):
    """Counts operations using given information.

    """
    _header_str = '{:25} {:>10} {:>13} {:>13} {:>13} {:>15} {:>10} {:>10} {:>10}'
    _line_str = ('{:25s} {:10d} {:13d} {:13d} {:13d} {:15.3f} {:10.3f}'
                 ' {:10.3f} {:10.3f}')

    def __init__(self, all_ops, add_bits_base=32, mul_bits_base=32):
        self.all_ops = all_ops
        # Full precision add is counted one.
        self.add_bits_base = add_bits_base
        # Full precision multiply is counted one.
        self.mul_bits_base = mul_bits_base

    def _aggregate_list(self, counts):
        return np.array(counts).sum(axis=0)

    def process_counts(self, total_params, total_mults, total_adds,
                       mul_bits, add_bits):
        # converting to Mbytes.
        total_params = int(total_params) / 8. / 1e6
        total_mults = total_mults * mul_bits / self.mul_bits_base / 1e6
        total_adds = total_adds * add_bits / self.add_bits_base / 1e6
        return total_params, total_mults, total_adds

    def _print_header(self):
        output_string = self._header_str.format(
            'op_name', 'inp_size', 'kernel_size', 'in channels', 'out channels',
            'params(MBytes)', 'mults(M)', 'adds(M)', 'MFLOPS')
        print(output_string)
        print(''.join(['='] * 125))

    def _print_line(self, name, input_size, kernel_size, in_channels,
                    out_channels, param_count, flop_mults, flop_adds, mul_bits,
                    add_bits, base_str=None):
        """Prints a single line of operation counts."""
        op_pc, op_mu, op_ad = self.process_counts(param_count, flop_mults,
                                                  flop_adds, mul_bits, add_bits)
        if base_str is None:
            base_str = self._line_str
        output_string = base_str.format(
            name, input_size, kernel_size, in_channels, out_channels, op_pc,
            op_mu, op_ad, op_mu + op_ad)
        print(output_string)
        return (name, input_size, kernel_size, in_channels, out_channels, op_pc,
                op_mu, op_ad, op_mu + op_ad)

    def print_summary(self, sparsity, param_bits, add_bits, mul_bits,
                      summarize_blocks=True):
        """Prints all operations with given options.

        Args:
          sparsity: float, between 0,1 defines how sparse each parametric layer is.
          param_bits: int, bits in which parameters are stored.
          add_bits: float, number of bits used for accumulator.
          mul_bits: float, number of bits inputs represented for multiplication.
          summarize_blocks: bool, if True counts within a block are aggregated and
            reported in a single line.

        """
        self._print_header()
        # Let's count starting from zero.
        total_params, total_mults, total_adds = [0] * 3
        for op_template in self.all_ops:
            op_name = type(op_template).__name__
            if op_name.startswith('block'):
                if not summarize_blocks:
                    # If debug print the ops inside a block.
                    for block_op_name, block_op_template in op_template:
                        param_count, flop_mults, flop_adds = count_ops(block_op_template,
                                                                       sparsity, param_bits)
                        temp_res = get_info(block_op_template)
                        input_size, kernel_size, in_channels, out_channels = temp_res
                        self._print_line('%s_%s' % (op_name, block_op_name), input_size,
                                         kernel_size, in_channels, out_channels,
                                         param_count, flop_mults, flop_adds, mul_bits,
                                         add_bits)
                # Count and sum all ops within a block.
                param_count, flop_mults, flop_adds = self._aggregate_list(
                    [count_ops(template, sparsity, param_bits)
                     for _, template in op_template])
                # Let's extract the input_size and in_channels from the first operation.
                input_size, _, in_channels, _ = get_info(op_template[0][1])
                # Since we don't know what is inside a block we don't know the following
                # fields.
                kernel_size = out_channels = -1
            else:
                # If it is a single operation just count.
                param_count, flop_mults, flop_adds = count_ops(op_template, sparsity,
                                                               param_bits)
                temp_res = get_info(op_template)
                input_size, kernel_size, in_channels, out_channels = temp_res
            # At this point param_count, flop_mults, flop_adds should be read.
            total_params += param_count
            total_mults += flop_mults
            total_adds += flop_adds
            # Print the operation.
            self._print_line(op_name, input_size, kernel_size, in_channels,
                             out_channels, param_count, flop_mults, flop_adds,
                             mul_bits, add_bits)

        # Print Total values.
        # New string since we are passing empty strings instead of integers.
        out_str = ('{:25s} {:10s} {:13s} {:13s} {:13s} {:15.3f} {:10.3f} {:10.3f} '
                   '{:10.3f}')
        return self._print_line(
            'total', '', '', '', '', total_params, total_mults, total_adds,
            mul_bits, add_bits, base_str=out_str)
