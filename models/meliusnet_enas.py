# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ
"""DenseNet, implemented in Gluon."""
from mxnet.gluon import HybridBlock, nn
from autogluon.contrib.enas import *

from bmxnet_examples.binary_models.basenet_dense import *

from models.basenet_dense_enas import BaseNetDenseEnas

import autogluon as ag


__all__ = ['MeliusNetEnas', 'MeliusNetEnasParameters', 'ImprovementBlockEnas',
           'meliusnet_flex_enas', 'meliusnet22_enas', 'meliusnet29_enas', 'meliusnet42_enas', 'meliusnet59_enas',
           'meliusnet_a_enas', 'meliusnet_b_enas', 'meliusnet_c_enas']

# Blocks
@enas_unit(replace_by_skip_connection=ag.space.Categorical(True, False))
class ImprovementBlockEnas(HybridBlock):
    r"""ImprovementBlock which improves the last n channels"""

    def __init__(self, channels, in_channels, replace_by_skip_connection=False, dilation=1, **kwargs):
        super().__init__(**kwargs)
        self.replace_by_skip_connection = replace_by_skip_connection
        if(not replace_by_skip_connection):
            self.body = nn.HybridSequential(prefix='')
            self.body.add(nn.BatchNorm())
            self.body.add(nn.activated_conv(channels=channels, kernel_size=3, stride=1,
                                            padding=dilation, in_channels=in_channels, dilation=dilation))


            self.use_sliced_addition = channels != in_channels
            if self.use_sliced_addition:
                assert channels < in_channels
                self.slices = [0, in_channels - channels, in_channels]
                self.slices_add_x = [False, True]

    def hybrid_forward(self, F, x):
        if(self.replace_by_skip_connection):
            return x
        residual = x
        x = self.body(x)
        if not self.use_sliced_addition:
            return x + residual

        parts = []
        for add_x, slice_begin, slice_end in zip(self.slices_add_x, self.slices[:-1], self.slices[1:]):
            if slice_end - slice_begin == 0:
                continue
            result = F.slice_axis(residual, axis=1, begin=slice_begin, end=slice_end)
            if add_x:
                result = result + x
            parts.append(result)
        return F.concat(*parts, dim=1)

class MeliusNetEnas(BaseNetDenseEnas):
    def _add_base_block_structure(self, dilation):
        module_list = []
        module_list.append(self._add_dense_block(dilation))
        improvement_block = ImprovementBlockEnas(self.growth_rate, self.num_features, dilation=dilation, prefix='')
        self.current_stage.add(improvement_block)
        module_list.append(improvement_block)
        return module_list

# Specification
meliusnet_enas_spec = {
    # name: block_config,     reduction_factors,                  downsampling
    None:   (None,            [1 / 2,     1 / 2,     1 / 2],      DOWNSAMPLE_STRUCT),
    '23':   ([2, 4, 6, 6],    [128 / 192, 192 / 384, 288 / 576],  DOWNSAMPLE_STRUCT.replace('fp_conv', 'cs,fp_conv:8')),
    '22':   ([4, 5, 4, 4],    [160 / 320, 224 / 480, 256 / 480],  DOWNSAMPLE_STRUCT),
    '29':   ([4, 6, 8, 6],    [128 / 320, 192 / 512, 256 / 704],  DOWNSAMPLE_STRUCT),
    '42':   ([5, 8, 14, 10],  [160 / 384, 256 / 672, 416 / 1152], DOWNSAMPLE_STRUCT),
    '59':   ([6, 12, 24, 12], [192 / 448, 320 / 960, 544 / 1856], DOWNSAMPLE_STRUCT),
    'a':    ([4, 5, 5, 6],    [160 / 320, 256 / 480, 288 / 576],  DOWNSAMPLE_STRUCT.replace('fp_conv', 'cs,fp_conv:4')),
    'b':    ([4, 6, 8, 6],    [160 / 320, 224 / 544, 320 / 736],  DOWNSAMPLE_STRUCT.replace('fp_conv', 'cs,fp_conv:2')),
    'c':    ([3, 5, 10, 6],   [128 / 256, 192 / 448, 288 / 832],  DOWNSAMPLE_STRUCT.replace('fp_conv', 'cs,fp_conv:4')),
}

# Constructor
get_meliusnet_enas = get_basenet_constructor(meliusnet_enas_spec, MeliusNetEnas)


def meliusnet_flex_enas(**kwargs):
    return get_meliusnet_enas(None, **kwargs)


def meliusnet22_enas(**kwargs):
    return get_meliusnet_enas('22', **kwargs)


def meliusnet29_enas(**kwargs):
    return get_meliusnet_enas('29', **kwargs)


def meliusnet42_enas(**kwargs):
    return get_meliusnet_enas('42', **kwargs)


def meliusnet59_enas(**kwargs):
    return get_meliusnet_enas('59', **kwargs)


def meliusnet_a_enas(**kwargs):
    return get_meliusnet_enas('a', **kwargs)


def meliusnet_b_enas(**kwargs):
    return get_meliusnet_enas('b', **kwargs)


def meliusnet_c_enas(**kwargs):
    return get_meliusnet_enas('c', **kwargs)
