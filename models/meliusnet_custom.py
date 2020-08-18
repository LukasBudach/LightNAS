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

"""MeliusNet constructed from config string, implemented in Gluon."""
import warnings
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import HybridConcurrent, Identity
from mxnet.gluon import HybridBlock

from binary_models.basenet_dense import get_basenet_constructor, DOWNSAMPLE_STRUCT
from binary_models.meliusnet import MeliusNet, ImprovementBlock, MeliusNetParameters
from binary_models.common_layers import add_initial_layers


__all__ = ['MeliusNetCustom', 'meliusnet_custom', 'MeliusNetCustomParameters',
           'meliusnet_custom', 'meliusnet22_custom', 'meliusnet29_custom',
           'meliusnet42_custom', 'meliusnet59_custom', 'meliusnet_a_custom', 'meliusnet_b_custom', 'meliusnet_c_custom'
           ]

class MeliusNetCustom(MeliusNet):

    def __init__(self, num_init_features, growth_rate, block_config, reduction, bn_size, downsample,
                 initial_layers="imagenet", dropout=0, classes=1000, dilated=False,
                config_string="DIDIDIDITDIDIDIDIDITDIDIDIDITDIDIDIDI", **kwargs):
        HybridBlock.__init__(self,**kwargs)
        self.num_blocks = len(block_config)
        self.dilation = (1, 1, 2, 4) if dilated else (1, 1, 1, 1)
        self.downsample_struct = downsample
        self.bn_size = bn_size
        self.growth_rate = growth_rate
        self.dropout = dropout
        self.reduction_rates = reduction
        self.config_string=config_string

        if block_config is not [-1,-1,-1,-1]:
            warnings.warn("Attention, the MeliusNetCustom block_config constructor parameter is not [-1,-1,-1,-1]."
                         " This parameter only exists for backward compatibility but isn't used anymore"
                         " because the configuration is read from the config_string. Make sure you understand how the"
                         " MeliusNetCustom class should be used.")

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            add_initial_layers(initial_layers, self.features, num_init_features)
            # Add dense blocks
            self.num_features = num_init_features
            if self.config_string.count("T") != 3:
                raise Exception("config_string must contain exactly 3 tansition layers")
            self.meliusnet_block_configs = self.config_string.split('T')
            for i,block_string in enumerate(self.meliusnet_block_configs):
                self._make_repeated_base_blocks(block_string, i)
                if i != len(block_config) - 1:
                    self._make_transition(i)
            self.finalize = nn.HybridSequential(prefix='')
            self.finalize.add(nn.BatchNorm())
            self.finalize.add(nn.Activation('relu'))
            if dilated:
                self.finalize.add(nn.AvgPool2D(pool_size=28))
            else:
                self.finalize.add(nn.AvgPool2D(pool_size=4 if initial_layers == "thumbnail" else 7))
            self.finalize.add(nn.Flatten())

            self.output = nn.Dense(classes)

    def _make_repeated_base_blocks(self, block_string, stage_index):
        dilation = self.dilation[stage_index]
        self.current_stage = nn.HybridSequential(prefix='stage{}_'.format(stage_index + 1))
        with self.current_stage.name_scope():
            for char in block_string:
                if char=="D":
                    self._add_dense_block(dilation, is_skip_block=False)
                if char=="S":
                    self._add_dense_block(dilation, is_skip_block=True)
                if char=="I":
                    self.current_stage.add(
                        ImprovementBlock(self.growth_rate, self.num_features, dilation=dilation, prefix='')
                    )

        self.get_layer(stage_index).add(self.current_stage)

    def _add_dense_block(self, dilation, is_skip_block=False):
        new_features = nn.HybridSequential(prefix='')

        def _add_conv_block(layer):
            new_features.add(nn.BatchNorm())
            new_features.add(layer)
            if self.dropout:
                new_features.add(nn.Dropout(self.dropout))

        if is_skip_block:
            _add_conv_block(nn.activated_conv(self.growth_rate, kernel_size=1, padding=dilation, dilation=dilation))
        else:
            if self.bn_size == 0:
                # no bottleneck
                _add_conv_block(nn.activated_conv(self.growth_rate, kernel_size=3, padding=dilation, dilation=dilation))
            else:
                # bottleneck design
                _add_conv_block(nn.activated_conv(self.bn_size * self.growth_rate, kernel_size=1))
                _add_conv_block(nn.activated_conv(self.growth_rate, kernel_size=3, padding=1))

        self.num_features += self.growth_rate

        dense_block = HybridConcurrent(axis=1, prefix='')
        dense_block.add(Identity())
        dense_block.add(new_features)
        self.current_stage.add(dense_block)

class MeliusNetCustomParameters(MeliusNetParameters):
    def __init__(self):
        super().__init__()

    def _is_it_this_model(self, model):
        return model.startswith('meliusnet') and '_custom' in model

    def _add_arguments(self, parser):
        parser.add_argument('--config-string', type=str, default=None,
                            help='Block config to build the custom meliusnet from '
                                 '(e.g. DIDIDIDITDIDIDIDIDITDIDIDIDITDIDIDIDI')

    def _map_opt_to_kwargs(self, opt, kwargs):
        kwargs['config_string'] = opt.config_string

# Specification
meliusnet_custom_spec = {
    # we just pass dummy values to block_config so that the right number of transition layers can be inferred
    # name:         block_config,     reduction_factors,                  downsampling
    '_custom':      ([-1,-1,-1,-1],   [1 / 2,     1 / 2,     1 / 2],     DOWNSAMPLE_STRUCT),
    '22_custom':   ([-1,-1,-1,-1],   [160 / 320, 224 / 480, 256 / 480],  DOWNSAMPLE_STRUCT),
    '29_custom':   ([-1,-1,-1,-1],   [128 / 320, 192 / 512, 256 / 704],  DOWNSAMPLE_STRUCT),
    '42_custom':   ([-1,-1,-1,-1],   [160 / 384, 256 / 672, 416 / 1152], DOWNSAMPLE_STRUCT),
    '59_custom':   ([-1,-1,-1,-1],   [192 / 448, 320 / 960, 544 / 1856], DOWNSAMPLE_STRUCT),
    'a_custom':    ([-1,-1,-1,-1],   [160 / 320, 256 / 480, 288 / 576],  DOWNSAMPLE_STRUCT.replace('fp_conv', 'cs,fp_conv:4')),
    'b_custom':    ([-1,-1,-1,-1],   [160 / 320, 224 / 544, 320 / 736],  DOWNSAMPLE_STRUCT.replace('fp_conv', 'cs,fp_conv:2')),
    'c_custom':    ([-1,-1,-1,-1],   [128 / 256, 192 / 448, 288 / 832],  DOWNSAMPLE_STRUCT.replace('fp_conv', 'cs,fp_conv:4')),
}

# Constructor
get_meliusnet_custom = get_basenet_constructor(meliusnet_custom_spec, MeliusNetCustom)

def meliusnet_custom(**kwargs):
    return get_meliusnet_custom('_custom', **kwargs)


def meliusnet22_custom(**kwargs):
    return get_meliusnet_custom('22_custom', **kwargs)


def meliusnet29_custom(**kwargs):
    return get_meliusnet_custom('29_custom', **kwargs)


def meliusnet42_custom(**kwargs):
    return get_meliusnet_custom('42_custom', **kwargs)


def meliusnet59_custom(**kwargs):
    return get_meliusnet_custom('59_custom', **kwargs)


def meliusnet_a_custom(**kwargs):
    return get_meliusnet_custom('a_custom', **kwargs)


def meliusnet_b_custom(**kwargs):
    return get_meliusnet_custom('b_custom', **kwargs)


def meliusnet_c_custom(**kwargs):
    return get_meliusnet_custom('c_custom', **kwargs)

