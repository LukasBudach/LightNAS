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
"""ResNets for ENAS, implemented in Gluon."""
from __future__ import division

from binary_models.common_layers import add_initial_layers

__all__ = ['ResNetV1Enas', 'ResNetV2Enas',
           'BasicBlockV1Enas', 'BasicBlockV2Enas',
           'BottleneckV1Enas', 'BottleneckV2Enas',
           'resnet18_v1_enas', 'resnet34_v1_enas', 'resnet50_v1_enas', 'resnet101_v1_enas', 'resnet152_v1_enas',
           'resnet18_v2_enas', 'resnet34_v2_enas', 'resnet50_v2_enas', 'resnet101_v2_enas', 'resnet152_v2_enas',
           'get_resnet_enas']

import os

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import base

import autogluon as ag
from autogluon.contrib.enas import *

# Helpers
def _conv3x3(bits, channels, stride, in_channels):
    return nn.QConv2D(channels, bits=bits, kernel_size=3,
                      strides=stride, padding=1, in_channels=in_channels)


# Blocks
@enas_unit(bits=ag.space.Categorical(1,2,4,8,16,32), share_parameters=True)
class BasicBlockV1Enas(HybridBlock):
    r"""BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """

    def __init__(self, channels, stride, downsample=False, in_channels=0, init=True, bits=1, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.stride = stride
        self.in_channels = in_channels
        self.bits = bits

        self.body = nn.HybridSequential(prefix='')
        self.downsample = None
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
        if init:
            self._init()

    def _init(self):
        self.body.add(nn.activated_conv(self.channels, kernel_size=3, stride=self.stride, padding=1,
                                        in_channels=self.in_channels, bits=self.bits))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.activated_conv(self.channels, kernel_size=3, stride=1, padding=1,
                                        in_channels=self.channels, bits=self.bits))
        self.body.add(nn.BatchNorm())

        if self.downsample is not None:
            self.downsample.add(nn.activated_conv(self.channels, kernel_size=1, stride=self.stride, padding=0,
                                                  in_channels=self.in_channels, prefix="sc_qconv_", bits=self.bits))
            self.downsample.add(nn.BatchNorm())

    def hybrid_forward(self, F, x):
        residual = x
        if self.downsample:
            residual = self.downsample(x)
        x = self.body(x)
        # usually activation here, but it is now at start of each unit
        return residual + x

@enas_unit(bits=ag.space.Categorical(1,2,4,8,16,32), share_parameters=True)
class BottleneckV1Enas(HybridBlock):
    r"""Bottleneck V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 50, 101, 152 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """

    def __init__(self, channels, stride,downsample=False, in_channels=0, bits=1, **kwargs):
        super().__init__(**kwargs)
        self.bits = bits
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.QConv2D(channels // 4, kernel_size=1, strides=stride, bits=self.bits))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels // 4, 1, channels // 4, bits=self.bits))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.QConv2D(channels, kernel_size=1, strides=1, bits=self.bits))
        self.body.add(nn.BatchNorm())
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.QConv2D(channels, kernel_size=1, strides=stride,
                                           use_bias=False, in_channels=in_channels, bits=self.bits))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(x + residual, act_type='relu')
        return x

@enas_unit(bits=ag.space.Categorical(1,2,4,8,16,32), share_parameters=True)
class BasicBlockV2Enas(HybridBlock):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """

    def __init__(self, channels, stride, downsample=False, in_channels=0, init=True, bits=1, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.stride = stride
        self.bits = bits
        self.in_channels = in_channels

        self.bn = nn.BatchNorm()
        self.body = nn.HybridSequential(prefix='')
        self.downsample = None
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')

        if init:
            self._init()

    def _init(self):
        self.body.add(nn.activated_conv(self.channels, kernel_size=3, stride=self.stride, padding=1,
                                        in_channels=self.in_channels, bits=self.bits))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.activated_conv(self.channels, kernel_size=3, stride=1, padding=1, 
                                        in_channels=self.channels, bits=self.bits))

        if self.downsample is not None:
            self.downsample.add(nn.activated_conv(self.channels, kernel_size=1, stride=self.stride, padding=0,
                                                  in_channels=self.in_channels, prefix="sc_qconv_", bits=self.bits))

    def hybrid_forward(self, F, x):
        bn = self.bn(x)
        if self.downsample:
            residual = self.downsample(bn)
        else:
            residual = x
        x = self.body(bn)
        return residual + x

@enas_unit(bits=ag.space.Categorical(1,2,4,8,16,32), share_parameters=True)
class BottleneckV2Enas(HybridBlock):
    r"""Bottleneck V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 50, 101, 152 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """

    def __init__(self, channels, stride, downsample=False, in_channels=0, bits=1, **kwargs):
        super().__init__(**kwargs)
        self.bits = bits
        self.bn1 = nn.BatchNorm()
        self.conv1 = nn.QConv2D(channels // 4, kernel_size=1, strides=1, use_bias=False, bits=self.bits)
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv3x3(channels // 4, stride, channels // 4, bits=self.bits)
        self.bn3 = nn.BatchNorm()
        self.conv3 = nn.QConv2D(channels, kernel_size=1, strides=1, use_bias=False, bits=self.bits)
        if downsample:
            self.downsample = nn.QConv2D(channels, 1, stride, use_bias=False,
                                         in_channels=in_channels, bits=self.bits)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        x = self.bn3(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv3(x)

        return x + residual


class ResNetEnas():
    def __init__(self, channels, classes, **kwargs):
        super().__init__(**kwargs)
        self.features = []
        self.output = nn.Dense(classes, in_units=channels[-1])

    r"""Helper methods which are equal for both resnets"""
    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0, **kwargs):
        layers = []
        layers.append(block(channels, stride, channels != in_channels, in_channels=in_channels, prefix='', **kwargs))
        for _ in range(layers - 1):
            layers.append(block(channels, 1, False, in_channels=channels, prefix='', **kwargs))
        return layers

    @property
    def enas_sequential(self):
        enas_sequential = []
        enas_sequential.extend(self.features)
        enas_sequential.extend(self.output)
        return enas_sequential


# Nets
class ResNetV1Enas(ResNetEnas):
    r"""ResNet V1 model from
    `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    initial_layers : bool, default imagenet
        Configure the initial layers.
    """

    def __init__(self, block, layers, channels, classes=1000, initial_layers="imagenet", **kwargs):
        super().__init__(channels, classes, **kwargs)
        assert len(layers) == len(channels) - 1

        self.features.append(nn.BatchNorm(scale=False, epsilon=2e-5))
        self.initial_layers = nn.HybridSequential()
        add_initial_layers(initial_layers, self.initial_layers, channels[0])
        self.features.append(self.initial_layers)
        self.features.append(nn.BatchNorm())

        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            self.features.append(
                self._make_layer(block, num_layer, channels[i + 1], stride, i + 1, in_channels=channels[i]))

        self.features.append(nn.Activation('relu'))
        self.features.append(nn.GlobalAvgPool2D())
        self.features.append(nn.Flatten())


class ResNetV2Enas(ResNetEnas):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    initial_layers : bool, default imagenet
        Configure the initial layers.
    """

    def __init__(self, block, layers, channels, classes=1000, initial_layers="imagenet", **kwargs):
        super().__init__(channels, classes, **kwargs)
        assert len(layers) == len(channels) - 1

        # self.features.add(nn.BatchNorm(scale=False, center=False))
        self.features.append(nn.BatchNorm(scale=False, epsilon=2e-5))
        self.initial_layers = nn.HybridSequential()
        add_initial_layers(initial_layers, self.initial_layers , channels[0])
        self.features.append(self.initial_layers)

        in_channels = channels[0]
        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            self.features.append(
                self._make_layer(block, num_layer, channels[i + 1], stride, i + 1, in_channels=in_channels))
            in_channels = channels[i + 1]

        # fix_gamma=False missing ?
        self.features.append(nn.BatchNorm())
        self.features.append(nn.Activation('relu'))
        self.features.append(nn.GlobalAvgPool2D())
        self.features.append(nn.Flatten())


# Specification
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

resnet_net_versions = [ResNetV1Enas, ResNetV2Enas]
resnet_block_versions = [{'basic_block': BasicBlockV1Enas, 'bottle_neck': BottleneckV1Enas},
                         {'basic_block': BasicBlockV2Enas, 'bottle_neck': BottleneckV2Enas}]


# Constructor
def get_resnet_enas(version, num_layers, pretrained=False, ctx=cpu(),
               root=os.path.join(base.data_dir(), 'models'), **kwargs):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    """
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    assert version >= 1 and version <= 2, \
        "Invalid resnet version: %d. Options are 1 and 2."%version
    resnet_class = resnet_net_versions[version-1]
    block_class = resnet_block_versions[version-1][block_type]
    net = resnet_class(block_class, layers, channels, **kwargs)
    if pretrained:
        raise ValueError("No pretrained model exists, yet.")
        # from ..model_store import get_model_file
        # net.load_parameters(get_model_file('resnet%d_v%d'%(num_layers, version),
        #                                    root=root), ctx=ctx)
    return net

def resnet18_v1_enas(**kwargs):
    r"""ResNet-18 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet_enas(1, 18, **kwargs)

def resnet34_v1_enas(**kwargs):
    r"""ResNet-34 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet_enas(1, 34, **kwargs)

def resnet50_v1_enas(**kwargs):
    r"""ResNet-50 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet_enas(1, 50, **kwargs)

def resnet101_v1_enas(**kwargs):
    r"""ResNet-101 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet_enas(1, 101, **kwargs)

def resnet152_v1_enas(**kwargs):
    r"""ResNet-152 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet_enas(1, 152, **kwargs)

def resnet18_v2_enas(**kwargs):
    r"""ResNet-18 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet_enas(2, 18, **kwargs)

def resnet34_v2_enas(**kwargs):
    r"""ResNet-34 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet_enas(2, 34, **kwargs)

def resnet50_v2_enas(**kwargs):
    r"""ResNet-50 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet_enas(2, 50, **kwargs)

def resnet101_v2_enas(**kwargs):
    r"""ResNet-101 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet_enas(2, 101, **kwargs)

def resnet152_v2_enas(**kwargs):
    r"""ResNet-152 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet_enas(2, 152, **kwargs)
