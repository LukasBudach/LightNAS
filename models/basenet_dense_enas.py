
import logging
import collections

import mxnet as mx
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import HybridConcurrent, Identity

from bmxnet_examples.binary_models.common_layers import ChannelShuffle, add_initial_layers

import autogluon as ag
from autogluon.contrib.enas import *

from bmxnet_examples.binary_models.model_parameters import ModelParameters


@enas_unit(replace_by_skip_connection=ag.space.Categorical(False))
class DenseBlockEnas(HybridBlock):

    def __init__(self, growth_rate, dilation, bn_size, dropout, replace_by_skip_connection=False, **kwargs):
        super().__init__(**kwargs)
        self.growth_rate = growth_rate
        self.dilation = dilation
        self.bn_size = bn_size
        self.dropout = dropout
        new_feature_computation = nn.HybridSequential(prefix='')
        self.replace_by_skip_connection = replace_by_skip_connection
        if self.replace_by_skip_connection:
            self._add_conv_block(new_feature_computation,
                                 nn.activated_conv(self.growth_rate, kernel_size=1, padding=0))
        else:
            if self.bn_size == 0:
                # no bottleneck
                self._add_conv_block(new_feature_computation,
                                    nn.activated_conv(self.growth_rate, kernel_size=3, padding=dilation,
                                                      dilation=dilation))
            else:
                # bottleneck design
                self._add_conv_block(new_feature_computation,
                                nn.activated_conv(self.bn_size * self.growth_rate, kernel_size=1))
                self._add_conv_block(new_feature_computation,
                                nn.activated_conv(self.growth_rate, kernel_size=3, padding=1))
        dense_block = HybridConcurrent(axis=1, prefix='')
        dense_block.add(Identity())
        dense_block.add(new_feature_computation)
        self.dense_block = dense_block

    def _add_conv_block(self, hybrid_sequential, layer):
        hybrid_sequential.add(nn.BatchNorm())
        hybrid_sequential.add(layer)
        if self.dropout:
            hybrid_sequential.add(nn.Dropout(self.dropout))

    def hybrid_forward(self, F, x):
        return self.dense_block(x)



# Net
class BaseNetDenseEnas(HybridBlock):
    r"""Densenet-BC model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    num_init_features : int
        Number of filters to learn in the first convolution layer.
    growth_rate : int
        Number of filters to add each layer (`k` in the paper).
    block_config : list of int
        List of integers for numbers of layers in each pooling block.
    bn_size : int, default 4
        Multiplicative factor for number of bottle neck layers.
        (i.e. bn_size * k features in the bottleneck layer)
    dropout : float, default 0
        Rate of dropout after each dense layer.
    classes : int, default 1000
        Number of classification classes.
    initial_layers : bool, default imagenet
        Configure the initial layers.
    """

    def __init__(self, num_init_features, growth_rate, block_config, reduction, bn_size, downsample,
                 initial_layers="imagenet", dropout=0, classes=1000, dilated=False, **kwargs):
        super().__init__(**kwargs)
        self.num_blocks = len(block_config)
        self.dilation = (1, 1, 2, 4) if dilated else (1, 1, 1, 1)
        self.downsample_struct = downsample
        self.bn_size = bn_size
        self.growth_rate = growth_rate
        self.dropout = dropout
        self.reduction_rates = reduction

        module_list = []

        #with self.name_scope():
        self.features = nn.HybridSequential(prefix='')
        add_initial_layers(initial_layers, self.features, num_init_features)
        module_list.append(self.features)
        # Add dense blocks
        self.num_features = num_init_features
        for i, repeat_num in enumerate(block_config):
            module_list.extend(self._make_repeated_base_blocks(repeat_num, i))
            if i != len(block_config) - 1:
                transition = self._make_transition(i)
                module_list.append(transition)
        self.finalize = nn.HybridSequential(prefix='')
        self.finalize.add(nn.BatchNorm())
        self.finalize.add(nn.Activation('relu'))
        if dilated:
           self.finalize.add(nn.AvgPool2D(pool_size=28))
        else:
           self.finalize.add(nn.AvgPool2D(pool_size=4 if initial_layers == "thumbnail" else 7))
        self.finalize.add(nn.Flatten())
        module_list.append(self.finalize)

        self.output = nn.Dense(classes)
        module_list.append(self.output)

        self.enas_sequential = ENAS_Sequential(module_list)
        self.hybrid_sequential = nn.HybridSequential()
        for module in module_list:
            self.hybrid_sequential.add(module)
        self.hybrid_sequential.initialize()

    def hybrid_forward(self, F, x):
        return self.hybrid_sequential(x)




    def get_layer(self, num):
        name = "layer{}".format(num)
        if not hasattr(self, name):
            setattr(self, name, nn.HybridSequential(prefix=''))
        return getattr(self, name)

    def _add_base_block_structure(self, dilation):
        raise NotImplementedError()

    def _make_repeated_base_blocks(self, repeat_num, stage_index):
        module_list = []
        dilation = self.dilation[stage_index]
        self.current_stage = nn.HybridSequential(prefix='stage{}_'.format(stage_index + 1))
        with self.current_stage.name_scope():
            for _ in range(repeat_num):
                module_list.extend(self._add_base_block_structure(dilation))
        self.get_layer(stage_index).add(self.current_stage)
        return module_list

    def _add_dense_block(self, dilation):
        dense_block = DenseBlockEnas(self.growth_rate, dilation, self.bn_size, self.dropout)
        self.num_features += self.growth_rate
        self.current_stage.add(dense_block)
        return dense_block

    def _make_transition(self, transition_num):
        dilation = self.dilation[transition_num + 1]
        num_out_features = self.num_features // self.reduction_rates[transition_num]
        num_out_features = int(round(num_out_features / 32)) * 32
        logging.info("Features in transition {}: {} -> {}".format(
            transition_num + 1, self.num_features, num_out_features
        ))
        self.num_features = num_out_features

        transition = nn.HybridSequential(prefix='')
        with transition.name_scope():
            for layer in self.downsample_struct.split(","):
                if layer == "bn":
                    transition.add(nn.BatchNorm())
                elif layer == "relu":
                    transition.add(nn.Activation("relu"))
                elif layer == "q_conv":
                    transition.add(nn.activated_conv(self.num_features, kernel_size=1))
                elif "fp_conv" in layer:
                    groups = 1
                    if ":" in layer:
                        groups = int(layer.split(":")[1])
                    transition.add(nn.Conv2D(self.num_features, kernel_size=1, groups=groups, use_bias=False))
                elif layer == "pool" and dilation == 1:
                    transition.add(nn.AvgPool2D(pool_size=2, strides=2))
                elif layer == "max_pool" and dilation == 1:
                    transition.add(nn.MaxPool2D(pool_size=2, strides=2))
                elif "cs" in layer:
                    groups = 16
                    if ":" in layer:
                        groups = int(layer.split(":")[1])
                    transition.add(ChannelShuffle(groups=groups))

        self.get_layer(transition_num + 1).add(transition)
        return transition