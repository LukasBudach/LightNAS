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
from binary_models.basenet_dense import *

__all__ = ['DenseNet', 'DenseNetParameters',
           'densenet_flex', 'densenet28', 'densenet37', 'densenet45']


class DenseNet(BaseNetDense):
    def _add_base_block_structure(self, dilation):
        self._add_dense_block(dilation)


class DenseNetParameters(BaseNetDenseParameters):
    def __init__(self):
        super(DenseNetParameters, self).__init__('DenseNet')

    def _is_it_this_model(self, model):
        return model.startswith('densenet')


# Specification
# block_config, reduction_factor, downsampling
densenet_spec = {
    None:  (None,           [1 / 2,   1 / 2,   1 / 2],   DOWNSAMPLE_STRUCT),
    '28':  ([6, 6, 6, 5],   [1 / 2.7, 1 / 2.7, 1 / 2.2], DOWNSAMPLE_STRUCT),
    '37':  ([6, 8, 12, 6],  [1 / 3.3, 1 / 3.3, 1 / 4],   DOWNSAMPLE_STRUCT),
    '45':  ([6, 12, 14, 8], [1 / 2.7, 1 / 3.3, 1 / 4],   DOWNSAMPLE_STRUCT),
}


# Constructor
get_densenet = get_basenet_constructor(densenet_spec, DenseNet)


def densenet_flex(**kwargs):
    return get_densenet(None, **kwargs)


def densenet28(**kwargs):
    return get_densenet('28', **kwargs)


def densenet37(**kwargs):
    return get_densenet('37', **kwargs)


def densenet45(**kwargs):
    return get_densenet('45', **kwargs)
