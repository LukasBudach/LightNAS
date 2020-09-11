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


from __future__ import division

import math
import sys
from functools import reduce
from operator import mul
import time

import mxnet as mx
import numpy as np
import tqdm as tqdm
from mxnet import gluon

from image_classification import get_data_iters, get_model
from util.arg_parser import get_parser, set_dummy_training_args

from mxnet.gluon.data.vision import transforms


import mxnet.ndarray as nd

all_blocks_on_json = '/home/padl20t4/LightNAS/trainings/meliusnet22_enas_2020_8_11_17_24/exported_models/inference_only/model98-symbol.json'
all_blocks_on_params = '/home/padl20t4/LightNAS/trainings/meliusnet22_enas_2020_8_11_17_24/exported_models/inference_only/model98-0098.params'

penalized_latency_json = '/home/padl20t4/LightNAS/trainings/meliusnet22_enas_2020_7_19_22_17/exported_models/inference_only/model0-symbol.json'
penalized_latency_params  = '/home/padl20t4/LightNAS/trainings/meliusnet22_enas_2020_7_19_22_17/exported_models/inference_only/model0-0000.params'

new_json = '/home/padl20t4/LightNAS/trainings/resnet18_v1_enas_2020_8_31_18_51/exported_models/inference_only/model99-symbol.json'
new_params = '/home/padl20t4/LightNAS/trainings/resnet18_v1_enas_2020_8_31_18_51/exported_models/inference_only/model99-0099.params'

json_filepath = new_json
params_filepath = new_params

ctx = mx.gpu(int(0))
print("ctx:",ctx)

### load the model
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    deserialized_net = gluon.nn.SymbolBlock.imports(json_filepath, ['data'], params_filepath, ctx=ctx)

mock_data = nd.zeros((1, 3, 32, 32))


out = deserialized_net(mock_data.as_in_context(ctx))
predictions = nd.argmax(out, axis=1)
print('mock data predictions:',predictions)

### load the data

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

val_data = gluon.data.DataLoader(gluon.data.vision.CIFAR100(train=False,fine_label=True).transform_first(transform_test),batch_size=16, shuffle=False, num_workers=4)


num_correct = 0
num_wrong = 0
total_time = time.time()


for data, label in val_data:

    data = mx.ndarray.cast(data=data, dtype='float32', out=None, name=None)
    result = deserialized_net(data.as_in_context(ctx))

    # print("Results:",result[0][15])

    # result = nd.argmax(result, axis=1)
    # result = mx.ndarray.cast(data=result, dtype='int', out=None, name=None)
    # print('Model predictions: ', result.asnumpy(), 'Label:', label)

#     break

    probabilities = result.softmax().asnumpy()
    ground_truth = label.asnumpy()

    predictions = np.argmax(probabilities, axis=1)
    likeliness = np.max(probabilities, axis=1)


    num_correct += np.sum(predictions == ground_truth)
    num_wrong += np.sum(predictions != ground_truth)

print("Correct: {:d}, Wrong: {:d}".format(num_correct, num_wrong))
print("Accuracy: {:.2f}%".format(100 * num_correct / (num_correct + num_wrong)))
print("Total time:", time.time()-total_time)