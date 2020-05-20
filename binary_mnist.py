import autogluon as ag
import mxnet as mx
import mxnet.gluon.nn as nn
from autogluon.contrib.enas import *
from mxnet import gluon, autograd
import numpy as np

BATCH_SIZE = 4
LEARNING_RATE = 0.003
MOMENTUM = 0.9
CTX = mx.gpu(0)
LOG_INTERVAL = 100

@enas_unit()
class QDenseUnit(mx.gluon.HybridBlock):
    def __init__(self, output_features,):
        super().__init__()
        self.qdense = nn.QDense(output_features, bits=1)
    def hybrid_forward(self, F, x):
        return self.qdense(x)

class Convert2dTo1dUnit(mx.gluon.HybridBlock):
    def hybrid_forward(self, F, x):
        #import pdb; pdb.set_trace()
        return x.reshape((0, -1))

@enas_unit()
class QConvUnit(mx.gluon.HybridBlock):
    def __init__(self, num_channels):
        super().__init__()
        self.qconv = nn.QConv2D(channels=num_channels, kernel_size=5, bits=1)
    def hybrid_forward(self, F, x):
        return self.qconv(x)

def simple_net_definition():
    return (
        QConvUnit(16),
        Convert2dTo1dUnit(),
        QDenseUnit(64),
        QDenseUnit(32),
        QDenseUnit(10)
    )

def create_hybrid_sequential(net_definition):
    simple_net = nn.HybridSequential()
    with simple_net.name_scope():
        for layer in net_definition:
            simple_net.add(layer)
    #simple_net.hybridize()

    simple_net.initialize()
    return simple_net

def create_enas_sequential(net_definition):
    return ENAS_Sequential(*net_definition)

def train_fn(epoch, num_epochs, net, batch):
    pass

def transformer(data, label):
    data = data.astype(np.float32) / 255
    data = data.transpose((2,0,1))
    return data, label

def test(net, val_data):
    metric = mx.metric.Accuracy()
    for data, label in val_data:
        data = data.as_in_context(CTX)
        label = label.as_in_context(CTX)
        output = net(data)
        metric.update([label], [output])

    return metric.get()

def train(net, train_data, val_data, epochs):
    # Collect all parameters from net and its children, then initialize them.
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=CTX)
    # Trainer is for updating parameters with gradient.
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': LEARNING_RATE, 'momentum': MOMENTUM})
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(epochs):
        # reset data iterator and metric at begining of epoch.
        metric.reset()
        for i, (data, label) in enumerate(train_data):
            # Copy data to ctx if necessary
            data = data.as_in_context(CTX)
            label = label.as_in_context(CTX)
            # Start recording computation graph with record() section.
            # Recorded graphs can then be differentiated with backward.
            with autograd.record():
                output = net(data)
                L = loss(output, label)
                L.backward()
            # take a gradient step with batch_size equal to data.shape[0]
            trainer.step(data.shape[0])
            # update metric at last.
            metric.update([label], [output])

            if i % LOG_INTERVAL == 0 and i > 0:
                name, acc = metric.get()
                print('[Epoch %d Batch %d] Training: %s=%f'%(epoch, i, name, acc))

        name, acc = metric.get()
        print('[Epoch %d] Training: %s=%f'%(epoch, name, acc))

        name, val_acc = test(net, val_data)
        print('[Epoch %d] Validation: %s=%f'%(epoch, name, val_acc))

    net.export("mnist-{}-bit".format(1), epoch=1)

def train_without_enas():
    train_data = gluon.data.DataLoader(
        gluon.data.vision.MNIST('./data', train=True, transform=transformer),
        batch_size=BATCH_SIZE, shuffle=True, last_batch='discard')

    val_data = gluon.data.DataLoader(
        gluon.data.vision.MNIST('./data', train=False, transform=transformer),
        batch_size=BATCH_SIZE, shuffle=False)

    net = create_hybrid_sequential(simple_net_definition())
    train(net, train_data, val_data, 5)

def train_with_enas():
    simple_binary_enas_net = create_enas_sequential(simple_net_definition())

    simple_binary_enas_net.initialize()

    x = mx.nd.random.uniform(shape=(1, 1, 28, 28))
    y = simple_binary_enas_net.evaluate_latency(x)
    print('Average latency is {:.2f} ms, latency of the current architecture is {:.2f} ms'.format(simple_binary_enas_net.avg_latency,
                                                                                                  simple_binary_enas_net.latency))

    simple_binary_enas_net.nparams
    reward_fn = lambda metric, net: metric * ((net.avg_latency / net.latency) ** 0.1)

    scheduler = ENAS_Scheduler(simple_binary_enas_net, train_set='mnist',
                               reward_fn=reward_fn, batch_size=BATCH_SIZE, num_gpus=1,
                               warmup_epochs=0, epochs=1, controller_lr=3e-3,
                               plot_frequency=10, update_arch_frequency=5)
    scheduler.run()



def main():
    train_without_enas()
    #train_with_enas()






if __name__=="__main__":
    main()