from bmxnet_examples.binary_models.meliusnet import meliusnet22
from models.meliusnet_enas import meliusnet22_enas
from autogluon import ImageClassification as task
import gluoncv as gcv
import autogluon as ag
from autogluon.contrib.enas import *
from mxnet.gluon.data.vision.datasets import CIFAR10
from config import DATASET_PATH
from mxnet import nd
import mxnet as mx
import mxnet.gluon.nn as nn


class Identity(mx.gluon.HybridBlock):
    def hybrid_forward(self, F, x):
        return x


class ConvBNReLU(mx.gluon.HybridBlock):
    def __init__(self, in_channels, channels, kernel, stride):
        super().__init__()
        padding = (kernel - 1) // 2
        self.conv = nn.Conv2D(channels, kernel, stride, padding, in_channels=in_channels)
        self.bn = nn.BatchNorm(in_channels=channels)
        self.relu = nn.Activation('relu')
    def hybrid_forward(self, F, x):
        return self.relu(self.bn(self.conv(x)))


@enas_unit()
class ResUnit(mx.gluon.HybridBlock):
    def __init__(self, in_channels, channels, hidden_channels, kernel, stride):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, hidden_channels, kernel, stride)
        self.conv2 = ConvBNReLU(hidden_channels, channels, kernel, 1)
        if in_channels == channels and stride == 1:
            self.shortcut = Identity()
        else:
            self.shortcut = nn.Conv2D(channels, 1, stride, in_channels=in_channels)

    def hybrid_forward(self, F, x):
        return self.conv2(self.conv1(x)) + self.shortcut(x)


def get_dataset():
    dataset = task.Dataset(name='FashionMNIST')
    test_dataset = task.Dataset(name='FashionMNIST', train=False)
    return dataset, test_dataset


def train_alexnet(epochs):
    net = gcv.model_zoo.AlexNet()
    train, test = get_dataset()
    task.fit(dataset=train, epochs=epochs, ngpus_per_trial=1, verbose=True, plot_results=True, visualizer='mxboard',
             output_directory='data/')


def train_meliusnet(epochs):
    net = meliusnet22_enas()

    #dataset = CIFAR10(DATASET_PATH+'/cifar10')
    #net.initialize()
    #t = nd.ones((2, 3, 32, 32))
    #net(t)
    train, test = get_dataset()
    task.fit(dataset=train, epochs=epochs, ngpus_per_trial=1, verbose=True, plot_results=True, visualizer='mxboard',
             output_directory='data/', batch_size=128, net=ag.space.Categorical(net), num_trials=1)

def train_meliusnet_enas(epochs):
    mynet = meliusnet22_enas().enas_sequential

    mynet.initialize()
    x = mx.nd.random.uniform(shape=(1, 3, 32, 32))
    mynet(x)
    y = mynet.evaluate_latency(x)
    print('Average latency is {:.2f} ms, latency of the current architecture is {:.2f} ms'.format(mynet.avg_latency,
                                                                                                  mynet.latency))
    mynet.nparams

    scheduler = ENAS_Scheduler(mynet, train_set='cifar10', batch_size=32, num_gpus=1,
                               warmup_epochs=0, epochs=epochs, controller_lr=3e-3,
                               plot_frequency=10, update_arch_frequency=5)
    scheduler.run()


def main():
    # train_meliusnet(2)
    # train_alexnet(10)
    train_meliusnet_enas(2)


if __name__=="__main__":
    main()