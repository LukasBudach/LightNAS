import autogluon as ag
import mxnet as mx
import mxnet.gluon.nn as nn
from autogluon.contrib.enas import *


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


def main():
    mynet = ENAS_Sequential(
        ResUnit(1, 8, hidden_channels=ag.space.Categorical(4, 8), kernel=ag.space.Categorical(3, 5), stride=2),
        ResUnit(8, 8, hidden_channels=8, kernel=ag.space.Categorical(3, 5), stride=2),
        ResUnit(8, 16, hidden_channels=8, kernel=ag.space.Categorical(3, 5), stride=2),
        ResUnit(16, 16, hidden_channels=8, kernel=ag.space.Categorical(3, 5), stride=1, with_zero=True),
        ResUnit(16, 16, hidden_channels=8, kernel=ag.space.Categorical(3, 5), stride=1, with_zero=True),
        nn.GlobalAvgPool2D(),
        nn.Flatten(),
        nn.Activation('relu'),
        nn.Dense(10, in_units=16),
    )

    mynet.initialize()

    x = mx.nd.random.uniform(shape=(1, 1, 28, 28))
    y = mynet.evaluate_latency(x)
    print('Average latency is {:.2f} ms, latency of the current architecture is {:.2f} ms'.format(mynet.avg_latency,
                                                                                                  mynet.latency))
    mynet.nparams
    reward_fn = lambda metric, net: metric * ((net.avg_latency / net.latency) ** 0.1)

    scheduler = ENAS_Scheduler(mynet, train_set='mnist',
                               reward_fn=reward_fn, batch_size=32, num_gpus=1,
                               warmup_epochs=0, epochs=1, controller_lr=3e-3,
                               plot_frequency=10, update_arch_frequency=5)
    scheduler.run()
    print(mynet.graph)


if __name__=="__main__":
    main()