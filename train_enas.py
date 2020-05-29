from models.meliusnet_enas import meliusnet22_enas, meliusnet59_enas
from autogluon.contrib.enas import *
import mxnet as mx
from mxnet import nd
from autogluon.utils import DataLoader

def create_mock_gluon_image_dataset(num_samples=10, img_width=32, img_height=32, num_channels=3, num_classes=10):
    X = nd.random.uniform(shape=(num_samples,num_channels,img_height,img_width))
    y = nd.random.randint(0, num_classes, shape=(num_samples,1))
    train_dataset = mx.gluon.data.dataset.ArrayDataset(X,y)
    val_dataset = mx.gluon.data.dataset.ArrayDataset(X,y)

    val_data = DataLoader(
        val_dataset, batch_size=10, shuffle=True,
        num_workers=1, prefetch=0, sample_times=10)

    return train_dataset, val_dataset

def train_net_enas(net, epochs, name, log_dir='./logs/',
                   batch_size=64, train_set='imagenet', val_set=None):
    mynet = net

    def save_graph_val_fn(supernet, epoch):
        supernet.graph.render(log_dir + '/' + name + '/architectures/epoch_' + str(epoch))

    mynet.initialize()
    x = mx.nd.random.uniform(shape=(1, 3, 32, 32))
    mynet(x)
    y = mynet.evaluate_latency(x)
    print('Average latency is {:.2f} ms, latency of the current architecture is {:.2f} ms'.format(mynet.avg_latency,
                                                                                                  mynet.latency))
    mynet.nparams

    scheduler = ENAS_Scheduler(mynet, train_set=train_set, val_set=val_set, batch_size=batch_size, num_gpus=1,
                               warmup_epochs=0, epochs=epochs, controller_lr=3e-3,
                               plot_frequency=10, update_arch_frequency=5, post_epoch_fn=save_graph_val_fn)
    scheduler.run()

def main():
    train_set = 'cifar100'
    val_set = None
    #train_set, val_set = create_mock_gluon_image_dataset()
    train_net_enas(meliusnet22_enas().enas_sequential, 3, 'meliusnet22', train_set=train_set, val_set=val_set, batch_size=5)
    #train_net_enas(meliusnet59_enas().enas_sequential, 3, 'meliusnet59', train_set=train_set, val_set=val_set, batch_size=5)


if __name__=="__main__":
    main()