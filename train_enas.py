from models.meliusnet_enas import meliusnet22_enas, meliusnet59_enas
from autogluon.contrib.enas import *
import mxnet as mx
from mxnet import nd
from autogluon.utils import DataLoader
from visualization.color_graphs import format_and_render
from pathlib import Path
try:
    from persistentargparse import PersistentArgumentParser
except ImportError:
    print('Could not import from persistentargparse. The module should be added as git submodule to this repository. '
          'Please run git submodule init and git submodule update --remote')
    exit(1)

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
                   batch_size=64, train_set='imagenet', val_set=None, num_gpus=1):

    def save_graph_val_fn(supernet, epoch):
        filepath = Path(log_dir + '/' + name + '/architectures/epoch_' + str(epoch) + '.dot')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        print('\nSaving graph to ' + str(filepath) + '\n')
        supernet.graph.save(filepath)
        format_and_render(filepath)

    #net is an ENAS_Sequential object
    net.initialize()
    x = mx.nd.random.uniform(shape=(1, 3, 32, 32))
    net(x)
    y = net.evaluate_latency(x)
    print('Average latency is {:.2f} ms, latency of the current architecture is {:.2f} ms'.format(net.avg_latency,
                                                                                                  net.latency))
    scheduler = ENAS_Scheduler(net, train_set=train_set, val_set=val_set, batch_size=batch_size, num_gpus=num_gpus,
                               warmup_epochs=0, epochs=epochs, controller_lr=3e-3,
                               plot_frequency=10, update_arch_frequency=5, post_epoch_fn=save_graph_val_fn)
    scheduler.run()


def main(args):
    train_set = args.train_data
    val_set = args.val_data
    #train_set, val_set = create_mock_gluon_image_dataset()
    #we set num_gpus=(1,) because when specifying a tuple we can set a specific gpu mapping
    train_net_enas(meliusnet22_enas().enas_sequential, args.epochs, 'meliusnet22_enas_kernelsize1', train_set=train_set,
                   val_set=val_set, batch_size=args.batch_size, num_gpus=(args.num_gpus,))
    #train_net_enas(meliusnet59_enas().enas_sequential, 3, 'meliusnet59', train_set=train_set, val_set=val_set, batch_size=5)


if __name__=="__main__":
    parser = PersistentArgumentParser(description='Train ENAS is the script provided to train the defined ENAS network '
                                                  'in order to find a different architecture for the net.')

    parser.add_argument('-b', '--batch-size', type=int, required=True, help='Batch size to use during training.')
    parser.add_argument('-e', '--epochs', type=int, required=True, help='Number of epochs to train for.')
    parser.add_argument('--num-gpus', type=int, required=True, help='Number of available GPUs to use for the training.')
    parser.add_argument('--train-data', type=str, required=True, help='Autogluon specifier for the dataset to use for '
                                                                      'training.')
    parser.add_argument('--val-data', type=str, required=False, help='Autogluon specifier for the dataset to use for '
                                                                     'validation.')

    main(parser.parse_args())
