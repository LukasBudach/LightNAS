from models.meliusnet_enas import meliusnet22_enas, meliusnet59_enas
from autogluon.contrib.enas import *
import mxnet.gluon.nn as nn
import mxnet as mx
from mxnet import nd
from visualization.color_graphs import format_and_render
from pathlib import Path
try:
    from persistentargparse import PersistentArgumentParser
except ImportError:
    print('Could not import from persistentargparse. The module should be added as git submodule to this repository. '
          'Please run git submodule init and git submodule update --remote')
    exit(1)
from bmxnet_examples.datasets.util import get_data_iters


def create_mock_gluon_image_dataset(num_samples=10, img_width=32, img_height=32, num_channels=3, num_classes=10):
    X = nd.random.uniform(shape=(num_samples,num_channels,img_height,img_width))
    y = nd.random.randint(0, num_classes, shape=(num_samples,1))
    train_dataset = mx.gluon.data.dataset.ArrayDataset(X,y)
    val_dataset = mx.gluon.data.dataset.ArrayDataset(X,y)

    return train_dataset, val_dataset

def train_net_enas(net, epochs, name, log_dir='./logs/', batch_size=64, train_set='imagenet', val_set=None,
                   num_gpus=0, export_to_inference=True, export_to_trainable=True, export_model_name='teste01',
                   verbose=True, custom_batch_fn=None):

    def save_graph_val_fn(supernet, epoch):
        viz_filepath = Path(log_dir + '/' + name + '/architectures/epoch_' + str(epoch) + '.dot')
        txt_filepath = Path(log_dir + '/' + name + '/architectures/epoch_' + str(epoch) + '.txt')

        # saves the vizualization
        viz_filepath.parent.mkdir(parents=True, exist_ok=True)
        print('\nSaving graph to ' + str(viz_filepath) + '\n')
        supernet.graph.save(viz_filepath)
        format_and_render(viz_filepath)

        # saves the architecture in txt format
        txt_file = open(txt_filepath, "w")
        txt_file.write(supernet.__repr__())
        txt_file.close()

    if export_to_inference and export_to_trainable:
        option = ['inference', 'trainable']
    elif export_to_inference:
        option = ['inference']
    elif export_to_trainable:
        option = ['trainable']
    else:
        option = ['ignore']

    def save_model(supernet, epoch):
        if 'ignore' in option:
            raise ValueError('If you are exporting your model, you must provide the model name as argument')

        for decision in option:
            if decision == 'inference':
                # TODO: GENERATE MOCK DATA ACCORDING TO THE TRAIN SET
                mock_data = mx.nd.random.normal(shape=(1, 3, 32, 32))
                hybnet = nn.HybridSequential()
                for layer in supernet.prune():
                    hybnet.add(layer)
                hybnet.hybridize()
                hybnet(mock_data)
                export_dir = 'exported_models/inference_only/'
                hybnet.export(export_dir + export_model_name, epoch=epoch)
                print('Inference model has been exported to {}'.format(export_dir))
            if decision == 'trainable':
                export_dir = 'exported_models/trainables/'
                hybnet.save_parameters(export_dir + export_model_name + '.params')
                print('Trainable model has been exported to {}'.format(export_dir))
            if decision == 'ignore':
                return


    # net is an ENAS_Sequential object
    net.initialize()
    # create an initial input for the network with the same dimensions as the data from the given train and val datasets
    x = mx.nd.random.uniform(shape=(1, 3, 32, 32))
    net(x)

    if verbose:
        print(net)
        net.summary(x)

    y = net.evaluate_latency(x)
    print('Average latency is {:.2f} ms, latency of the current architecture is {:.2f} ms'.format(net.avg_latency,
                                                                                                  net.latency))
    scheduler = ENAS_Scheduler(net, train_set=train_set, val_set=val_set, batch_size=batch_size, num_gpus=num_gpus,
                               warmup_epochs=0, epochs=epochs, controller_lr=3e-3,
                               plot_frequency=10, update_arch_frequency=5, post_epoch_fn=save_graph_val_fn,
                               post_epoch_save=save_model, custom_batch_fn = custom_batch_fn)
    scheduler.run()


def main(args):
    train_set = args.dataset
    val_set = None
    batch_fn = None
    # if the mock training data is asked for, create the mock dataset for training and validation
    if args.use_bmx_examples_datasets:
        train_data, val_data, batch_fn = get_data_iters(args)
    if train_set == 'mock':
        train_set, val_set = create_mock_gluon_image_dataset()
    # define additional arguments for the network construction
    kwargs = {'initial_layers': args.initial_layers, 'classes': args.num_classes}
    train_net_enas(globals()[args.model](**kwargs).enas_sequential, args.epochs, 'meliusnet22_enas_kernelsize1',
                   train_set=train_set, val_set=val_set, batch_size=args.batch_size, num_gpus=args.num_gpus,
                   verbose=args.verbose, custom_batch_fn=batch_fn)


if __name__ == "__main__":
    parser = PersistentArgumentParser(description='Train ENAS is the script provided to train the defined ENAS network '
                                                  'in order to find a different architecture for the net.')

    parser.add_argument('-b', '--batch-size', type=int, required=True, help='Batch size to use during training.')
    parser.add_argument('-e', '--epochs', type=int, required=True, help='Number of epochs to train for.')
    parser.add_argument('--initial-layers', type=str, required=True, help='Initial layer specifyer.',
                        choices=['imagenet', 'thumbnail'])
    parser.add_argument('-n', '--model', type=str, required=True, help='Network architecture to be trained (e.g. '
                                                                         'meliusnet22_enas).')
    parser.add_argument('--num-gpus', type=int, required=True, help='Number of available GPUs to use for the training.')
    parser.add_argument('--dataset', type=str, required=True, help='If --use-bmx-examples-datasets=False'
                                                                      ' the Autogluon specifier for'
                                                                      ' the dataset to use for training. Pass mock in'
                                                                      ' order to mock the training and validation data.'
                                                                        'If --use-bmx-examples-datasets=True'
                                                                      ' then the name of the '
                                                                      'bmxnet examples dataset to use.')
    parser.add_argument('--use-bmx-examples-datasets', type=bool, default=False,
                                                                help='Flag whether the string given with --train-data'
                                                                'should be interpreted as an Autogluon dataset'
                                                                 'or whether the datasets from bmxnet examples'
                                                                 'should be used.')
    parser.add_argument('--num-classes', type=int, required=False, default=100, help='Number of classes of the dataset.')
    parser.add_argument('--verbose', type=bool, required=False, help='Prints a summary and the network repr after initializing the network.')
    parser.add_argument('--export-to-inference', type=bool, required=False, help='If save model for further inference.')
    parser.add_argument('--export-to-trainable', type=bool, required=False, help='If save model as a trainable model.')
    parser.add_argument('--export-model-name', type=str, required=False, help='Name of the saved model.')
    parser.add_argument('--data-dir', type=str, required=False, help='Required when using the imagenet dataset from the'
                                                                     'bmxnet examples datasets.')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of threads used for data loading. Default is'
                                                                   ' 0 (all work is done in main thread).')

    main(parser.parse_args())