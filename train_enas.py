from models.meliusnet_enas import meliusnet22_enas, meliusnet59_enas
from autogluon.contrib.enas import *
from datetime import datetime
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

# dictionary mapping dataset name to list of [image width, image height, number of channels, number of classes]
dataset_prop = {
    'cifar10': [32, 32, 3, 10],
    'cifar100': [32, 32, 3, 100],
    'imagenet': [0, 0, 3, 0]    # TODO: not sure what dimension the imagenet images Joseph provides have and whether they need to be cropped or not
}


def create_mock_gluon_image_dataset(num_samples=10, img_width=32, img_height=32, num_channels=3, num_classes=10):
    X = nd.random.uniform(shape=(num_samples,num_channels,img_height,img_width))
    y = nd.random.randint(0, num_classes, shape=(num_samples,1))
    train_dataset = mx.gluon.data.dataset.ArrayDataset(X,y)
    val_dataset = mx.gluon.data.dataset.ArrayDataset(X,y)

    return train_dataset, val_dataset


def train_net_enas(net, epochs, train_dir, batch_size=64, train_set='cifar100', val_set=None,
                   num_gpus=0, net_init_shape=(1, 3, 32, 32), export_to_inference=True, export_to_trainable=True,
                   export_model_name='teste01', verbose=True, custom_batch_fn=None):

    def save_graph_val_fn(supernet, epoch):
        viz_filepath = (train_dir / ('logs/architectures/epoch_' + str(epoch))).with_suffix('.dot')
        txt_filepath = (train_dir / ('logs/architectures/epoch_' + str(epoch))).with_suffix('.txt')

        # saves the visualization
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
        if export_model_name is None:
            raise ValueError('If you are exporting your model, you must provide the model name as argument')

        for decision in option:
            if decision == 'ignore':
                continue

            mock_data = mx.nd.random.normal(shape=net_init_shape, ctx=mx.gpu() if num_gpus > 0 else mx.cpu())
            hybnet = nn.HybridSequential()
            for layer in supernet.prune():
                hybnet.add(layer)
            hybnet.hybridize()
            hybnet(mock_data)

            if decision == 'inference':
                export_dir = train_dir / 'exported_models/inference_only'
                export_dir.mkdir(parents=True, exist_ok=True)
                hybnet.export(export_dir / export_model_name, epoch=epoch)
                print('Inference model has been exported to {}'.format(export_dir))
            if decision == 'trainable':
                export_dir = train_dir / 'exported_models/trainables'
                export_dir.mkdir(parents=True, exist_ok=True)
                hybnet.save_parameters(str((export_dir / (export_model_name + "_{:04d}".format(epoch)))
                                           .with_suffix('.params').resolve()))
                print('Trainable model has been exported to {}'.format(export_dir))

    # net is an ENAS_Sequential object
    net.initialize()
    # create an initial input for the network with the same dimensions as the data from the given train and val datasets
    x = mx.nd.random.uniform(shape=net_init_shape)
    net(x)

    if verbose:
        print(net)
        net.summary(x)

    y = net.evaluate_latency(x)
    print('Average latency is {:.2f} ms, latency of the current architecture is {:.2f} ms'.format(net.avg_latency,
                                                                                                  net.latency))
    scheduler = ENAS_Scheduler(net, train_set=train_set, val_set=val_set, batch_size=batch_size, num_gpus=num_gpus,
                               warmup_epochs=0, epochs=epochs, controller_lr=3e-3, plot_frequency=10,
                               update_arch_frequency=5, post_epoch_fn=save_graph_val_fn, post_epoch_save=save_model,
                               custom_batch_fn = custom_batch_fn)
    scheduler.run()


def main(args):
    # define additional arguments for the network construction
    kwargs = {'initial_layers': args.initial_layers}
    if args.dataset is not None:
        kwargs['classes'] = dataset_prop[args.dataset][3]
        init_shape = (1, dataset_prop[args.dataset][2], dataset_prop[args.dataset][0], dataset_prop[args.dataset][1])
        train_set = args.dataset
        val_set = args.dataset
        batch_fn = None
        # if the mock training data is asked for, create the mock dataset for training and validation
        if args.use_bmx_examples_datasets:
            train_data, val_data, batch_fn = get_data_iters(args)
    else:
        # since train set is not defined, we need to mock
        kwargs['classes'] = dataset_prop[args.mock][3]
        init_shape = (1, dataset_prop[args.mock][2], dataset_prop[args.mock][0], dataset_prop[args.mock][1])
        train_set, val_set = create_mock_gluon_image_dataset(img_width=dataset_prop[args.mock][0],
                                                             img_height=dataset_prop[args.mock][1],
                                                             num_channels=dataset_prop[args.mock][2])
        batch_fn = None
    # define a new name for this training
    now = datetime.now()
    training_name = args.training_name if args.training_name is not None else args.model + '_{}_{}_{}_{}_{}'\
        .format(now.year, now.month, now.day, now.hour, now.minute)
    train_net_enas(globals()[args.model](**kwargs).enas_sequential, args.epochs,
                   train_dir=Path('./trainings/{}'.format(training_name)), train_set=train_set, val_set=val_set,
                   batch_size=args.batch_size, num_gpus=args.num_gpus, net_init_shape=init_shape, verbose=args.verbose,
                   export_model_name=args.export_model_name, export_to_inference=args.export_to_inference,
                   export_to_trainable=args.export_to_trainable, custom_batch_fn=batch_fn)


if __name__ == "__main__":
    supported_datasets = ['cifar10', 'cifar100']

    parser = PersistentArgumentParser(description='Train ENAS is the script provided to train the defined ENAS network '
                                                  'in order to find a different architecture for the net.')

    parser.add_argument('-b', '--batch-size', type=int, required=True, help='Batch size to use during training.')
    parser.add_argument('-e', '--epochs', type=int, required=True, help='Number of epochs to train for.')
    parser.add_argument('--initial-layers', type=str, required=True, help='Initial layer specifier.',
                        choices=['imagenet', 'thumbnail'])
    parser.add_argument('-m', '--model', type=str, required=True, help='Network architecture to be trained (e.g. '
                                                                       'meliusnet22_enas).')
    parser.add_argument('--num-gpus', type=int, required=True, help='Number of available GPUs to use for the training.')
    parser.add_argument('--training-name', type=str, help='Name you want to use for this training run, will be used in '
                                                          'log and model saving.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', type=str, help='If --use-bmx-examples-datasets=False the Autogluon specifier for '
                                                   'the dataset to use for training. If --use-bmx-examples-datasets='
                                                   'True then the name of the bmxnet examples dataset to use.',
                       choices=supported_datasets)
    group.add_argument('--mock', type=str, help='Specifier for the dataset that is to be used for mocking.',
                       choices=supported_datasets)
    parser.add_argument('--use-bmx-examples-datasets', action='store_true', help='Flag whether the string given with '
                                                                                 '--train-data should be interpreted as'
                                                                                 ' an Autogluon dataset or the datasets'
                                                                                 ' from bmxnet_examples should be used.')
    parser.add_argument('--data-dir', type=str, help='Path to the directory containing the datasets. Required when '
                                                     'using the imagenet dataset from the bmxnet examples datasets.')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of threads used for data loading. Default is'
                                                                   ' 0 (all work is done in main thread).')

    parser.add_argument('--verbose', action='store_true', help='Prints a summary and the network repr after '
                                                               'initializing the network.')
    parser.add_argument('--export-to-inference', action='store_true', help='Set to save model for further inference.')
    parser.add_argument('--export-to-trainable', action='store_true', help='Set to save model as a trainable model.')
    parser.add_argument('--export-model-name', type=str, default='model', help='Name of the saved model.')

    main(parser.parse_args())
