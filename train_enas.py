from models.meliusnet_enas import *
from models.resnet_enas import *
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
    'imagenet': [224, 224, 3, 1000]
}


def create_mock_gluon_image_dataset(num_samples=20, img_width=32, img_height=32, num_channels=3, num_classes=10):
    """ Creates a small dataset out of random data for fast and easy debugging and testing.

    :param num_samples: number of images in the dataset, default 20
    :type num_samples: int
    :param img_width: width of each image in the dataset in pixels, default 32 (CIFAR)
    :type img_width: int
    :param img_height: height of each image in the datast in pixels, default 32 (CIFAR)
    :type img_height: int
    :param num_channels: number of channels of each picture e.g. 3 to simulate RGB pictures, default 3 (CIFAR)
    :type num_channels: int
    :param num_channels: number of classes a picture can be classified as, default 10 (CIFAR-10)
    :type num_channels: int

    :return: tuple of two gluon datasets (one train- and one validation dataset)
    :rtype: Tuple[mx.gluon.data.dataset.ArrayDataset, mx.gluon.data.dataset.ArrayDataset]
    """
    X = nd.random.uniform(shape=(num_samples,num_channels,img_height,img_width))
    y = nd.random.randint(0, num_classes, shape=(num_samples,1))
    train_dataset = mx.gluon.data.dataset.ArrayDataset(X,y)
    val_dataset = mx.gluon.data.dataset.ArrayDataset(X,y)

    return train_dataset, val_dataset


def train_net_enas(net, epochs, training_name, batch_size=64, train_set='cifar100', val_set=None,
                   num_gpus=0, num_workers=4, net_init_shape=(1, 3, 32, 32), export_to_inference=True,
                   export_to_trainable=True, export_model_name='model', verbose=True, custom_batch_fn=None,
                   eval_split_pct=0.5, external_eval=False):
    """ Main function for ENAS training of a given supernet.

        :param net: the supernet
        :type net: autogluon.contrib.enas.enas.ENAS_Sequential
        :param epochs: number of epochs to train for
        :type epochs: int
        :param training_name: name for the training, e.g. used to display training in tensorboard and for results
                              directory name
        :type training_name: str
        :param batch_size: batch size for training the sampled architectures, default 64
        :type batch_size: int
        :param train_set: dataset used for training, default cifar100
        :type train_set: str or mx.gluon.data.Dataset or mx.gluon.data.dataloader.DataLoader
        :param val_set: dataset used for validation, if train_set is a string and val_set is None the validation data of
                        the dataset specified as train_set will be taken, default None
        :type val_set: NoneType or str or mx.gluon.data.Dataset or mx.gluon.data.dataloader.DataLoader
        :param num_gpus: when given as integer the first num_gpus gpus of the machine are used when specified as tuple
                         the gpus given as the numbers in the tuple are used, default 0
        :type num_gpus: int or tuple
        :param num_workers: number of threads used for controller sampling, default 4
        :type num_workers: int
        :param net_init_shape: shape of the network input, used for initializing the network, default (1, 3, 32, 32) (CIFAR)
        :type net_init_shape: tuple
        :param export_to_inference: whether symbol files of the trained architectures at the end of each epoch should
                                    be written which then can be loaded and used for inference, default True
        :type export_to_inference: bool
        :param export_to_trainable: If true, only the trained parameters of the currently sampled network architecture
                                    are written to disk at the end of each epoch, default True
        :type export_to_trainable: bool
        :param export_model_name: file name of the exported models, default model
        :type export_model_name: str
        :param verbose: whether additional information like the net summary should be printed during training, default True
        :type verbose: bool
        :param custom_batch_fn: custom function for loading batches from the dataset, default None
        :type custom_batch_fn: function or NoneType
        :param eval_split_pct: percentage of the dataset which should be used for evaluating the accuracy, default 0.5
        :type eval_split_pct: float
        :param external_eval: whether the evaluation should happen after each epoch during training or only once
                              after the training concluded, default False
        :type external_eval: bool

        :return: nothing
    """
    if export_to_inference and export_to_trainable:
        option = ['inference', 'trainable']
    elif export_to_inference:
        option = ['inference']
    elif export_to_trainable:
        option = ['trainable']
    else:
        option = ['ignore']

    train_dir = Path('./trainings/{}'.format(training_name))

########################################################################################################################
########################################## Get Dataset for external evaluation #########################################
########################################################################################################################
    if external_eval:
        from autogluon.task.image_classification.dataset import get_built_in_dataset
        from autogluon.utils.dataloader import DataLoader

        print('There will be post training evaluation and no post epoch evaluation!')

        def split_val_data(val_dataset):
            """Splits the validation dataset in a validation and a test part. The validation part can then be used to
            train the controller whereas the test dataset is only used for evaluating the network accuracy. The split is
            conducted in the same way as is done in autogluon/autogluon/contrib/enas/enas_scheduler.py

            :param val_dataset: The original validation dataset which should be split
            :type val_dataset: str or mx.gluon.data.Dataset or mx.gluon.data.dataloader.DataLoader
            :return: nothing
            """
            eval_part = round(len(val_dataset) * eval_split_pct)
            print('The first {}% of the validation dataset will be held back for evaluation instead.'.format(eval_split_pct*100))
            eval_dataset = tuple([[], []])
            new_val_dataset = tuple([[], []])
            for i in range(eval_part):
                eval_dataset[0].append(val_dataset[i][0])
                eval_dataset[1].append(val_dataset[i][1])
            for i in range(eval_part, len(val_dataset)):
                new_val_dataset[0].append(val_dataset[i][0])
                new_val_dataset[1].append(val_dataset[i][1])

            eval_dataset = mx.gluon.data.ArrayDataset(eval_dataset[0], eval_dataset[1])
            new_val_dataset = mx.gluon.data.ArrayDataset(new_val_dataset[0], new_val_dataset[1])

            return new_val_dataset, eval_dataset

        # if external validation is done, get the dataset now instead of in the ENAS_Scheduler and split it
        # acquiring the dataset is done in the same way as the ENAS_Scheduler would do it.
        if isinstance(train_set, str):
            train_set = get_built_in_dataset(train_set, train=True, batch_size=batch_size,
                                             num_workers=num_workers, shuffle=True, fine_label=True)
            val_set = get_built_in_dataset(val_set, train=False, batch_size=batch_size,
                                           num_workers=num_workers, shuffle=True, fine_label=True)
        if isinstance(train_set, mx.gluon.data.Dataset):
            # split the validation set into an evaluation and validation set
            val_dataset, eval_dataset = split_val_data(val_set)
            train_set = DataLoader(
                    train_set, batch_size=batch_size, shuffle=True,
                    last_batch="discard", num_workers=num_workers)
            # very important, make shuffle for training contoller
            val_set = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, prefetch=0, sample_times=10)  # sample_times = ENASScheduler controller_batchsize
            eval_set = DataLoader(
                    eval_dataset, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, prefetch=0, sample_times=10)  # sample_times = ENASScheduler controller_batchsize
        elif isinstance(train_set, mx.gluon.data.dataloader.DataLoader):
            val_dataset, eval_dataset = split_val_data(val_set._dataset)


            val_set = DataLoader.from_other_with_dataset(val_set, val_dataset)
            eval_set = DataLoader.from_other_with_dataset(val_set, eval_dataset)

        eval_split_pct = 0

########################################################################################################################
################################################# Function definitions #################################################
########################################################################################################################
    def save_graph_val_fn(supernet, epoch):
        """Callback function for saving a visualization of the currently sampled architecture at the end of each epoch.
        :param supernet: The supernet in its current state
        :type supernet: autogluon.contrib.enas.enas.ENAS_Sequential
        :param epoch: the current epoch
        :type epoch: int

        :return: nothing
        """
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

    def save_model(supernet, epoch):
        """Callback function for saving the currently sampled architecture and its parameters
        :param supernet: The supernet in its current state
        :type supernet: autogluon.contrib.enas.enas.ENAS_Sequential
        :param epoch: the current epoch
        :type epoch: int

        :return: nothing
        """
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
                hybnet.export(export_dir / Path(str(export_model_name) + str(epoch)), epoch=epoch)
                print('Inference model has been exported to {}'.format(export_dir))
            if decision == 'trainable':
                export_dir = train_dir / 'exported_models/trainables'
                export_dir.mkdir(parents=True, exist_ok=True)
                hybnet.save_parameters(str((export_dir / (export_model_name + "_{:04d}".format(epoch)))
                                           .with_suffix('.params').resolve()))
                print('Trainable model has been exported to {}'.format(export_dir))

    def evaluation(sched):
        """Evaluates the final, chosen, trained architecture using the held out test/evaluation dataset.
        :param sched: The ENAS_Scheduler containing the trained supernet
        :type supernet: autogluon.contrib.enas.enas.ENAS_Scheduler

        :return: nothing
        """
        from tqdm import tqdm

        print('------------------- Running post training evaluation -------------------')
        if hasattr(eval_set, 'reset'): eval_set.reset()
        # data iter, avoid memory leak
        it = iter(eval_set)
        if hasattr(it, 'reset_sample_times'): it.reset_sample_times()
        tbar = tqdm(it)
        # update network arc
        config = sched.controller.inference()
        sched.supernet.sample(**config)
        metric = mx.metric.Accuracy()
        for batch in tbar:
            sched.eval_fn(sched.supernet, batch, metric=metric, **sched.val_args)
            reward = metric.get()[1]
            tbar.set_description('Eval Acc: {}'.format(reward))
        print('>> Evaluation Accuracy: {}'.format(reward))

########################################################################################################################
################################################### Network Training ###################################################
########################################################################################################################
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
    checkpoint_name = train_dir / 'enas_checkpoint/checkpoint.ag'
    scheduler = ENAS_Scheduler(net, train_set=train_set, val_set=val_set, batch_size=batch_size, num_gpus=num_gpus,
                               warmup_epochs=0, epochs=epochs, controller_lr=3e-3, plot_frequency=10,
                               update_arch_frequency=5, post_epoch_fn=save_graph_val_fn, post_epoch_save=save_model,
                               custom_batch_fn = custom_batch_fn, num_cpus=num_workers, eval_split_pct=eval_split_pct,
                               tensorboard_log_dir='./tensorboard_logs/', training_name=training_name,
                               checkname=checkpoint_name)
    scheduler.run()

    if external_eval:
        evaluation(scheduler)


def main(args):
    """Main function running arg parsing and dataset initialization, as well as starting the ENAS training
    :param args: The arguments used to configure the ENAS training
    :type args: argparse.Namespace

    :return: nothing
    """
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
            train_set, val_set, batch_fn = get_data_iters(args)
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
    if args.model.startswith('resnet'):
        kwargs['grad_cancel'] = args.grad_cancel
    train_net_enas(globals()[args.model](**kwargs).enas_sequential, args.epochs,
                   training_name=training_name, train_set=train_set, val_set=val_set,
                   batch_size=args.batch_size, num_gpus=args.num_gpus, num_workers=args.num_workers,
                   net_init_shape=init_shape, verbose=args.verbose, export_model_name=args.export_model_name,
                   export_to_inference=args.export_to_inference, export_to_trainable=args.export_to_trainable,
                   custom_batch_fn=batch_fn, eval_split_pct=args.eval_split_percentage,
                   external_eval=args.only_post_training_eval)


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
    parser.add_argument('--num-workers', type=int, default=4, help='Number of threads used for data loading. Default is'
                                                                   ' 0 (all work is done in main thread).')

    parser.add_argument('--verbose', action='store_true', help='Prints a summary and the network repr after '
                                                               'initializing the network.')
    parser.add_argument('--export-to-inference', action='store_true', help='Set to save model for further inference.')
    parser.add_argument('--export-to-trainable', action='store_true', help='Set to save model as a trainable model.')
    parser.add_argument('--export-model-name', type=str, default='model', help='Name of the saved model.')
    parser.add_argument('--augmentation', choices=["low", "medium", "high"], default="medium",
                      help='How much augmentation should be used. Only considered when bmx-examples-datasets are used.')

    parser.add_argument('--eval-split-percentage', type=float, required=True,
                        help='Percentage of the validation data that should be held back for an additional evaluation loop.')
    parser.add_argument('--only-post-training-eval', action='store_true',
                        help='Set to disable the evaluation loop after each epoch and run evaluation once after the '
                             'training concluded instead.')
    parser.add_argument('--grad-cancel', type=float,
                        help='Upper threshold for 1 bit convolution gradient (For now only for resnet relevant).')

    main(parser.parse_args())
