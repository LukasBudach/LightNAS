
# LightNAS

## Installation

We recommend to use a virtual environment for the installation (e.g. virtualenv or conda)

 1. **Install BMXNet**
	 - you can find instructions how to install BMXNet from source in the [BMXNet repository](https://gitlab.hpi.de/joseph.bethge/bmxnet)

2. **Install other dependencies e.g. via pip** 
	 - install:  mxboard, pydot

3. **clone the repository with all submodules (there is a submodule with a modified version of AutoGluon and another one called Persistenargparse which is used for command line argument parsing)**
	- you can use ```console git clone --recurse-submodules``` to clone the repository with all its submodules

4. **install AutoGluon**
	 - go into the directory of the AutoGluon submodule (it is just called autogluon)
	 - install AutoGluon in development mode by calling ```python setup.py develop```


## Training
For training you need to execute train_enas.py. There are some command line arguments you can specify. Use ```python train_enas.py --help``` or look at the argparse specifications at the end of the train_enas.py file for further information on them. We use the persistentargparse module to save and load arguments in a convenient way using configuration files. There already exist several config file for the most common use cases in the config directory/. They can be used to start an ENAS training with either Meliusnet-22 or ResNet-18 as base architectures. There also exists a mock configuration for both kinds of training which uses mock data and allows for faster and easier debugging. You can use a configuration file by specifying its path as the ```-c``` argument.  For example call ```python train_enas.py -c ./configs/default_config_melius22_cifar100.yml``` to start an ENAS training with Meliusnet-22 as base architecture using cifar100 as dataset.


## Training Results
Training is logged in tensorboard. In the *shell_scripts/* folder there is a script called *run_tensorbaord.sh*. This script needs tensorboard to be installed. After running that script from inside the shell_scripts/ folder tensorboard is running on localhost under port 8989. You can now open your browser to view tensorboard on localhost:8989. In tensorboard you can see the development of the training and validation accuracy as well as the reward. In addition you can view visualizations of how the sampled architectures evolved.

Additional information can be found in the *trainings/* directory. There for each training a new folder is created. According to the arguments specified for training this folder may contain the following:

 - *logs/* directory containing a visualization of the final sampled architecture for each epoch
 - *exported_models/*directory containing symbol files and parameters of the sampled and trained architecture for each epoch
 - *enas_checkpoint/* directory containing the last checkpoint of the ENAS training which stores the state of the ENAS training and could be used to continue a training later on




## Further notes and trouble shooting
 - Our repository includes an only slightly modified copy of the [BMXNet examples repository](https://github.com/hpi-xnor/BMXNet-v2-examples)  in the directory bmxnet_examples/. For simplicity we went with a copy instead of a git submodule here since we had to do some minor adoptions which would not have been possible when keeping this repository as a submodule only. We still want to refer to the [BMXNet examples repository](https://github.com/hpi-xnor/BMXNet-v2-examples) for further information.
 - Under *visualization/* there exists a script called *color_graphs.py* which can be for additional to formatting and rendering the graphs of found architectures which are written to the *logs/* folder of a training
 - If it seems that changes you did in the AutoGluon module are not applied it might be because you forgot to specify the ```develop``` argument during installing AutoGluon
 - Problems might also arise is the submodules are not up to date. Use ```git status``` to get the status of the update and update them if necessary (e.g. by going in the submodule directories and calling ```git pull``` in the submodule directories you want to update.)
