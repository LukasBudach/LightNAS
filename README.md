
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



## Further notes and trouble shooting
 - Our repository includes an only slightly modified copy of the [BMXNet examples repository](https://github.com/hpi-xnor/BMXNet-v2-examples)  in the directory bmxnet_examples/. For simplicity we went with a copy instead of a git submodule here since we had to do some minor adoptions which would not have been possible when keeping this repository as a submodule only. We still want to refer to the [BMXNet examples repository](https://github.com/hpi-xnor/BMXNet-v2-examples) for further information.
 - If it seems that changes you did in the AutoGluon module are not applied it might be because you forgot to specify the ```develop``` argument during installing AutoGluon
 - Problems might also arise is the submodules are not up to date. Use ```git status``` to get the status of the update and update them if necessary (e.g. by going in the submodule directories and calling ```git pull``` in the submodule directories you want to update.)
