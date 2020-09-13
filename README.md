
# LightNAS

## Installation

We recommend to use a virtual environment for the installation (e.g. virtualenv or conda). This project has been developed and used on Linux only, due to requirements not being supported on Mac (AutoGluon on GPU) or on Windows (AutoGluon in general, possibly BMXNet). Addtionally, the AutoGluon toolkit requires the python versions ``3.6.x`` or ``3.7.x`` to be used.

 1. **Install BMXNet**
	 - you can find instructions how to install BMXNet from source in the [BMXNet repository](https://github.com/hpi-xnor/BMXNet-v2)

2. **Install other dependencies e.g. via pip** 
	 - install:  mxboard, pydot
	 - **OR** call ```pip install -r requirements.txt```

3. **clone this repository with all submodules** (there is a submodule with a modified version of AutoGluon and another one called Persistenargparse which is used for command line argument parsing)
	- you can use ```git clone --recurse-submodules``` to clone the repository with all its submodules
	- afterwards, make sure that the directories *persistentargparse/* and *autogluon/* exist and are not empty
	    - if any of the directories is empty, please navigate to the *LightNas* root directory and call ```git submodule update --init --recursive```

4. **install AutoGluon**
	 - in your *LightNas* directory, go into the directory of the AutoGluon submodule (it is just called *autogluon*)
	 - install AutoGluon in development mode by calling ```python setup.py develop``` (this updates the module if you modify code in it)
	 - **OR** install AutoGluon by building it first ```python setup.py build``` and then install it ```python setup.py install``` (this will not update with the changing code and will need to be rebuilt and reinstalled)


## Training
For training you need to execute train_enas.py. There are some command line arguments you can specify. Use ```python train_enas.py --help``` or look at the argparse specifications at the end of the train_enas.py file for further information on them. We use the persistentargparse module to save and load arguments in a convenient way, using configuration files. There already exist several default config files for the most common use cases in the *config/* directory. They can be used to start an ENAS training with either Meliusnet-22 or ResNet-18 as base architectures. There also exists a mock configuration for both kinds of training which uses mock data and allows for faster and easier debugging. You can use a configuration file by specifying its path as the ```-c``` argument.  For example call ```python train_enas.py -c ./configs/default_config_melius22_cifar100.yml``` to start an ENAS training with Meliusnet-22 as base architecture using cifar100 as dataset.

> If you do not use a configuration file, or provide additional commandline arguments when using one, a new config file will be saved in *configs/autosaved/*.


## Training Results
Training is logged in tensorboard. In the *shell_scripts/* folder there is a script called *run_tensorbaord.sh*. This script needs tensorboard to be installed. After running that script from inside the *shell_scripts/* folder, tensorboard is running on localhost under port 8989. You can now open your browser to view tensorboard (localhost:8989). In tensorboard, you can see the development of the training and validation accuracy as well as the reward. In addition, you can view visualizations of how the sampled architectures evolved.

Additional results can be found in the *trainings/* directory. In there, a new folder is created for each training. According to the arguments specified for training this folder may contain any of the following:

 - *logs/* directory, containing a visualization of the final sampled architecture for each epoch
 - *exported_models/* directory, containing symbol files and parameters of the sampled and trained architecture for each epoch
 - *enas_checkpoint/* directory, containing the last checkpoint of the ENAS training, which stores the state of the ENAS training and could be used to continue a training later on


## Further notes and trouble shooting
 - Our repository includes an only slightly modified copy of the [BMXNet examples repository](https://github.com/hpi-xnor/BMXNet-v2-examples)  in the directory *bmxnet_examples/*. For simplicity, we went with a copy instead of a git submodule here, since we had to do some minor adoptions which would not have been possible when keeping this repository as a submodule only. We still want to refer to the [BMXNet examples repository](https://github.com/hpi-xnor/BMXNet-v2-examples) for further information.
 - Under *visualization/* there exists a script called *color_graphs.py* which can be used for additional formatting and rendering of the graphs of found architectures which are written to the *logs/* folder of a training
 - If it seems like changes you did in the AutoGluon module are not applied it might be because you forgot to specify the ```develop``` argument during installing AutoGluon
 - Problems might also arise if the submodules are not up to date. Use ```git status``` to get the status of the update and update them if necessary (e.g. by going in the submodule directories and calling ```git pull``` in the submodule directories you want to update. It may be required to specify from where to pull, in that case call ```git pull origin master```)
