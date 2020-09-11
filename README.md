
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
	 - install AutoGluon in development mode using the by calling ```python setup.py develop```


## Training
- todo




## Old stuff (should be deleted in the end)
 - Write down setup!
 - Write down some weird quirks we have (like copied bmxnet examples)
