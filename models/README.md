# LightNAS - Custom Models

This directory contains the customized models used for ENAS training. The regular MeliusNet and ResNet models, provided in the bmxnet_examples, do not allow for block skipping or other ways of ENAS training, so we decided to copy the model definitions and adapt them for ENAS. 

## Basenet Dense and MeliusNet

In the ENAS training of MeliusNet, we allow the controller to either use or skip a Dense or Improvement Block. For this reason, the ``DenseBlockEnas`` and ``ImprovementBlockEnas`` are used instead of regular Dense or Improvement Blocks. Both are defined with the decorator ```@enas_unit(replace_by_skip_connection=ag.space.Categorical(True, False))```, indicating to the AutoGluon ENAS that the decision to skip can be made. In case of the ``DenseBlockEnas``, it is replaced by a 1x1 Convolution if it is to be skipped, as the channel number still must change for the network to contain validity. The ``ImprovementBlockEnas`` is replaced by an Identity Block if it is to be skipped.

## MeliusNet Custom

For external testing and architecture reproduction, we implemented a way to define a custom MeliusNet consisting of Dense Blocks, Improvement Blocks and 1x1 Convolutions using a string. The Identity Blocks are left out, as adding them has no effect on the network. This custom MeliusNet is integrated in our modified BMXNet Examples by having been registered in *bmxnet_examples/binary_models/\_\_init\_\_.py*

An example network configuration string would be **DTSI**. This string corresponds to the network architecture Initial layers - **D**enseBlock - Skip ImprovementBlock (left out) - **T**ranstition Layer - **S**kip Dense (1x1 Convolution) - **I**mprovementBlock - final layers

## ResNet
 
Instead of skipping blocks, the decision in ResNet is whether to use the Residual Blocks in full (floating point) precision, or binarize their weights. Thus, the ``BasicBlockVxEnas`` and ``BottleNeckBlockVxEnas`` classes are defined with the decorator ``@enas_unit(bits=ag.space.Categorical(1,32), share_parameters=True)``, allowing ENAS the choice of using 1 bit or 32 bits. The weights for each block are trained in full precision and simply binarized if the 1 bit option is chosen.