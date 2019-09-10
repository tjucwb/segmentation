# segmentation
There are 4 python files here, namely data_input.py, fully_conv_net.py, seg_modi.py, utils.py
All files are written in python 3.6 and tensorflow-gpu 1.10, numpy 1.16, opencv 4.1.0.25

The aim for the included files is to accomplish segmentation in an image, for this aim, two methods are used:
    1. Seg.modi.py accomplishes segmentation based on an Auto-encoder like network. 
This file can be run directly if data are available
Function and parameter instruction are included in the file.
    2. Fully_conv_net.py accomplishes segmentation based on fully-connected-layer network with pre-trained backbone (Vgg-19) for feature extraction (network architecture as [1])
This file can be run directly if data are available

utils.py includes all the needed functions for fully_conv_net.py, which includes some api function, plot function, error rate function, model download and input function.

Data_input.py is a file includes a class database, in this class, all data-related functions are included, including training/test data/annotation extraction, gradient-based operation on data and data fetch for training.  

[1] Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015. 
