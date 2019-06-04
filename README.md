# PyTorch Feedback Alignment Layers
PyTorch implementation of linear and convolutional layers with fixed, random feedback weights. Training a network that uses these layers relies on the feedback alignment effect described in ["Random synaptic feedback weights support error backpropagation for deep learning" by Lillicrap et al., 2016](https://www.nature.com/articles/ncomms13276).

Includes a sample convolutional network and code to train the network on CIFAR-10.

## Attribution

`utils.py` and `cifar10_example.py` adapted from:
https://github.com/kuangliu/pytorch-cifar
