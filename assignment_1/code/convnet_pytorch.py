"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object.

    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem


    TODO:
    Implement initialization of the network.
    """
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet, self).__init__()

    structure = [['C', n_channels, 64], ['M', 64, 64], ['C', 64, 128], ['M', 128, 128], \
    ['C', 128, 256], ['C', 256, 256], ['M', 256, 256], ['C', 256, 512], ['C', 512, 512], \
    ['M', 512, 512], ['C', 512, 512], ['C', 512, 512], ['M', 512, 512], ['A', 512, 512], \
    ['L', 512, n_classes]]

    max_pool = {'stride': 2, 'padding': 1, 'kernel': 3}
    conv = {'stride': 1, 'padding': 1, 'kernel': 3}
    avg_pool = {'stride': 1, 'padding': 0, 'kernel': 1}

    layers = []
    for layer in structure:
        if layer[0] == 'C':
            layers.append(nn.Conv2d(in_channels=layer[1], out_channels=layer[2], \
                kernel_size=conv['kernel'], stride=conv['stride'], padding=conv['padding']))
            layers.append(nn.BatchNorm2d(layer[2]))
            layers.append(nn.ReLU())

        elif layer[0] == 'M':
            layers.append(nn.MaxPool2d(kernel_size=max_pool['kernel'], \
                stride=max_pool['stride'], padding=max_pool['padding']))
        elif layer[0] == 'A':
            layers.append(nn.AvgPool2d(kernel_size=avg_pool['kernel'], \
                stride=avg_pool['stride'], padding=avg_pool['padding']))
        else:
            layers.append(nn.Linear(layer[1], n_classes))

    self.layers = nn.Sequential(*layers)

    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through
    several layer transformations.

    Args:
      x: input to the network
    Returns:
      out: outputs of the network

    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    for layer in self.layers[:-1]:
        x = layer(x)

    out = self.layers[-1](x.view(x.shape[0], -1))
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
