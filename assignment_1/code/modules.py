"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data.
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module.

    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and
    std = 0.0001. Initialize biases self.params['bias'] with 0.

    Also, initialize gradients with zeros.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    mu, sig = 0, 0.0001
    gaussian = np.random.normal(mu, sig, (in_features, out_features))
    zeros = np.zeros((in_features, out_features))
    bias = np.zeros((out_features,))

    self.params = {'weight': gaussian, 'bias': bias}
    self.grads = {'weight': zeros, 'bias': bias}
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.out = x
    out = np.dot(x, self.params['weight']) + self.params['bias']

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to
    layer parameters in self.grads['weight'] and self.grads['bias'].
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.grads['weight'] = np.dot(dout, self.out.T)
    self.grads['bias'] = np.sum(dout, axis = 0) # sum over columns, so axis = 0
    dx = np.dot(dout, self.grads['weight'].T)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.

    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = x[x>=0]
    self.out = out
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = dout
    dx[self.out<=0] = 0
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    b = np.amax(x, axis=1) # maximum along row, rows indicate the classes
    # reshape to vector?
    y = np.exp(x-b)
    out = y / np.sum(y, axis=1)
    self.out = out
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    s = dout.reshape(-1,1)
    dx = np.diagflat(s) - np.dot(s, s.T) # not etirely sure, retrieved from https://medium.com/@aerinykim/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    TODO:
    Implement forward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = - np.sum(y * np.log(x))/ x.shape[0]
    self.out = out
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = - np.divide(y, x) / x.shape[0]
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
