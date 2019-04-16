"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.

  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch

  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  preds = torch.max(predictions,1)[1]
  tags = torch.max(targets,1)[1]

  return (preds == tags).float().mean()

def train():
  """
  Performs training and evaluation of MLP model.

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  # prepare input data
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  _, width, height, channels = cifar10['train']._images.shape
  _, n_outputs =  cifar10['train']._labels.shape
  n_inputs = width * height * channels

  network = MLP(n_inputs,dnn_hidden_units,n_outputs)

  optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate) # or SGD?
  loss_fn = nn.CrossEntropyLoss()

  train_losses, train_acc, test_losses, test_acc = [], [], [], []
  current_loss = 0.0

  for step in range(FLAGS.max_steps):
      network.train()
      optimizer.zero_grad()

      x, y = cifar10['train'].next_batch(FLAGS.batch_size)
      x, y = torch.tensor(x, requires_grad=True), torch.tensor(y, dtype=torch.float)
      x = x.reshape(FLAGS.batch_size,-1)

      output = network(x)
      labels = torch.max(y,1)[1]

      loss = loss_fn(output, labels)
      loss.backward()
      optimizer.step()
      current_loss += loss.item()

      if (step+1) % FLAGS.eval_freq == 0:
          train_acc.append(accuracy(output, y))
          train_losses.append(current_loss / float(FLAGS.eval_freq))
          current_loss = 0.0


          x_test, y_test = cifar10['test'].next_batch(FLAGS.batch_size)
          x_test, y_test = torch.tensor(x_test, requires_grad=True), torch.tensor(y_test, dtype=torch.float)
          x_test = x_test.reshape(FLAGS.batch_size, -1)

          output_test = network(x_test)

          # average loss over 100 iterations

          test_losses.append(loss_fn(output_test, torch.max(y_test,1)[1]).item())
          test_acc.append(accuracy(output_test, y_test))

          print("Step {}".format(step))

  size_test = cifar10['test']._num_examples
  x, y = cifar10['test'].next_batch(size_test)
  x, y = torch.tensor(x, requires_grad=True), torch.tensor(y, dtype=torch.float)
  x = x.reshape(size_test, -1)

  # Get network output for batch and get loss and accuracy
  out = network(x)
  print("Accuracy: {}".format(accuracy(out, y)))

  # plot graph of accuracies
  plt.subplot(211)
  plt.plot(test_acc, label="test accuracy")
  plt.plot(train_acc, label="training accuracy")
  plt.title('Accuracy')
  plt.legend()

  plt.subplot(212)
  plt.plot(test_losses, label = "test loss")
  plt.plot(train_losses, label = "training loss")
  plt.title('Cross-entropy loss')
  plt.legend()

  plt.show()


  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
