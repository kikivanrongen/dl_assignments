# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import random
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from dataset import TextDataset
from model import TextGenerationModel

################################################################################
def train(config):
    # empty file to write generated text to
    with open('generated.txt', 'w'): pass

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, \
                                lstm_num_hidden=config.lstm_num_hidden, device=device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        batch_inputs = torch.stack(batch_inputs)
        embedding = one_hot(batch_inputs, dataset._vocab_size)

        batch_targets = torch.stack(batch_targets)

        h_0 = torch.zeros(config.lstm_num_layers, config.batch_size, config.lstm_num_hidden)
        c_0 = torch.zeros(config.lstm_num_layers, config.batch_size, config.lstm_num_hidden)
        output = model.forward(embedding, h_0, c_0)
        optimizer.zero_grad()

        losses, accuracies = [], []
        for i, out in enumerate(output):
            label = batch_targets[i,:]

            loss = criterion(out, label)
            losses.append(loss)

            accuracy = (torch.max(out, 1)[1] == label).float().mean()
            accuracies.append(accuracy)

        loss = sum(losses) / len(losses)
        accuracy = sum(accuracies) / len(accuracies)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optimizer.step()


        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:

            train_step = float(step) / float(config.train_steps)
            print("[{}] Train Step {:.0f}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), train_step*1000000, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step % config.sample_every == 0:
            sample(dataset, model)
            #sample2(dataset, model)

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')

def sample(dataset, model):

    start_letters = random.choices(dataset._chars, k=10)
    h_0 = torch.zeros(config.lstm_num_layers, 1, config.lstm_num_hidden)
    c_0 = torch.zeros(config.lstm_num_layers, 1, config.lstm_num_hidden)

    with open('generated.txt', 'a') as generated_file:

        for letter in start_letters:
            idx = dataset._char_to_ix[letter]
            input = one_hot_char(idx, dataset.vocab_size)
            input = input.reshape(1, 1, dataset.vocab_size)
            sentence = [letter]
            next_letter = letter
            for i in range(config.seq_length):
                output = model(input, h_0, c_0)
                output = F.softmax(output[-1, :, :]).detach().cpu().numpy()
                output = output[0]

                # higher temperature means more random sentences
                output = np.log(output) / config.temperature
                torch.set_printoptions(precision=8)
                output = torch.from_numpy(output).float()
                output = F.softmax(output).cpu().detach().numpy()

                next_letter = np.random.choice(range(output.shape[0]), p=output)
                letter = dataset._ix_to_char[next_letter]

                sentence += letter

                inputs = one_hot_char(next_letter, dataset.vocab_size)
                inputs = inputs.reshape(1, 1, dataset.vocab_size)
                input = torch.cat((input, inputs), dim=0)
            char_sentence = ''.join(sentence)
            generated_file.write(char_sentence + '\n')
        generated_file.write('\n')
    generated_file.close()

def one_hot(batch,depth):
    emb = nn.Embedding(depth, depth)
    emb.weight.data = torch.eye(depth)
    return emb(batch)

def one_hot_char(input, vocabulary_size):
    emb = torch.zeros(vocabulary_size)
    emb[input] = 1
    return emb

 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')
    parser.add_argument('--temperature', type=float, default=1.0)

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)
