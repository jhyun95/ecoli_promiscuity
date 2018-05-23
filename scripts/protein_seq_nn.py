#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 16:03:54 2018

@author: jhyun95
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

''' Basic RNN architecture taken from: 
    https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html '''

SEQ_FILE = '../data/protein_sequences.faa'
MODEL_FILE = '../data/iML1515.json'
GENE_GROUPS_FILE = '../data/gene_groups_no_transport.csv'

AA_CODES = 'GALMFWKQESPVCIYHRNDT'

def main():
#    print(sequence_to_tensor('EPNPAYETLMNAVKLVREQKVTF'))
    enzyme_data = load_enzyme_sequences_and_promiscuity()
    enzyme_data = list(enzyme_data.values())
    print('Loaded sequences for', len(enzyme_data), 'enzymes.')
    model = SimpleRNN(len(AA_CODES), hidden_size=20, output_size=2)

    criterion = nn.NLLLoss()
    print('Initialized RNN model')
    
    for i in range(10):
        print("EPOCH", i+1)    
        train_epoch_rnn(model, criterion, enzyme_data, split=0.9)
        
    torch.save(model.state_dict(), '../torch_models/SimpleRNN_E10')
#    model = SimpleRNN(len(AA_CODES), hidden_size=20, output_size=2)
#    model.load_state_dict(torch.load('../torch_models/SimpleRNN_E10'))
    
def train_epoch_rnn(rnn, criterion, data, lr=0.005, split=0.9):
    ''' Single epoch, splits into train/test, trains on all training data '''
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    train_size = int(len(data)*split)
    train_set = indices[:train_size]
    test_set = indices[train_size:]
    
    ''' Training steps '''
    print('Training...')
    current_loss = 0.0; count = 1
    for i in train_set:
        seq, prom = data[i] 
        output, loss = train_rnn(rnn, criterion, seq, prom, lr)
        current_loss += loss
        if count % 50 == 0:
            print('Trained', count, 'of', str(train_size)+'.', 
                  'Average Loss:', current_loss.data[0] / count)
        count += 1
        
    ''' Test steps '''
    print('Testing...')
    correct = 0; test_size = len(data) - train_size
    for i in test_set:
        seq, prom = data[i]
        hidden = Variable(rnn.init_hidden())
        seq_tensor = Variable(sequence_to_tensor(seq))
        for j in range(len(seq)):
            output, hidden = rnn.forward(seq_tensor[j], hidden)
        prom_predict = output.max(1)[1].data[0]
        correct += (int(prom) == prom_predict)
    print('Accuracy:', correct, 'of', test_size, 
          '(' + str(correct/test_size) + ')')
        
def train_rnn(rnn, criterion, sequence, is_promiscuous, lr=0.005):
    ''' Single training step for RNN '''
    rnn.zero_grad()
    seq_tensor = Variable(sequence_to_tensor(sequence))
    prom_tensor = Variable(torch.LongTensor([int(is_promiscuous)]))
    hidden = Variable(rnn.init_hidden())
    
    for i in range(seq_tensor.size()[0]): # run sequence through RNN
        output, hidden = rnn(seq_tensor[i], hidden)    
    loss = criterion(output, prom_tensor) # update loss function and gradient
    loss.backward()    
    for p in rnn.parameters(): # update parameters
        p.data.add_(-lr, p.grad.data)
    return output, loss
    
def sequence_to_tensor(sequence):
    ''' Converts sequence to one-hot pytorch tensor '''
    n = len(sequence)
    seq_tensor = torch.zeros(n,1,len(AA_CODES))
    for i in range(n):
        a = AA_CODES.find(sequence[i])
        seq_tensor[i,0,a] = 1
    return seq_tensor

def load_enzyme_sequences_and_promiscuity(gene_groups_file=GENE_GROUPS_FILE,
                                          seq_file=SEQ_FILE):
    ''' Load amino acid sequences '''
    sequence = ""; gene = None
    gene_sequences = {}
    for line in open(seq_file, 'r+'):
        if '>' == line[0]: # header line
            if sequence != "":
                gene_sequences[gene] = sequence
            _, uniprot, gene, desc = line.split('|')
            sequence = ""
        else: # sequence line
            sequence += line.strip()
    gene_sequences[gene] = sequence
            
    ''' Load enzyme gene lists and map to sequences, promisucity 
        Multigene enzymes have sequences concatenated '''
    df = pd.read_csv(gene_groups_file)
    rows, cols = df.shape
    enzyme_data = {}
    for i in range(rows):
        gene_group = df.loc[i]['gene_group'].split(';')
        gene_group = tuple(gene_group)
        reactions = df.loc[i]['associated_reactions'].split(';')
        promiscuity = len(reactions) > 1
        sequence = ""
        try:
            for gene in gene_group:
                sequence += gene_sequences[gene]
            enzyme_data[gene_group] = (sequence, promiscuity)
        except KeyError:
            print(gene_group, 'has unknown gene', gene)
    return enzyme_data

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        ''' Simplest possible RNN model, single-layer, output activator,
            predicts the probability of a binary output '''
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size # cell state dimension
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) # input + cell state -> next state
        self.i2o = nn.Linear(input_size + hidden_size, output_size) # input + cell state -> output
        self.softmax = nn.LogSoftmax(dim=1) # output -> observed probability

    def forward(self, x, hidden):
        combined = torch.cat([x, hidden], 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        ''' Initailize the hidden cell state '''
        return torch.zeros(1, self.hidden_size)
    
if __name__ == '__main__':
    main()