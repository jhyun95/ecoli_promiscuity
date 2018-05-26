# -*- coding: utf-8 -*-
"""
Created on Fri May 25 16:25:34 2018

@author: jhyun_000
"""

import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(1)
from protein_seq_nn import load_enzyme_sequences_and_promiscuity, sequence_to_tensor

''' Basic RNN architecture taken from: 
    https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html '''

SEQ_FILE = '../data/protein_sequences.faa'
MODEL_FILE = '../data/iML1515.json'
GENE_GROUPS_FILE = '../data/gene_groups_no_transport.csv'

AA_CODES = 'GALMFWKQESPVCIYHRNDT'

def main():
    ''' Load sequences. Limit maximum sequence length for performance issues '''
    enzyme_data = load_enzyme_sequences_and_promiscuity(len_limit=800)
    enzyme_data = list(enzyme_data.values())
    print('Loaded sequences for', len(enzyme_data), 'enzymes.')
    
    ''' Simple RNN example, set length limit to <=1000 '''
    model = SimpleRNN(len(AA_CODES), hidden_size=5, output_size=2)
    print('Initialized RNN model')
    model = train_epochs_simple_rnn(model, enzyme_data, split=0.9)    
    name = 'SimpleRNN_H5_E10'
    torch.save(model.state_dict(), '../torch_models/'+name)
    
#    model.load_state_dict(torch.load('../torch_models/'+name))
    
def train_epochs_simple_rnn(rnn, data, lr=0.005, split=0.9, epochs=10):
    for epoch in range(1,epochs+1):
        print("EPOCH", epoch)
        ''' Per epoch, splits into train/test, trains on all training data '''
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        train_size = int(len(data)*split)
        train_set = indices[:train_size]
        test_set = indices[train_size:]
        
        ''' Training steps '''
        print('Training...')
        rnn.train()
        current_loss = 0.0; count = 1
        for i in train_set:
            seq, prom = data[i] 
            output, loss = train_simple_rnn(rnn, seq, prom, lr)
            current_loss += loss
            if count % 50 == 0:
                print('Trained', count, 'of', str(train_size)+'.', 
                      'Average Loss:', current_loss.item() / count)
            count += 1
            
        ''' Test steps '''
        print('Testing...')
        rnn.eval()
        correct = 0; test_size = len(data) - train_size
        for i in test_set:
            rnn.zero_grad()
            seq, prom = data[i]
            hidden = rnn.init_hidden()
            seq_tensor = sequence_to_tensor(seq)
            for j in range(len(seq)):
                output, hidden = rnn(seq_tensor[j], hidden)
            prom_predict = output.max(1)[1].item()
            correct += (int(prom) == prom_predict)
        print('Accuracy:', correct, 'of', test_size, 
              '(' + str(correct/test_size) + ')')
    return rnn
        
def train_simple_rnn(rnn, sequence, is_promiscuous, lr=0.005):
    ''' Single training step for RNN, uses NLL loss '''
    rnn.zero_grad()
    seq_tensor = sequence_to_tensor(sequence)
    hidden = rnn.init_hidden()
    for i in range(seq_tensor.size()[0]): # run sequence through RNN
        output, hidden = rnn(seq_tensor[i], hidden)
    prom_tensor = torch.LongTensor([int(is_promiscuous)])
    loss = nn.functional.nll_loss(output, prom_tensor) # update loss function and gradient
    loss.backward() 
    for p in rnn.parameters(): # update parameters
        p.data.add_(-lr, p.grad.data)
    return output, loss

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