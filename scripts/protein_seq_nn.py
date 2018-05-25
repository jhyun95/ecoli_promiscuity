#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 16:03:54 2018

@author: jhyun95
"""

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
torch.set_num_threads(1)

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
    
    ''' Pytorch built in RNNs, set length limit to <= 800 '''
    model = nn.LSTM(input_size=len(AA_CODES), hidden_size=2, num_layers=1)
#    model = BinaryLSTM(input_size=len(AA_CODES), hidden_size=5)
    train_epochs_lstm(model, enzyme_data, split=0.9, epochs=10)
    name = 'LSTM_H2_E10'
    torch.save(model.state_dict(), '../torch_models/'+name)
    
    ''' Simple RNN example, set length limit to <=1000 '''
#    model = SimpleRNN(len(AA_CODES), hidden_size=5, output_size=2)
#    print('Initialized RNN model')
#    model = train_epochs_simple_rnn(model, enzyme_data, split=0.9)    
#    name = 'SimpleRNN_H5_E10'
#    torch.save(model.state_dict(), '../torch_models/'+name)
    
#    model.load_state_dict(torch.load('../torch_models/'+name))
    
def train_epochs_lstm(model, data, lr=0.05, split=0.9, epochs=10):
    optimizer = optim.Adam(model.parameters(),lr=lr)
    for epoch in range(1,epochs+1):
        print("EPOCH", epoch)
        ''' Per epoch, splits into train/test, trains on all training data '''
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        train_size = int(len(data)*split)
        train_set = indices[:train_size]
        test_set = indices[train_size:]
        
        ''' Training steps '''
        print('Training...'); model.train()
        current_loss = 0.0; count = 1
        
        for i in train_set:
            seq, prom = data[i] 
            seq_tensor = sequence_to_tensor(seq)
            output, hidden = model(seq_tensor)
                
            optimizer.zero_grad()
            prom_tensor = torch.LongTensor([int(prom)])
            last_output = output[-1] # for nn.LSTM
#            last_output = output # for BinaryLSTM
#            loss = nn.functional.nll_loss(last_output, prom_tensor) # NLL loss is negative? bug
            loss = nn.functional.cross_entropy(last_output, prom_tensor)
            loss.backward()
            current_loss += loss
            optimizer.step()
            
            if count % 50 == 0:
                print('Trained', count, 'of', str(train_size)+'.', 
                      'Average Loss:', current_loss.item() / count)
            count += 1
            
        ''' Test steps '''
        print('Testing...'); model.eval()
        correct = 0; test_size = len(data) - train_size
        for i in test_set:
            seq, prom = data[i]
            seq_tensor = sequence_to_tensor(seq)
            output, hidden = model(seq_tensor)
            prom_predict = output[-1].max(1)[1].item() # for nn.LSTM
#            prom_predict = output.max(1)[1].item() # for BinaryLSTM
            correct += (int(prom) == prom_predict)
        print('Accuracy:', correct, 'of', test_size, 
              '(' + str(correct/test_size) + ')')
        
    return model
    
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
    
def sequence_to_tensor(sequence):
    ''' Converts sequence to one-hot pytorch tensor '''
    n = len(sequence)
    seq_tensor = torch.zeros(n,1,len(AA_CODES))
    for i in range(n):
        a = AA_CODES.find(sequence[i])
        seq_tensor[i,0,a] = 1
    return seq_tensor

def load_enzyme_sequences_and_promiscuity(gene_groups_file=GENE_GROUPS_FILE,
                                          seq_file=SEQ_FILE, len_limit=1000):
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
            if len(sequence) <= len_limit:
                enzyme_data[gene_group] = (sequence, promiscuity)
        except KeyError:
            print(gene_group, 'has unknown gene', gene)
    return enzyme_data

def annotate_metabolites(gene_group_file='../data/gene_groups_with_cath.csv', 
                         model_file='../data/iML1515.json',
                         out_file='../data/gene_groups_no_transport_extended.csv'):
    ''' Annotates the gene group/enzyme file with known substrates 
        and products according to the provided metabolic model. '''
    import cobra.io
    model = cobra.io.load_json_model(model_file)
    df = pd.read_csv(gene_group_file)
    rows, cols = df.shape
    substrates = []; products = []
    
    for i in range(rows):
        reactionIDs = df.loc[i]['associated_reactions'].split(';')
        enzyme_substrates = set(); enzyme_products = set()
        for rxnID in reactionIDs:
            rxn = model.reactions.get_by_id(rxnID)
            is_reversible = rxn.reversibility
            for met in rxn.metabolites:
                if rxn.metabolites[met] < 0 or is_reversible: # substrate
                    enzyme_substrates.add(met.id)
                if rxn.metabolites[met] > 0 or is_reversible: # product
                    enzyme_products.add(met.id)
        enzyme_substrates = ';'.join(enzyme_substrates)
        enzyme_products = ';'.join(enzyme_products)
        substrates.append(enzyme_substrates)
        products.append(enzyme_products)
    df['substrates'] = substrates
    df['products'] = products
    df = df.rename({'Unnamed: 0':''}, axis='columns')
    df.to_csv(out_file, sep=',', index=0)
    return df

class BinaryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        ''' Single cell LSTM with hidden_size dimension for LSTM output,
            which is then passed to a linear layer and reduced to 2 outputs '''
        super(BinaryLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.fc = nn.Linear(in_features=hidden_size, out_features=2)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        output, hidden = self.lstm(x)
        output = output[-1].view(1,1,self.hidden_size) # last output
        output = self.fc(output).view(1,2)
        output = self.softmax(output)
        return output, hidden

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