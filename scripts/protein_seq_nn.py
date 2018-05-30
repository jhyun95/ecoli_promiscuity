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

''' Basic RNN architecture taken from: 
    https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html '''

SEQ_FILE = '../data/protein_sequences.faa'
MODEL_FILE = '../data/iML1515.json'
GENE_GROUPS_FILE = '../data/gene_groups_no_transport.csv'

AA_CODES = 'GALMFWKQESPVCIYHRNDT'

def main():
#    sequence_to_fuzzy_tensors('MKVKVLSLLVPALLVAGAANAAEVYNKDGNKLDLYGKVDGLHYFSDNKDVDGDQTYMRLG')
    ''' Load sequences. Limit maximum sequence length for performance issues '''
    enzyme_data = load_enzyme_sequences_and_promiscuity(len_limit=720)
    enzyme_data = list(enzyme_data.values())
    print('Loaded sequences for', len(enzyme_data), 'enzymes.')
    
    ''' Pytorch built in RNNs, set length limit to <= 800 '''
    model = nn.LSTM(input_size=len(AA_CODES), hidden_size=2, num_layers=1)
#    model = BinaryLSTM(input_size=len(AA_CODES), hidden_size=5)
    train_epochs_lstm(model, enzyme_data, resolution=5, split=0.9, epochs=10)
    name = 'LSTM_H2_E10_R5'
    torch.save(model.state_dict(), '../torch_models/'+name)
    
def train_epochs_lstm(model, data, resolution=3, lr=0.05, split=0.9, epochs=10):
    optimizer = optim.Adam(model.parameters(),lr=lr)
    for epoch in range(1,epochs+1):
        print("EPOCH", epoch)
        ''' Per epoch, splits into train/test, trains on all training data '''
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        train_enzyme_size = int(len(data)*split) # number of enzymes
        test_enzyme_size = len(data) - train_enzyme_size
        train_size = train_enzyme_size * resolution # number of generated data points
        test_size = test_enzyme_size * resolution
        train_set = indices[:train_enzyme_size]
        test_set = indices[train_enzyme_size:]
        
        ''' Training steps '''
        print('Training...'); model.train()
        current_loss = 0.0; count = 1
        for i in train_set:
            seq, prom = data[i] 
            prom_tensor = torch.LongTensor([int(prom)])
            for seq_tensor in sequence_to_fuzzy_tensors(seq, resolution=resolution, normalize=True):
                optimizer.zero_grad()
                output, hidden = model(seq_tensor)
                last_output = output[-1] # for nn.LSTM
#                last_output = output # for BinaryLSTM
#                loss = nn.functional.nll_loss(last_output, prom_tensor) # NLL loss is negative? bug
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
        correct = 0; actual_correct = 0
        for i in test_set:
            seq, prom = data[i]
            fuzzy_predictions = [0,0] # predictions for each window
            for seq_tensor in sequence_to_fuzzy_tensors(seq, resolution=resolution, normalize=True):
                output, hidden = model(seq_tensor)
                prom_predict = output[-1].max(1)[1].item() # for nn.LSTM
#               prom_predict = output.max(1)[1].item() # for BinaryLSTM
                correct += (int(prom) == prom_predict)
                fuzzy_predictions[prom_predict] += 1
            if fuzzy_predictions[int(prom)] > fuzzy_predictions[1-int(prom)]: # consensus prediction
                actual_correct += 1
        print('Accuracy:', correct, 'of', test_size, 
              '(' + str(correct/test_size) + ')')
        print('Consensus Accuracy:', actual_correct, 'of', test_enzyme_size, 
              '(' + str(actual_correct/test_enzyme_size) + ')')
        
    return model
    
def sequence_to_tensor(sequence):
    ''' Converts sequence to one-hot pytorch tensor '''
    n = len(sequence)
    seq_tensor = torch.zeros(n,1,len(AA_CODES))
    for i in range(n):
        a = AA_CODES.find(sequence[i])
        seq_tensor[i,0,a] = 1
    return seq_tensor

def sequence_to_fuzzy_tensors(sequence, resolution=3, normalize=True):
    ''' Converts sequence to a set of "fuzzy" one-hot pytorch tensors '''
    n = len(sequence); seq_tensors = []
    for i in range(resolution):
        num_windows = int(np.floor((n - i) / resolution))
        ohe_array = np.zeros((num_windows, len(AA_CODES)), dtype=np.float)
        for j in range(num_windows):
            window = sequence[i+j*resolution:i+(j+1)*resolution]
            for aa in list(window):
                ind = AA_CODES.find(aa)
                ohe_array[j,ind] += 1
        if normalize: # row euclidean norms to 1
            row_sums = np.linalg.norm(ohe_array, axis=1)
            ohe_array = ohe_array / row_sums[:, np.newaxis]
        seq_tensor = torch.from_numpy(ohe_array).float()
        seq_tensor = seq_tensor.view(num_windows,1,len(AA_CODES))
        seq_tensors.append(seq_tensor)
    return seq_tensors

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

if __name__ == '__main__':
    main()