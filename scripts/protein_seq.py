#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 09:30:54 2018

@author: jhyun95
"""

import urllib
import pandas as pd
import matplotlib.pyplot as plt

SEQ_FILE = '../data/protein_sequences.faa'
MODEL_FILE = '../data/iML1515.json'
GEMPRO_FILE = '../data/iML1515-GEMPro.csv'

UNIPROT_UNANNOTATED = {'b1097':'P28306'} # missing annotation in GEMPro
UNIPROT_FIXES = {'P16683':'A0A140NFA3'} # replace obsolete entries
''' P16683 now maps to 2 genes (PhnE1/PhnE2) in K12, 1 gene in BL21-DE3
    (PhnE). For simplicity, I use the BL21-DE3 sequence, which is similar 
    to the concatenation of the two sequences from K12. '''

def main():
#    get_sequences_from_uniprot()
#    get_gene_group_size_distr()
#    plot_seq_length_distr()
    get_protein_domains()
    
def plot_seq_length_distr(gene_groups_file='../data/gene_groups_no_transport.csv',
                          sequence_file='../data/protein_sequences.faa'):
    ''' Plots the distribution of sequence lengths enzymes inferred 
        from iML1515 GPRs. For multi-gene enzymes, combines lengths. '''
        
    ''' Get lengths for each gene '''
    gene_aa_lengths = {'s0001':0} #s0001 has no annotation, ignore
    f = open(sequence_file,'r+')
    sequence = ""; current_gene = ""
    for line in f:
        if '>' in line: # header
            if len(current_gene) > 0:
                gene_aa_lengths[current_gene] = len(sequence)
            _, uniprot, current_gene, desc = line[1:].split("|")
            sequence = ""
        else: # sequence
            sequence += line.strip()
    gene_aa_lengths[current_gene] = len(sequence)
    f.close()
    
    ''' Get enzymes -> genes -> lengths, and generate histogram '''
    df = pd.read_csv(gene_groups_file)
    rows, cols = df.shape
    gg_lengths_single = [] # single gene enzymes
    gg_lengths_multiple = [] # multiple gene enzymes
    for i in range(rows):
        gene_group = df.loc[i]['gene_group'].split(';')
        length = sum(map(lambda x: gene_aa_lengths[x], gene_group))
        if len(gene_group) == 1 and length > 0:
            gg_lengths_single.append(length)
        elif length > 0:
            gg_lengths_multiple.append(length)
    print(min(gg_lengths_single), max(gg_lengths_single))
    plt.hist(gg_lengths_single, bins=range(0,5100,100), alpha=0.8, label='single gene')
    plt.hist(gg_lengths_multiple, bins=range(0,5100,100), alpha=0.8, label='multi-gene')
    plt.title('Distribution of AA lengths for iML1515 extracted enzymes')
    plt.xlabel('Length'); plt.ylabel('Frequency')
    plt.legend()
    
def get_gene_group_size_distr(gene_groups_file='../data/gene_groups_no_transport.csv'):
    ''' Returns distribution of #genes per enzyme inferred from iML1515 GPRs '''
    df = pd.read_csv(gene_groups_file)
    rows, cols = df.shape
    gg_sizes = {}
    for i in range(rows):
        gene_group = df.loc[i]['gene_group'].split(';')
        size = len(gene_group)
        if not size in gg_sizes:
            gg_sizes[size] = 0
        gg_sizes[size] += 1
    print(gg_sizes)
    return gg_sizes

def get_protein_domains(gempro_file=GEMPRO_FILE):
    ''' Loads domains for genes iML1515 '''
    df = pd.read_csv(gempro_file)
    rows, cols = df.shape
    unique_domains = set()
    genes_annotated = 0
    total_annotations = 0
    gene_to_domains = {}
    for i in range(rows):
        gene = df.loc[i]['m_gene']
        domains = df.loc[i]['seq_domains']
        domains = domains.split(';') if not pd.isnull(domains) else []
        if not gene in gene_to_domains and len(domains) > 0: # don't repeat
            genes_annotated += 1
            gene_to_domains[gene] = domains
            total_annotations += len(domains)
            for domain in domains:
                unique_domains.add(domain)
    print('Unique domains:', len(unique_domains))
    print('Domain annotations:', total_annotations)
    print('Genes annotated:', genes_annotated)

def get_sequences_from_uniprot(gempro_file=GEMPRO_FILE, output_file=SEQ_FILE):
    ''' Gets the amino acid sequence for each gene in iML1515 
        or other annotated GEM-PRO file '''
    
    ''' Load all genes and Uniprot annotations from iML1515 '''
    gene_to_uniprot = UNIPROT_UNANNOTATED
    df = pd.read_csv(gempro_file)
    rows, cols = df.shape
    for i in range(rows):
        gene = df.loc[i]['m_gene']
        uniprot = df.loc[i]['seq_uniprot']
        if not gene in gene_to_uniprot:
            gene_to_uniprot[gene] = uniprot
        elif gene_to_uniprot[gene] != uniprot:
            print('WARNING:', gene, 'has multiple Uniprot IDs')
    print('Loaded', len(gene_to_uniprot), 'genes with UniprotIDs.')
            
    ''' Check which sequences have already been loaded '''
    loaded_genes = set()
    f = open(output_file,'r+')
    for line in f:
        if '>' == line[0]: # check headers
            _, uniprot, gene, desc = line[1:].split('|')
            loaded_genes.add(gene)
    f.close()
    print('Found', len(loaded_genes), 'genes with sequences already.')
    
    ''' Pull remaining sequences from UniprotKB '''
    f = open(output_file,'a+'); counter = 1; n = len(gene_to_uniprot)
    for gene in gene_to_uniprot:
        print('Gene', counter, 'of', str(n)+":", gene+'...', end=' ')
        if not gene in loaded_genes:
            uniprot = gene_to_uniprot[gene]
            if uniprot in UNIPROT_FIXES:
                uniprot = UNIPROT_FIXES[uniprot]
            url = 'https://www.uniprot.org/uniprot/' + uniprot + '.fasta'
            data = urllib.request.urlopen(url) # should try-except URLError?
            for bytestring in data:
                line = bytestring.decode("utf-8") 
                if '>' in line: # insert locus tag into header
                    header = line.replace(uniprot,uniprot+"|"+gene)
                    f.write(header)
                else: # sequence data
                    f.write(line)
            print('Done.')
        else: # already loaded
            print('Already loaded.')
        counter += 1
    f.close()

if __name__ == "__main__":
    main()
