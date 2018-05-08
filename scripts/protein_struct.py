# -*- coding: utf-8 -*-
"""
Created on Mon May  7 19:01:48 2018

@author: jhyun_000
"""

import scipy.sparse
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
# For viz, needs pygraphviz
# https://stackoverflow.com/questions/40528048/pip-install-pygraphviz-no-package-libcgraph-found
# pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" 
# --install-option="--library-path=/usr/lib/graphviz/"

MODEL_FILE = '../data/iML1515.json'
GEMPRO_FILE = '../data/iML1515-GEMPro.csv'
CATH_FILE = '../data/cath-domain-list.txt'
GENE_GROUPS_FILE = '../data/gene_groups_no_transport.csv'
CATH_OUT_FILE = '../data/gene_groups_with_cath.csv'

def main():
#    print('MAPPING ENZYMES TO CATH DOMAINS')
#    map_enzymes_to_cath()
    visualize_cath_hierarchy(depth=4)
    
def visualize_cath_hierarchy(gene_groups_file=CATH_OUT_FILE, depth=4):
    ''' Plots the CATH domain hierarchy for enzymes with CATH annotations.
        Redder nodes = associated with more promiscuous enzymes,
        Bluer nodes = associated with more specific enzymes, 
        Node labels = number of enzymes mapped to the node '''
    
    ''' Loads enzmye CATH annotations '''
    print('Loading CATH annotations...')
    df = pd.read_csv(gene_groups_file)
    enz_to_cath = {}; enz_to_prom = {}
    rows, cols = df.shape
    for i in range(rows):
        gene_group = tuple(df.loc[i]['gene_group'].split(';'))
        promiscuity = len(df.loc[i]['associated_reactions'].split(';'))
        enz_to_prom[gene_group] = promiscuity
        cath_classes = df.loc[i]['associated_cath_classes']
        if not pd.isnull(cath_classes):
            cath_classes = cath_classes.split(';')
            cath_classes = map(lambda x: '.'.join(x.split('.')[:depth]), cath_classes)
            cath_classes = tuple(cath_classes)
            enz_to_cath[gene_group] = cath_classes
    print('    Loaded annotations for', len(enz_to_cath), 'enzymes.')
    
    ''' Construct CATH hierarchy and assign enzymes to nodes '''
    print('Constructing CATH hierarchy graph...')
    enz_order = list(enz_to_cath.keys()) # order enzymes for index IDs to save memory
    cath_graph = {'ROOT':[]}; cath_node_to_enz = {'ROOT':set()}
    for enz in enz_to_cath:
        enzID = enz_order.index(enz)
        cath_classes = enz_to_cath[enz]
        for cath in cath_classes:
            for i in range(depth-1):
                hierarchy = cath.split('.')
                parent = '.'.join(hierarchy[:i+1])
                child = '.'.join(hierarchy[:i+2])
                # Update enzymes assigned to each node
                if not parent in cath_node_to_enz:
                    cath_node_to_enz[parent] = set()
                if not child in cath_node_to_enz:
                    cath_node_to_enz[child] = set()
                cath_node_to_enz['ROOT'].add(enzID)
                cath_node_to_enz[parent].add(enzID)
                cath_node_to_enz[child].add(enzID)
                # Update graph connections
                if i == 0 and not parent in cath_graph['ROOT']: # highest level
                    cath_graph['ROOT'].append(parent)
                if not parent in cath_graph:
                    cath_graph[parent] = [child]
                elif not child in cath_graph[parent]:
                    cath_graph[parent].append(child)
                if not child in cath_graph:
                    cath_graph[child] = []
    print('    Constructed graph with', len(cath_graph), 'nodes.')
    
    ''' Convert adjancency list to sparse matrix '''
    print('Constructing adjacency matrix...')
    n = len(cath_graph)
    adj = scipy.sparse.lil_matrix((n,n))
    node_order = list(cath_graph.keys())
    for i in range(n):
        parent = node_order[i]
        for child in cath_graph[parent]:
            j = node_order.index(child)
            adj[i,j] = 1; adj[j,i] = 1 # undirected
    
    ''' Generate graph visualization '''
    G = nx.from_scipy_sparse_matrix(adj)
    color_values = []; labels = {}
    for i in range(n):
        node = node_order[i]
        enzymes = cath_node_to_enz[node]
        count = len(enzymes); promiscuous = 0.0
        for enzID in enzymes:
            enz = enz_order[enzID]
            promiscuous += int(enz_to_prom[enz] > 1)
        color_values.append(promiscuous/count)
        labels[i] = count
    
    nsize = 50; fsize = 8
    try: # if pygraphviz is available
        layout = nx.drawing.nx_agraph.graphviz_layout(G, prog='twopi')
        nx.draw(G, pos=layout, node_size=nsize, 
                cmap=plt.get_cmap('coolwarm'), node_color=color_values,
                with_labels=True, labels=labels, font_size=fsize)    
    except ImportError:
        print('No pygraphviz, using spring layout')
        nx.draw_spring(G, node_size=nsize,
                       cmap=plt.get_cmap('coolwarm'), node_color=color_values,
                       with_labels=True, labels=labels, font_size=fsize)  
    
def map_enzymes_to_cath(gene_group_file=GENE_GROUPS_FILE, 
                        gem_pro_file=GEMPRO_FILE,
                        cath_file=CATH_FILE,
                        cath_output_file=CATH_OUT_FILE):
    ''' Maps enzymes (as a list of genes) to a set of CATH domains. Separates 
        into complete (all genes in group have PDBs and all PDBs have CATH 
        domains) to incomplete (incomplete gene -> PDB or PDB -> CATH). '''
    enz_to_pdb, enz_to_pdb_incomplete, unique_pdbs = map_enzymes_to_pdb(); print("")
    pdb_to_cath, cath_data = get_pdb_to_cath(unique_pdbs); print("")
    
    complete = 0; incomplete = 0; absent = 0
    enz_to_cath = {}; enz_to_cath_incomplete = {}
    for enz in list(enz_to_pdb.keys()) + list(enz_to_pdb_incomplete.keys()):
        cath_domains = []; is_complete = enz in enz_to_pdb
        pdbs = enz_to_pdb[enz] if is_complete else enz_to_pdb_incomplete[enz]
        for pdb in pdbs:
            if pdb in pdb_to_cath:
                cath = pdb_to_cath[pdb]
                cath_domains += cath
            else:
                is_complete = False
        if is_complete and len(cath_domains) > 0:
            complete += 1
            enz_to_cath[enz] = tuple(cath_domains)
        elif len(cath_domains) > 0:
            incomplete += int(len(cath_domains) > 0)
            enz_to_cath_incomplete[enz] = tuple(cath_domains)
        else:
            absent += int(len(cath_domains) == 0)
                
    n = len(enz_to_pdb) + len(enz_to_pdb_incomplete)
    print('Total Enzymes with PDB annotations:', n)
    print('Enzymes with complete CATH domain annotation:', complete)
    print('Enzymes with partial CATH domain annotation:', incomplete)
    print('Enzymes with no CATH domain annotation:', absent)
    
    if cath_output_file: # integrate with gene groups file
        df = pd.read_csv(gene_group_file)
        df.rename(columns={'Unnamed: 0':''}, inplace=True)
        rows, cols = df.shape
        pdb_mapping = []; cath_mapping = []; cath_code_mapping = []
        for i in range(rows):
            gg = tuple(df.loc[i]['gene_group'].split(';'))
            pdbs = enz_to_pdb[gg] if gg in enz_to_pdb else []
            caths = enz_to_cath[gg] if gg in enz_to_cath else []
            cath_codes = map(lambda x: cath_data[x], caths)
            pdbs = ';'.join(pdbs); pdb_mapping.append(pdbs)
            caths = ';'.join(caths); cath_mapping.append(caths)
            cath_codes = ';'.join(cath_codes); cath_code_mapping.append(cath_codes)
        df['associated_pdb'] = pdb_mapping
        df['associated_cath_classes'] = cath_code_mapping
        df['associated_cath_domains'] = cath_mapping
        df.to_csv(cath_output_file, index=False)
    return enz_to_cath, enz_to_cath_incomplete, cath_data
    
def map_enzymes_to_pdb(gene_group_file=GENE_GROUPS_FILE, gem_pro_file=GEMPRO_FILE):
    ''' Maps enzymes (as a list of genes) to a set of associated PDB IDs.
        Separates into complete (all genes in group have PDBs) to incomplete 
        (some or no genes in gene group have PDBs). Also returns a set of
        unique PDBs. '''
    gene_to_pdb, unique_pdbs = get_gene_to_pdb(gem_pro_file)
    gene_groups = get_gene_groups(gene_group_file)
    enz_to_pdb = {}; enz_to_pdb_incomplete = {}
    complete = 0; incomplete = 0; absent = 0
    for gg in gene_groups:
        pdbs = set(); is_complete = True
        for gene in gg:
            if gene in gene_to_pdb:
                pdbs.add(gene_to_pdb[gene])
            else:
                is_complete = False
        if is_complete and len(pdbs) > 0:
            complete += 1
            enz_to_pdb[tuple(gg)] = pdbs
        elif len(pdbs) > 0:
            incomplete += 1            
            enz_to_pdb_incomplete[tuple(gg)] = pdbs
        else: 
            absent += int(len(pdbs) == 0)
    
    print('Total Enzymes:', len(gene_groups))
    print('Enzymes with complete PDB annotations:', complete)
    print('Enzymes with partial PDB annotations:', incomplete)
    print('Enzymes with no PDB annotations:', absent)
    return enz_to_pdb, enz_to_pdb_incomplete, unique_pdbs

def get_pdb_to_cath(pdbs, cath_file=CATH_FILE):
    ''' Maps pdbs to a list of known CATH domains. Also extracts hierarchical
        data for those CATH domains. '''
    pdb_to_cath = {}
    cath_data = {}
    f = open(cath_file, 'r+')
    for line in f:
        if line[0] != '#': # not a comment line
            entry = line.split()
            domain = entry[0]; data = entry[1:]
            domain_pdb = domain[:4]
            if domain_pdb in pdbs:
                if not domain_pdb in pdb_to_cath:
                    pdb_to_cath[domain_pdb] = []
                pdb_to_cath[domain_pdb].append(domain)
                cath_class = '.'.join(data[:-1])
#                cath_res = float(data[-1])
                cath_data[domain] = cath_class
    print('PDBs with CATH domains:', len(pdb_to_cath))
    print('Total CATH domains loaded:', len(cath_data))
    return pdb_to_cath, cath_data

def get_gene_to_pdb(gempro_file=GEMPRO_FILE):
    ''' Maps gene locus tags to PDB IDs '''
    df = pd.read_csv(GEMPRO_FILE)
    rows, cols = df.shape
    gene_to_pdb = {}
    unique_pdbs = set()
    for i in range(rows):
        gene = df.loc[i]['m_gene']
        pdb = df.loc[i]['struct_pdb']
        if not pd.isnull(pdb):
            pdb = pdb.lower()
            gene_to_pdb[gene] = pdb
            unique_pdbs.add(pdb)
    print('Genes with PDBs:', len(gene_to_pdb))
    print('Unique PDBs:', len(unique_pdbs))
    return gene_to_pdb, unique_pdbs

def get_gene_groups(gene_groups_file=GENE_GROUPS_FILE):
    ''' Loads enzymes as groups of genes '''
    df = pd.read_csv(gene_groups_file)
    rows, cols = df.shape
    gene_groups = []
    for i in range(rows):
        genes = df.loc[i]['gene_group'].split(';')
        gene_groups.append(genes)
    return gene_groups 
        
if __name__ == '__main__':
    main()