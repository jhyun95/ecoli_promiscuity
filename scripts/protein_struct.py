# -*- coding: utf-8 -*-
"""
Created on Mon May  7 19:01:48 2018

@author: jhyun_000
"""

import pandas as pd
import matplotlib.pyplot as plt

MODEL_FILE = '../data/iML1515.json'
GEMPRO_FILE = '../data/iML1515-GEMPro.csv'
CATH_FILE = '../data/cath-domain-list.txt'
GENE_GROUPS_FILE = '../data/gene_groups_no_transport.csv'

def main():
    print('MAPPING ENZYMES TO CATH DOMAINS')
    enz_to_cath, enz_to_cath_incomplete, cath_data = map_enzymes_to_cath()
    
def map_enzymes_to_cath(gene_group_file=GENE_GROUPS_FILE, 
                        gem_pro_file=GEMPRO_FILE,
                        cath_file=CATH_FILE):
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
                cath_domains.append(cath)
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
                for i in range(len(data)-1): # classes are integers
                    data[i] = int(data[i]) 
                data[-1] = float(data[i]) # resolution is float
                cath_data[domain] = data
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