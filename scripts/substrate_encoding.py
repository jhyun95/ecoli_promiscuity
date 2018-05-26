# -*- coding: utf-8 -*-
"""
Created on Fri May 25 16:33:58 2018

@author: jhyun_000
"""

import os, shutil, urllib, tarfile
import pandas as pd
from rdkit.Chem.rdmolfiles import MolFromMolBlock
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintsFromMols

def main():
    ''' Generate very high-dimension hashing to identify non-resolvable
        clashes in hashing (for instance, fingerprinting cannot distinguish
        between different stereoisomers, or metal ion binding) '''
    print('FINDING UNRESOLVABLE FINGERPRINT CLASHES...')
    fp, allowable_clashes = fingerprint_metabolites(bits=8192, output_file=None)
    
    ''' Test a smaller, more reasonable fingerprinting strategy '''
    print('TESTING FINGERPRINTING STRATEGY...')
    fp, clashes = fingerprint_metabolites(bits=1024, bitsPerHash=1,
                        allowable_clashes=allowable_clashes,
                        output_file='../data/fingerprinting/fingerprints.csv')

def fingerprint_metabolites(mol_dir='../data/mol_files', 
                            output_file='../data/fingerprinting/fingerprints.csv',
                            allowable_clashes=[],
                            minPath=1, maxPath=7, bits=2048, bitsPerHash=2,
                            useHs=True, minSize=128):
    ''' Generates a fixed-length encoding of metabolites using RDKFingerprint '''
    mol_files = [f for f in os.listdir(mol_dir) if os.path.isfile(os.path.join(mol_dir, f))]
    mol_data = []
    
    ''' Generate RDKit Mol objects from files '''
    for mol_file in mol_files:
        bigg = mol_file[:-11]
        mol_path = mol_dir + '/' + mol_file
        mol_block = "\n     RDKit          2D\n\n" # needs this header
        for line in open(mol_path,'r+'):
            mol_block += line
        mol = MolFromMolBlock(mol_block)
        if mol == None: # mol parsing failed
            print('Failed to parse', mol_file)
        else: # successfully extracted mol
            mol_data.append((bigg,mol))
        
    ''' Generate fixed-length chemical fingerprints. Refer here for arguments:
        http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#RDKFingerprint 
        Requires these parameters to explicitly specified. All but fpSize are 
        set to their default values. '''
    print("Generating fingerprints of length", bits)
    fps = FingerprintsFromMols(mol_data, reportFreq=1000, fpSize=bits,
                               minPath=minPath, maxPath=maxPath, 
                               bitsPerHash=bitsPerHash, useHs=useHs, 
                               tgtDensity=0, minSize=minSize)
    
    ''' Examine hash uniqueness '''
    unique_hashes = {}
    for bigg, fp in fps:
        bitstring = fp.ToBitString()
        if not bitstring in unique_hashes:
            unique_hashes[bitstring] = []
        unique_hashes[bitstring].append(bigg)
    print('Number of metabolites hashed:', len(fps))
    print('Number of unique hashes:', len(unique_hashes))
    clashes = []
    for bitstring in unique_hashes:
        hash_group = tuple(sorted(unique_hashes[bitstring]))
        if len(hash_group) > 1: # duplicate hashes
            clashes.append(hash_group)
            if not hash_group in allowable_clashes:
                print('Clash:', ';'.join(hash_group))
    
    ''' Write to file if output path is provided '''
    if output_file:
        f = open(output_file, 'w+')
        f.write('BIGG,Fingerprint\n')
        for bigg, fp in fps:
            f.write(bigg + ',' + fp.ToBitString() + '\n')
        f.close()
    return fps, clashes

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

def get_mol_files_from_kegg(bigg_to_kegg_file='../data/iML1515.chemicals.tsv',
                            output_name='../data/mol_files'):
    ''' Load BIGG to KEGG ID mappings. If multplie KEGG IDs are given for
        a single BIGG ID, take the first one that starts with "C" '''
    df = pd.read_csv(bigg_to_kegg_file, sep='\t', header=None)
    rows, cols = df.shape
    bigg_to_kegg = {}
    for i in range(rows):
        annotations = df.loc[i][6]
        if type(annotations) == str:
            db_mapping = {}
            for annotation in annotations.split(';'):
                db, value = annotation.split(':')
                if db == 'kegg' and value[0] == 'C' and not 'kegg' in db_mapping: 
                    # for kegg labels, take the first label that starts with C
                    db_mapping[db] = value
                elif db != 'kegg': # for non-kegg labels
                    db_mapping[db] = value
            if 'bigg' in db_mapping and 'kegg' in db_mapping:
                bigg_to_kegg[db_mapping['bigg']] = db_mapping['kegg']
    print('Loaded KEGG IDs for', len(bigg_to_kegg), 'BIGG IDs.')
    
    ''' Query KEGG database for MOL files '''
    output_dir = output_name + '/'; os.mkdir(output_dir)
    base_url = 'http://www.kegg.jp/dbget-bin/www_bget?-f+m+compound+' 
    counter = 1
    for bigg in bigg_to_kegg:
        kegg = bigg_to_kegg[bigg]
        url = base_url + kegg
        data = urllib.request.urlopen(url, timeout=10) # should try-except URLError?
        print(str(counter)+':', 'querying', kegg, '('+bigg+')')
        mol_filename = bigg + '_' + kegg + '.mol'
        f = open(output_dir + mol_filename, 'w+')
        for bytestring in data:
            line = bytestring.decode("utf-8") 
            if len(line.strip()) > 0: # ignore whitespace
                f.write(line)
        f.close(); counter += 1
    
    ''' Compress MOL files into tar.gz and delete temporary directory '''
    output_targz = output_name + '.tar.gz'
    with tarfile.open(output_targz, "w:gz") as tar:
        tar.add(output_dir, arcname=os.path.basename(output_dir))
    shutil.rmtree(output_dir)
              
if __name__ == '__main__':
    main()      