# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:48:46 2018

@author: jhyun_000
"""

import numpy as np
import pandas as pd
import cobra 

def main():  
    reaction_predictor()
    
def reaction_predictor():
    fps, fp_to_bigg = load_fingerprints()
    model = cobra.io.load_json_model('../data/iML1515.json')
    df = pd.read_csv('../data/gene_groups_no_transport_extended.csv')
    rows, cols = df.shape
    
    ''' If a pair of substrates have a Jaccard distance of at least 
        this, then consider them as possible "seeds" for enzyme 
        promiscuity (i.e. consider metabolites near either substrate
        as potential new substrates) '''
    POSSIBLE_PROMISCUITY_CUTOFF = 0.75 
    
    ''' In a reaction, attempt to identify what the core substrate 
        to product mapping is (opposed to other cofactors or metabolites).
        If a substrate/product pair has Jaccard distance of at least this,
        it is considered a possible core reaction mapping '''
    REACTION_MAPPING_CUTOFF = 0.5
    
    for i in range(rows): # pick which enzymes to print here
        if (i+1) % 50 == 0:
            print('---- Enzyme', i+1, 'of', rows, '-----------------------------')
        ''' Extract reaction and metabolite information for the enzyme '''
        genes = df.loc[i,'gene_group']
        rxns = df.loc[i,'associated_reactions'].split(';')
        substrates = df.loc[i,'substrates'].split(';')
        substrates = map(lambda x: x[:-2], substrates) # drop compartment
        substrates = set(substrates) # drop duplicates
        substrates = list(filter(lambda x: x in fps, substrates)) # drop those w/o fps
        S = len(substrates)
        
        ''' Compute pairwise similarities between substrates for the enzyme '''
        similarities = np.zeros((S,S))
        for i in range(S):
            fp1 = fps[substrates[i]]
            similarities[i,i] = 1
            for j in range(i):
                fp2 = fps[substrates[j]]
                sim = get_jaccard(fp1,fp2)
                similarities[i,j] = sim
                similarities[j,i] = sim
        
        ''' Test if other metabolites come close '''
        for rxnID in rxns:
            rxn = model.reactions.get_by_id(rxnID)
            rxn_substrates, rxn_products = get_substrates_and_products( 
                rxn, model, fps, drop_compartment=True)
            rs = len(rxn_substrates); rp = len(rxn_products)
            
            ''' Check each substrate -> product pair as a possible primary reaction '''
            for a in range(rs):
                rxn_sub = rxn_substrates[a]
                ind = substrates.index(rxn_sub)
                sub_sims = similarities[ind,:] # get intra-substrates similarities
                sub_sims[ind] = 0 # ignore self similarity
                closest = np.max(sub_sims) # closest neighbor
                similarity_cutoff = 1.0 if closest < POSSIBLE_PROMISCUITY_CUTOFF else closest
                fp_sub = fps[rxn_substrates[a]]
                
                for b in range(rp):
                    rxn_prd = rxn_products[b]
                    fp_prd = fps[rxn_prd]
                    is_likely_mapping = get_jaccard(fp_sub, fp_prd) > REACTION_MAPPING_CUTOFF
                    if is_likely_mapping: # substrate is similar enough to product
                        transform = get_transform(fp_sub, fp_prd)
                        substrate_mask = np.logical_and(fp_sub, transform)
                
                        ''' Check other metabolites to be possible candidates:
                            1) Check that metabolite isn't already a substrate
                            2) Check if metabolite has a fingerprint
                            3) Check if it is more similar to the known substrate
                                other known substrate are to that substrate 
                            4) Check if the bits affected by the sub > prod 
                                mapping are identical in the candidate substrate
                            5) Check if applying the exact bit mapping produces
                                a known endogenous metabolite '''
                            
                        for met in model.metabolites:
                            candidate_substrate = met.id[:-2]
                            fp_exists = candidate_substrate in fps
                            new_substrate = not candidate_substrate in substrates                    
                            
                            if fp_exists and new_substrate:
                                candidate_substrate_fp = fps[candidate_substrate] 
                                similarity = get_jaccard(candidate_substrate_fp, fp_sub)
                                
                                if similarity > similarity_cutoff:
                                    cand_mask = np.logical_and(candidate_substrate_fp, transform)
                                    shared_structures = np.logical_xor(cand_mask, substrate_mask)
                                    valid_structures_present = np.count_nonzero(shared_structures) == 0
                                    
                                    if valid_structures_present:
                                        candidate_product_fp = apply_transform(candidate_substrate_fp, transform)
                                        candidate_product_fp_string = ''.join(
                                            map(str, map(int, candidate_product_fp)))
                                        candidate_product_exists = candidate_product_fp_string in fp_to_bigg
                                   
                                        if candidate_product_exists:
                                            candidate_product = fp_to_bigg[candidate_product_fp_string]
                                            print(genes, rxn)
                                            print('\t', rxn_sub, '>', rxn_prd, ':',
                                                  candidate_substrate, '>', candidate_product)
  
def apply_transform(fp, transform):
    ''' Flips the bits at the positions denoted by transform '''
    return np.logical_xor(fp, transform)

def get_transform(fp1, fp2):
    ''' Positions of flipped bits from fp1 to fp2 '''
    return np.logical_xor(fp1, fp2)

def get_mismatches(fp1, fp2):
    ''' Number of mismatched bits '''
    return np.sum(get_transform(fp1, fp2))

def get_jaccard(fp1, fp2):
    ''' Number of shared 1 bits '''
    return np.sum(np.logical_and(fp1, fp2)) / np.sum(np.logical_or(fp1, fp2))

def get_substrates_and_products(rxn, model, fps, drop_compartment=True):
    ''' Splits a reaction into its substrate and products. 
        Only includes metabolites that meet the criteria:
            1) A valid molecular fingerprint exists
            2) Contains carbon
        Ignores reversibility. Default drops compartment. '''
    substrates = []; products = []
    for met in rxn.metabolites:
        bigg = met.id[:-2]
        stoich = rxn.metabolites[met]
        fp_exists = bigg in fps
        has_carbon = 'C' in met.formula
        if fp_exists and has_carbon:
            target = bigg if drop_compartment else met.id
            if stoich < 0: # substrate
                substrates.append(target)
            else: # product
                products.append(target)
    return substrates, products

def load_fingerprints(fingerprint_file='../data/fingerprinting/fingerprints1024_no_clash.csv',
                      ignore_nulls=True):
    df = pd.read_csv(fingerprint_file, index_col=0)
    fingerprints = {}; fp_to_bigg = {}
    for bigg in df.index:
        fpstring = df.loc[bigg,'Fingerprint']
        fp = np.array(list(map(int, list(fpstring))), dtype=np.bool)
        if np.count_nonzero(fp) > 0: # not all zeros
            fingerprints[bigg] = fp
            fp_to_bigg[fpstring] = bigg
        else: # all zeros, ignore
            print('Ignoring all 0 fingerprint:', bigg)
    print('Loaded fingerprints for', len(fingerprints), 'BIGG metabolites.')
    return fingerprints, fp_to_bigg

if __name__ == '__main__':
    main()
        

    