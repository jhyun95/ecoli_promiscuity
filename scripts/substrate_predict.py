# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:48:46 2018

@author: jhyun_000
"""

import numpy as np
import pandas as pd
import cobra 
import seaborn as sns
import matplotlib.pyplot as plt

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

def main():      
    ''' Some exploratory plots ''' 
    fps, _ = load_fingerprints()
#    print(get_jaccard(fps['atp'], fps['hco3']))
#    plot_distance_distr('hco3', fps)   
#    mets = 'atp_c;idp_c;dudp_c;dttp_c;dgtp_c;dcdp_c;dtdp_c;udp_c;ctp_c;dutp_c;dctp_c;cdp_c;itp_c;amp_c;h_c;gdp_c;dadp_c;dgdp_c;gtp_c;damp_c;utp_c;adp_c;datp_c'
#    mets = map(lambda x: x[:-2], mets.split(';'))
#    plot_metabolite_hierarchy(mets, fps, method='weighted', title='b0474 Substrates')

    ''' Substrate prediction '''    
#    reaction_predictor() # too convoluted
    reaction_predictor_2()
    
def reaction_predictor_2():
    ''' Slightly refined version of reaction_predictor '''
    fps, fp_to_bigg = load_fingerprints()
    model = cobra.io.load_json_model('../data/iML1515.json')
    df = pd.read_csv('../data/gene_groups_no_transport_extended.csv')
    rows, cols = df.shape
    candidate_counter = 0
    
    ''' SETTINGS '''
    ''' When considering a pair of substrates for an enzyme (s1, s2), consider
        the distance between s1 and s2 vs the distribution of distances from s1
        to all metabolites, and from s2 to all metabolites. If the s1-s2 
        distance is below this percentile in both distributions, accept the 
        pair as a probable form of enzyme substrate promiscuity. '''
    PERCENTILE_CUTOFF = 5.0
    ''' When checking metabolites for new substrate candidates, the algorithm
        compares the distance of a candidate to either substrate in an 
        s1-s2 promiscuous pair (described above). If the candidate's distance
        to either substrate is within the pair distance * this scaling factor,
        it is called as a probable novel substrate'''
    CANDIDATE_DISTANCE_SCALE = 1.0
    ''' For novel candidates only accept those where the predicted product
        also exists in the model '''
    CHECK_PRODUCT_EXISTENCE = True 
    
    ''' Compute pairwise distances between all metabolites '''
    print('Computing pairwise metabolite distances...')
    bigg_order = list(fps.keys()); m = len(bigg_order)
    distances = np.zeros((m,m))
    for i in range(m):
        distances[i,i] = 0
        bigg1 = bigg_order[i]; fp1 = fps[bigg1]
        for j in range(i):
            bigg2 = bigg_order[j]; fp2 = fps[bigg2]
            dist = 1 - get_jaccard(fp1,fp2)
            distances[i,j] = dist; distances[j,i] = dist
            
    ''' Extract enzyme reaction lists '''
    print('Loading enzyme information...')
    enzymes = {}
    for i in range(rows): # pick which enzymes to print here
        genes = df.loc[i,'gene_group'].split(';')
        rxns = df.loc[i,'associated_reactions'].split(';')
        enzymes[tuple(genes)] = tuple(rxns)
        
    ''' Predict promiscuous reactions '''
    counter = 0
    print('Predicting enzyme promiscuity...')
    for enzyme in enzymes:
        if (counter+1) % 50 == 0:
            print('---- Enzyme', counter+1, 'of', rows, '-----------------------------')
        counter += 1
        ''' Find all substrate and products with fingerprints for each enzyme '''
        rxnIDs = enzymes[enzyme]
        substrates = []; products = []
        for rxnID in rxnIDs:
            rxn = model.reactions.get_by_id(rxnID)
            rxn_subs, rxn_prods = get_substrates_and_products(
                rxn, model, fps, check_reversibility=False)
            substrates += rxn_subs
            products += rxn_prods
        substrates = list(set(substrates)) # remove duplicates
        products = list(set(substrates)) # remove duplicates
        substrate_fps = set(map(lambda x: fingerprint_to_bitstring(fps[x]), substrates)) # list of fingerprints
        
        ''' Get distances between known substrates '''
        s = len(substrates)
        sub_distances = np.zeros((s,s))
        for i in range(s):
            sub_distances[i,i] = 0
            for j in range(s):
                ind1 = bigg_order.index(substrates[i])
                ind2 = bigg_order.index(substrates[j])
                dist = distances[ind1, ind2]
                sub_distances[i,j] = dist
                sub_distances[j,i] = dist
                
        ''' For each substrate, compute its nearest substrate '''
        neighbors = {}
        sub_distances += np.eye(s) # temporarily increase self-distance to avoid matching to self
        for i in range(s):
            sub_dist_row = sub_distances[i,:]
            min_dist_ind = np.argmin(sub_dist_row)
            substrate = substrates[i]; neighbor = substrates[min_dist_ind]
            neighbors[substrate] = neighbor
        sub_distances -= np.eye(s) # reset substrate distances
            
        ''' Retain only nearest-neighbor pairings if they are bidirectional, 
            and their distance is within a percentile of their respective
            distance distributions to all possible metabolites '''
        promiscuity_pairs = []
        for substrate in neighbors:
            neighbor = neighbors[substrate]
            if neighbors[neighbor] == substrate: # bidirectional
                s1 = substrate; ind1 = bigg_order.index(s1)
                s2 = neighbor; ind2 = bigg_order.index(s2)
                dist_distr1 = distances[ind1,:]
                dist_distr2 = distances[ind2,:]
                cutoff1 = np.percentile(dist_distr1, PERCENTILE_CUTOFF)
                cutoff2 = np.percentile(dist_distr2, PERCENTILE_CUTOFF)
                pair_dist = 1 - get_jaccard(fps[s1], fps[s2])
                if pair_dist < min(cutoff1, cutoff2): 
                    promiscuity_pairs.append((substrate, neighbor, pair_dist))
        
        ''' For each identified pair of highly similar substrates, identify
            metabolites that are closer to either one than the pair itself '''
        candidates = []
        for substrate, neighbor, pair_dist in promiscuity_pairs:
            for j in range(m):
                sub_index = bigg_order.index(substrate)
                candidate = bigg_order[j]
                candidate_dist = distances[sub_index, j]
                candidate_str = fingerprint_to_bitstring(fps[candidate])
                if candidate_dist < pair_dist * CANDIDATE_DISTANCE_SCALE and \
                 j != sub_index and not candidate_str in substrate_fps:
                    candidates.append((candidate, substrate, neighbor))
                        
        ''' Within every reaction, consider every substrate:product pair. 
            1) Take the bit positions that are changed from substrate -> product 
            2) For all candidates corresponding to that substrate, check if 
                those positions are the same between the substrate and 
                the candidate
            3) If those positions are conserved, mark the metabolite as 
                a candidate novel substrate 
            4) Optionally, check if the candidate product is a known metabolite '''
        for rxnID in rxnIDs:
            rxn = model.reactions.get_by_id(rxnID)
            rxn_subs, rxn_prods = get_substrates_and_products(
                rxn, model, fps, check_reversibility=False)
            for s in range(len(rxn_subs)):
                for p in range(len(rxn_prods)):
                    sub = rxn_subs[s]; prd = rxn_prods[p]
                    fp_sub = fps[sub]; fp_prd = fps[prd]
                    mask = get_transform(fp_sub, fp_prd) 
                    
                    for candidate, nearest, neighbor in candidates:
                        if nearest == sub:
                            fp_cand = fps[candidate]
                            shared_structures = np.logical_not(np.logical_xor(fp_cand, fp_sub))
                            shared_mask_structures = np.logical_and(mask, shared_structures)
                            all_mask_structures_shared = np.sum(shared_mask_structures) == np.sum(mask)
                            all_mask_structures_shared = True
                            if all_mask_structures_shared and not candidate in substrates: 
                                cand_prd_fp = apply_transform(fp_cand, mask)
                                cand_str = fingerprint_to_bitstring(cand_prd_fp)
                                cand_prd = fp_to_bigg[cand_str] if cand_str in fp_to_bigg else '?'
                                if cand_prd != '?' or not CHECK_PRODUCT_EXISTENCE:
                                    candidate_counter += 1
                                    print(enzyme, rxn)
                                    print('\t', sub, '>', prd, ':', candidate, '>', cand_prd)
                                    
    print('Novel reactions predicted:', candidate_counter)
                    
        
def reaction_predictor():
    ''' Predictor based on inferring substrate:product mappings for every
        reaction, selecting candidate substrates similar to known substrates,
        and applying the substrate:product mapping to candidates to see if
        the candidate product already exists in the model. '''
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
                                            
                                            
def plot_metabolite_hierarchy(metIDs, fps, method='average', title=None):
    ''' Plots the hierarchical clustering of a set of metabolites, 
        based on distances between molecular fingerprints '''
    metIDs = list(filter(lambda x: x in fps, metIDs))
    m = len(metIDs); distances = np.zeros((m,m))
    for i in range(m):
        fp1 = fps[metIDs[i]]
        distances[i,i] = 0
        for j in range(i):
            fp2 = fps[metIDs[j]]
#            dist = get_mismatches(fp1,fp2)
            dist = 1 - get_jaccard(fp1,fp2)
            distances[i,j] = dist; distances[j,i] = dist
    
    ''' Visualize a distance matrix using a heatmap + hiearchical 
        clustering (distance method and metric can be specified) '''
    X = ssd.squareform(distances)
    Z = sch.linkage(X, method=method)
    fig, ax = plt.subplots(1,2, figsize=(6.5,3)); ax1,ax2 = ax
    dend = sch.dendrogram(Z, ax=ax2, labels=metIDs)
    plt.xticks(rotation=60)
    met_order =  dend['ivl']
    row_order = list(map(lambda x: metIDs.index(x), met_order))
    row_order = np.array(list(map(int, row_order)))
    
    distances_clustered = distances[row_order,:]
    distances_clustered = distances_clustered[:, row_order]
    sns.heatmap(distances_clustered, xticklabels=met_order, yticklabels=met_order, ax=ax1)
    if title == None:
        title = 'Hiearchical clustering of distances (' + method + ')'
    fig.suptitle(title); fig.tight_layout()
    fig.subplots_adjust(top=0.9)
                                            
def plot_distance_distr(query_bigg, fps, unique=False):
    ''' Plots distribution of distances from one metabolite to all others '''
    query_fp = fps[query_bigg]
    distances = []
    for bigg in fps:
        if bigg != query_bigg:
#            dist = get_mismatches(query_fp, fps[bigg])
            dist = 1 - get_jaccard(query_fp, fps[bigg])
            distances.append(dist)
#    bins = np.arange(0,len(query_fp),20)
    bins = np.arange(0,1.025,0.025)
    sns.distplot(distances, bins=bins)
  
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

def fingerprint_to_bitstring(fp):
    return ''.join(map(str, fp.astype(np.int).tolist()))

def get_substrates_and_products(rxn, model, fps, drop_compartment=True,
                                check_reversibility=False):
    ''' Splits a reaction into its substrate and products. 
        Only includes metabolites that meet the criteria:
            1) A valid molecular fingerprint exists
            2) Contains carbon
        Default ignores reversibility. Default drops compartment. '''
    substrates = []; products = []
    for met in rxn.metabolites:
        bigg = met.id[:-2]
        stoich = rxn.metabolites[met]
        fp_exists = bigg in fps
        has_carbon = 'C' in met.formula
        if fp_exists and has_carbon:
            target = bigg if drop_compartment else met.id
            if stoich < 0 or (rxn.reversibility and check_reversibility): # substrate
                substrates.append(target)
            if stoich > 0 or (rxn.reversibility and check_reversibility): # product
                products.append(target)
    return substrates, products

def load_fingerprints(fingerprint_file='../data/fingerprinting/fingerprints1024_no_clash.csv',
                      ignore_nulls=True):
    ''' Loads fingerprints as bigg IDs : fingerprint vector 
        Also returns the reverse mapping fingerprint vector : bigg. '''
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
        

    