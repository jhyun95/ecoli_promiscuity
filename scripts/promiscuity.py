#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:16:51 2018

@author: jhyun95
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cobra

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import sklearn.decomposition, sklearn.cluster, sklearn.manifold, sklearn.preprocessing

MODEL_FILE = '../data/iML1515.json'
GEMPRO_FILE = '../data/iML1515-GEMPro.csv'
EXCLUDED_METS = ['h_c', 'h_p', 'h_e', 'h2o_c', 'h2o_p', 'h2o_e'] 
    # Exclude protons and water as charge balancing metabolites

def main():
    ''' Load base datasets '''
    model = cobra.io.load_json_model(MODEL_FILE)
#    gg_df = extract_gene_groups(model, '../data/gene_groups_no_transport.csv', True)
    gg_df = pd.read_csv('../data/gene_groups_no_transport.csv')
    rows, cols = gg_df.shape
    
    ''' Compute distance matrix from Jaccard index on substrates '''
#    similarities = compute_jaccard_similarities(gg_df, model, '../data/jaccard_similarity_no_transport.csv')
    similarities = pd.read_csv('../data/jaccard_similarity_no_transport.csv', header=0, index_col=0).values
    distances = 1.0 - similarities
    
    ''' Compute promiscuities and labels '''
    promiscuity = np.zeros(rows)
    for i in range(rows):
        promiscuity[i] = compute_promiscuity(i,gg_df,model)
    is_promiscuous = promiscuity > 1.0
    promiscuity_colors = list(map(lambda x: 'r' if x else 'b', is_promiscuous))
    
    ''' Generate promiscuity histogram '''
    bins = np.arange(min(promiscuity)-1, max(promiscuity))
    plt.hist(promiscuity-0.5, bins=bins) 
    plt.xlabel('# of associated reactions')
    plt.ylabel('# of enzymes')
    plt.title('Promiscuity distribution in iML1515')
    
    ''' PCA and tSNE visualization of Jaccard distances '''
    print('Generating PCA plot...')
    pca_distance_matrix(distances, promiscuity_colors)
    print('Generating tSNE plot...')
    tsne_distance_matrix(distances, promiscuity_colors)
    
    ''' Heatmap visualization of Jaccard distances with hierarchical clustering '''
    print('Running heatmap visualization + clustering...')
    hierarchical_distance_heatmap(distances, promiscuity_colors, 'average')
    
    ''' Comparing clusterings of 6 different approaches via Rand Index'''
    print('Comparing clustering approaches...')
    compare_clusterings(distances)
    
    ''' More detailed analysis of large neighborhood DBSCAN clustering '''
    print('Analyzing DBSCAN clusters...')
    evaluate_DBSCAN_clustering(distances, is_promiscuous, gg_df, model)
    
def evaluate_DBSCAN_clustering(distances, is_promiscuous, gene_group_df, model):
    ''' Evalulate 13 clusters from DBSCAN, eps=0.9, min_samples=3
        (shown to be consistent withsome hierarchical clustering) '''
    db9 = sklearn.cluster.DBSCAN(eps=0.9, min_samples=3, metric='precomputed').fit(distances)
    labels = db9.labels_    
    clusters = {}
    for i in set(labels):
        clusters[i] = []
    for j in range(len(labels)):
        label = labels[j]; clusters[label].append(j)
    print('Homogeneity:', sklearn.metrics.homogeneity_score(is_promiscuous, labels))
    print('Completeness:', sklearn.metrics.completeness_score(is_promiscuous, labels))
    print('Silhouette:', sklearn.metrics.silhouette_score(distances, labels, metric='precomputed'), '\n')
        
    ''' Check if any cluster is enriched for promiscuous enzymes '''
    print('Cluster\tSize\t# Promiscuous\t% Promiscuous')
    for label in clusters:
        n_promiscuous = 0; n = len(clusters[label])
        for gg in clusters[label]:
            n_promiscuous += int(is_promiscuous[gg])
        percent_promiscuous = round(n_promiscuous/n * 100, 2)
        output = [label, n, n_promiscuous, percent_promiscuous]
        print('\t'.join(list(map(str,output))))
    print('')
        
    ''' Count unique reactions and subsystems represented within cluster each '''
    df = gene_group_df
    for label in clusters:
        if len(clusters[label]) < 50:
            associated_reactions = []
            for gg in clusters[label]:
                associated_reactions += df.loc[gg]['associated_reactions'].split(';')
            unique_reactions = set(associated_reactions)
            associated_subsystems = []
            for rxnID in unique_reactions:
                associated_subsystems.append(
                    model.reactions.get_by_id(rxnID).subsystem)
            unique_subsystems = {}
            for subsystem in set(associated_subsystems):
                unique_subsystems[subsystem] = associated_subsystems.count(subsystem)
            print('Cluster:', label, len(unique_reactions), unique_subsystems)
            
    ''' Plot along clustered heatmap and PCA plot '''
    fig, ax = plt.subplots(1,2, figsize=(6.5,3)); ax1,ax2 = ax
    row_order = []
    for label in clusters:
        row_order += clusters[label]
    distances_clustered = distances[row_order,:]
    distances_clustered = distances_clustered[:, row_order]
    sns.heatmap(distances_clustered, xticklabels=False, yticklabels=False, ax=ax1)
    ax1.set_xlabel('Enzyme'); ax1.set_ylabel('Enzyme')
    
    pca = sklearn.decomposition.PCA(n_components=10)
    X_pca = pca.fit(distances).transform(distances)
    variances = pca.explained_variance_ratio_

    for label in clusters:
        enzymes = clusters[label]
        ax2.scatter(X_pca[enzymes,0], X_pca[enzymes,1])
    ax2.set_xlabel('PC1 (' + str(round(variances[0]*100,1)) + '%)')
    ax2.set_ylabel('PC2 (' + str(round(variances[1]*100,1)) + '%)')
    
    title = 'DBSCAN clustering of Jaccard distances'
    fig.suptitle(title); fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    
def compare_clusterings(distances, hier_clusters=13):
    ''' Rand Indexes for 6 different clustering methods, as defined
        in the checkpoint 1 writeup. Parameters chosen so all methods
        produce 13 clusters (or 13 clusters + unassigned points):
        1) Hierarchical Clustering - Nearest Point 
        2) Hierarchical Clustering - UPGMA
        3) Hierarchical Clustering - WGPMA
        4) DBSCAN, eps = 0.5, min_samples = 6
        4) DBSCAN, eps = 0.7, min_samples = 5 
        4) DBSCAN, eps = 0.9, min_samples = 3 '''
    def rand_index(labels1, labels2):
        n = len(labels1); consistent = 0; counter = 0
        for i in range(n):
            for j in range(i):
                relate1 = labels1[i] == labels1[j]
                relate2 = labels2[i] == labels2[j]
                if relate1 == relate2:
                    consistent += 1
                counter += 1
        return consistent / counter
    
    print('Running Hierarchical clusterings...')
    X_hier = ssd.squareform(distances) # make condensed distance matrix
    Z_single = sch.linkage(X_hier, method='single')
    Z_average = sch.linkage(X_hier, method='average')
    Z_weighted = sch.linkage(X_hier, method='weighted')
    hier_single_labels = sch.cut_tree(Z_single, n_clusters=hier_clusters)[:,0]
    hier_average_labels = sch.cut_tree(Z_average, n_clusters=hier_clusters)[:,0]
    hier_weighted_labels = sch.cut_tree(Z_weighted, n_clusters=hier_clusters)[:,0]
        
    print('Running DBSCAN clusterings...')
    X_db = distances
    db5 = sklearn.cluster.DBSCAN(eps=0.5, min_samples=6, metric='precomputed').fit(X_db)
    db7 = sklearn.cluster.DBSCAN(eps=0.7, min_samples=5, metric='precomputed').fit(X_db)
    db9 = sklearn.cluster.DBSCAN(eps=0.9, min_samples=3, metric='precomputed').fit(X_db)
    db5_labels = db5.labels_; unassigned5 = list(db5_labels).count(-1)
    db7_labels = db7.labels_; unassigned7 = list(db7_labels).count(-1)
    db9_labels = db9.labels_; unassigned9 = list(db9_labels).count(-1)
    print('DBSCAN Unassigned', unassigned5, unassigned7, unassigned9)
            
    print('Computing Rand Indexes...')
    labels = [hier_single_labels, hier_average_labels, hier_weighted_labels,
              db5_labels, db7_labels, db9_labels]; l = len(labels)
    rand_indices = np.ndarray(shape=(l,l))
    for i in range(l):
        rand_indices[i][i] = 1.0
        for j in range(i):
            ri = rand_index(labels[i], labels[j])
            rand_indices[i][j] = ri
            rand_indices[j][i] = ri
    print(rand_indices)

    ticklabels = ['HC_single', 'HC_average', 'HC_weighted', 
                  'DB_e0.5_ms6', 'DB_e0.7_ms5', 'DB_e0.9_ms3']
    fig, ax = plt.subplots(1,1)
    sns.heatmap(rand_indices, xticklabels=ticklabels, yticklabels=ticklabels, cmap='Blues', ax=ax)
    ax.set_title('Rand Indexes between different clustering methods')
    plt.tight_layout()
    
def hierarchical_distance_heatmap(distances, method):
    ''' Visualize a distance matrix using a heatmap + hiearchical 
        clustering (distance method and metric can be specified) '''
    X = ssd.squareform(distances)
    Z = sch.linkage(X, method=method)
    fig, ax = plt.subplots(1,2, figsize=(6.5,3)); ax1,ax2 = ax
    dend = sch.dendrogram(Z, ax=ax2, no_labels=True)
    ax2.set_xlabel('Enzyme')
    row_order = dend['ivl']; 
    row_order = np.array(list(map(int, row_order)))
    distances_clustered = distances[row_order,:]
    distances_clustered = distances_clustered[:, row_order]
    sns.heatmap(distances_clustered, xticklabels=False, yticklabels=False, ax=ax1)
    ax1.set_xlabel('Enzyme'); ax1.set_ylabel('Enzyme')
    title = 'Hiearchical clustering of distances (' + method + ')'
    fig.suptitle(title); fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    
def pca_distance_matrix(distances, colors):
    ''' Visualize a distance matrix using PCA. Shows plot with and without
        z-score normalization (may not be necessary since all elements in
        Jaccard matrix are bounded by [0,1] and elements are not normal) '''
    X = distances # distance matrix
    std_scale = sklearn.preprocessing.StandardScaler().fit(X)
    X_std = std_scale.transform(X) # z-score normalized
    
    pca = sklearn.decomposition.PCA(n_components=10)
    X_pca = pca.fit(X).transform(X) # compute components and transform to PC1/PC2
    pca_std = sklearn.decomposition.PCA(n_components=10)
    X_pca_std = pca_std.fit(X_std).transform(X_std) # same, but with z-score normalized
    
    variances = pca.explained_variance_ratio_
    variances_std = pca_std.explained_variance_ratio_
    print('Variance by PC (unnormalized):', variances)
    print('Variance by PC (normalized):', variances_std)
    
    fig, ax = plt.subplots(1,2, figsize=(6.5,3)); ax1,ax2 = ax
    ax1.scatter(X_pca[:,0], X_pca[:,1], c=colors)
    ax1.set_xlabel('PC1 (' + str(round(variances[0]*100,1)) + '%)')
    ax1.set_ylabel('PC2 (' + str(round(variances[1]*100,1)) + '%)')
    ax1.set_title('Unnormalized')
    ax2.scatter(X_pca_std[:,0], X_pca_std[:,1], c=colors)
    ax2.set_xlabel('PC1 (' + str(round(variances_std[0]*100,1)) + '%)')
    ax2.set_ylabel('PC2 (' + str(round(variances_std[1]*100,1)) + '%)')
    ax2.set_title('Z-score Normalized')
    plt.suptitle('PCA plot of enzymes (Jaccard distance between known substrates)')
    fig.tight_layout(); fig.subplots_adjust(top=0.82)
    
def tsne_distance_matrix(distances, colors):
    ''' Visualize a distance matrix using tSNE '''
    X = distances # distance matrix
    X_tsne = sklearn.manifold.TSNE(n_components=2).fit_transform(X)
    fig, ax = plt.subplots(1,1)
    ax.scatter(X_tsne[:,0], X_tsne[:,1], c=colors)
    ax.set_title('tSNE plot of enzymes (Jaccard distance between known substrates)')
    
def compute_jaccard_similarities(gene_group_df, model, output_file=None):
    ''' Compute similarity table between gene groups based on Jaccard index, 
        taking unique substrates of each gene group as input set '''
    rows, cols = gene_group_df.shape
    similarities = np.zeros((rows,rows))
    for i in range(rows):
        print(i, gene_group_df.loc[i]['gene_group'])
        for j in range(i+1):
            score = jaccard_similarity(i,j,gene_group_df, model) if i != j else 1.0
            similarities[i][j] = score
            similarities[j][i] = score
    if output_file:
        ggIDs = range(1,rows+1)
        js_df = pd.DataFrame(data=similarities, index=ggIDs, columns=ggIDs)
        js_df.to_csv(output_file)
    return similarities

def jaccard_similarity(gene_group_id1, gene_group_id2, gene_group_df, model):
    ''' Similarity as the Jaccard index between the sets of all possible 
        substrates accepted by the gene groups '''
    subs1 = get_unique_substrates(gene_group_id1, gene_group_df, model)
    subs2 = get_unique_substrates(gene_group_id2, gene_group_df, model)
    union = subs1.union(subs2)
    intersection = subs1.intersection(subs2)
    return len(intersection) / float(len(union))

def compute_promiscuity(gene_group_id, gene_group_df, model):
    ''' Brainstorming approaches to compute quantify: 
            1 - Number of unique reactions associated with gene group
            2 - Number of unique substrate associated with gene group 
            3 - Pairwise simlarity of unique substrates 
        Currently applying approach 1 '''
    return len(gene_group_df.loc[gene_group_id]['associated_reactions'].split(';'))
        
def get_unique_substrates(gene_group_id, gene_group_df, model):
    ''' Gets the metabolite ID of all substrates accepted by a gene group.
        Defined as substrates for each reaction the gene group is associated 
        with, (including products if the reaction is reversible). '''
    rxns = gene_group_df.loc[gene_group_id]['associated_reactions']
    all_substrates = set()
    for rxnID in rxns.split(';'):
        rxn = model.reactions.get_by_id(rxnID)
        lb = rxn.lower_bound; ub = rxn.upper_bound
        for substrate in rxn.metabolites:
            stoich = rxn.metabolites[substrate]
            if not substrate.id in EXCLUDED_METS:
                if stoich < 0 and ub > 0: # substrate, forward reaction allowed
                    all_substrates.add(substrate.id)
                if stoich > 0 and lb < 0: # product, reverse reaction allowed
                    all_substrates.add(substrate.id)
    return all_substrates

def extract_uniprot(gempro_file=GEMPRO_FILE):
    gempro_df = pd.read_csv(open(gempro_file,'r+'))
    gene_to_uniprot = {} # maps gene locus tags to uniprot accession IDs
    rows = gempro_df.shape[0]
    for i in range(rows):
        gene = gempro_df['m_gene'][i]
        uniprotID = gempro_df['seq_uniprot'][i].strip()
        if not gene in gene_to_uniprot and len(uniprotID) > 0:
            gene_to_uniprot[gene] = uniprotID
        else: # gene has already been added
            if gene_to_uniprot[gene] != uniprotID: # multiple IDs for one gene
                print("WARNING:", gene, 'has multiple uniprotIDs')
    print('Loaded UniprotIDs for', len(gene_to_uniprot), 'genes.')
    return gene_to_uniprot

def extract_gene_groups(model, output_file, exclude_transport=False):
    gene_to_uniprot = extract_uniprot(GEMPRO_FILE)
    gene_groups = {} # map independently functional gene groups (i.e. enzymes) to reactions and uniprot IDs
        
    for reaction in model.reactions:
        isValid = reaction.lower_bound != 0.0 or reaction.upper_bound != 0.0
        isTransport = 'TRANSPORT' in reaction.subsystem.upper() or \
            'TRANSPORT' in reaction.name.upper()
        if isValid and not(exclude_transport and isTransport):
            gpr = reaction.gene_reaction_rule
            if len(gpr.strip()) > 0: # reaction is associated with gene products
                for gene_group in gpr.split('or'): # extract independently sufficient groups of genes for the reaction
                    if gene_group.count('(') != gene_group.count(')'): # or-split failed
                        print('WARNING: Could not parse gene group:', reaction.id, gpr, gene_group)
                    else: # successfully isolated gene group
                        gg = gene_group.replace('(',' '); gg = gg.replace(')',' ')
                        genes = gg.split('and') # split individual genes based on and
                        genes = list(map(lambda g: g.strip(), genes)) # remove trailing spaces
                        genes = tuple(sorted(genes)) # sort for consistent ordering
                        if not genes in gene_groups: # new gene group encountered  
                            uniprot = []
                            for gene in genes:
                                if gene in gene_to_uniprot:
                                    uniprot.append(gene_to_uniprot[gene])
                                else:
                                    print("WARNING:", gene, 'in', reaction.id, 'has no associated UniprotID')
                                    uniprot.append('None') 
                            gene_groups[genes] = [uniprot, [reaction.id]]
                        else: # if already encountered this gene group
                            gene_groups[genes][1].append(reaction.id)
        elif not isValid:
            print("WARNING:", reaction.id, 'has LB = UB = 0.0, not included')
        
    ''' Export results to dataframe '''
    gene_group_df = pd.DataFrame(columns=['gene_group', 'associated_uniprot_IDs', 'associated_reactions'])
    counter = 1
    for gene_group in gene_groups:
        uniprot, rxns = gene_groups[gene_group]
        entry = [';'.join(gene_group), ';'.join(uniprot), ';'.join(rxns)]
        gene_group_df.loc[counter] = entry
        counter += 1
    
    rows, cols = gene_group_df.shape
    print('Extracted', rows, 'gene groups.')
    if output_file:
        gene_group_df.to_csv(output_file)        
    return gene_group_df

if __name__ == "__main__":
    main()