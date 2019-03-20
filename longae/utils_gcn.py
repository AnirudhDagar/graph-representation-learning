"""
The MIT License
Copyright (c) 2017 Thomas Kipf

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

Modified from https://github.com/tkipf/gae to work with citation network data.
"""

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys

np.random.seed(1982)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        # My additions
        print ("Printing this unstripped text:", line)
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    # What is the use of mask here?
    print ("Created Mask: ", mask)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_citation_data(dataset_str):
    """Load citation data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']

    """
    
    x               the feature vectors of the labeled training instances
    
    y               the one-hot labels of the labeled training instances
    
    tx              the feature vectors of the test instances
    
    ty              the one-hot labels of the test instances
    
    allx            the feature vectors of both labeled and unlabeled training instances (a superset of x)
    
    ally            the labels for instances in allx
    
    graph           a dict in the format {index: [index_of_neighbor_nodes]}
    
    test.index      the indices of test instances in graph, for the inductive setting
    
    """

    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            # Depending on the python version we need to change the pickle loading method.
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    # Converting the list to a tuple to make it immutable.
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    print ("\nPrinted after passing into tuple:", objects)
    print ("\n\nx:", x)
    print ("\ny:", y)
    print ("\ntx:", tx)
    print ("\nty:", y)
    print ("\nallx:", allx)
    print ("\nally:", ally)
    print ("\ngraph:", graph)


    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)

        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        print ("tx changed citeseer: ", tx)

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
        print ("ty changed citeseer: ", ty)

    # Saving it as a sparse matrix and convert it into linked list representation
    features = sp.vstack((allx, tx)).tolil()
    print ("features:", features)

    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    print ("Adjacency Matrix created from the graph:", adj)

    labels = np.vstack((ally, ty))
    print ("labels:", labels)
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    print ("labels after reordering the indices:", labels)

    idx_test = test_idx_range.tolist()
    print ("\ntest indices idx_test:", idx_test)

    idx_train = range(len(y))
    print ("\nrain indices idx_train:", idx_train)

    idx_val = range(len(y), len(y)+500)
    print ("\nval indices idx_val:", idx_val)

    print("Calling the sample mask function to create masks")
    train_mask = sample_mask(idx_train, labels.shape[0])
    print ("train_mask", train_mask)
    val_mask = sample_mask(idx_val, labels.shape[0])
    print ("val_mask", val_mask)
    test_mask = sample_mask(idx_test, labels.shape[0])
    print ("test_mask", test_mask)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    print ("y_train:", y_train)
    y_val[val_mask, :] = labels[val_mask, :]
    print ("y_val:", y_val)
    y_test[test_mask, :] = labels[test_mask, :]
    print ("y_test:", y_test)

    # Adj and features matrices are in scipy sparse linked list format.
    # Other matrices are in numpy array format
    return adj.tolil(), features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    print ("coords:", coords)
    values = sparse_mx.data
    print ("values:", values)
    shape = sparse_mx.shape
    print ("shape:", shape)

    return coords, values, shape


def split_citation_data(adj):
    """
    Function to build test set with 10% positive links and
    the same number of randomly sampled negative links.
    NOTE: Splits are randomized and results might slightly deviate
    from reported numbers in the paper.
    """

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    print ("adj_triu:", adj_triu)
    print ("Calling the sparse_to_tuple function and passing adj_triu in it")
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = range(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return (np.all(np.any(rows_close, axis=-1), axis=-1) and
                np.all(np.any(rows_close, axis=0), axis=0))

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # NOTE: the edge list only contains single direction of edge!
    return np.concatenate([test_edges, np.asarray(test_edges_false)], axis=0)

