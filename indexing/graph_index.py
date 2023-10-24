from sklearn.cluster import SpectralClustering
import json
import time
import networkx as nx
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import networkx as nx
import argparse
import os

from numpy import linalg as LA
import scipy
from scipy.sparse import csgraph

# from sklearn.cluster import SpectralClustering

def compute_index_without_extra(index_map, mode="item", co_indexing=False):
    reformed_index_map = {}
    for item, labels in index_map.items():
        # reformed_item_CF_index_map[item] = [
        #     "-".join(labels[: i + 1]) for i in range(len(labels))
        # ]
        reformed_index_map[item] = " ".join(labels)

    vocabulary = []

    return reformed_index_map, vocabulary


def construct_indices_from_graph(args, mode="item"):
    print("Current Graph mode: {}".format(mode))
    cluster_number = args.item_cluster_number if mode == "item" else args.user_cluster_number
    cluster_size = args.item_cluster_size if mode == "item" else args.user_cluster_size
    emb_size = args.embedding_length
    base_dir = os.path.join(args.data_dir, args.task, mode + "_graph_indices")
    if args.co_indexing:
        base_dir = os.path.join(args.data_dir, args.task, "co_graph_indices")
    if args.separate_indexing:
        base_dir = os.path.join(args.data_dir, args.task, "useritem_graph_indices")
        
    if not os.path.isfile(
           os.path.join(base_dir,  mode + "_computed_deduplicated_{}_{}_{}_graph_index.json".format(
            cluster_number, cluster_size, emb_size)
        )
    ):
        with open(
            os.path.join(base_dir, mode + "_deduplicated_{}_{}_{}_graph_index.json".format(
                cluster_number, cluster_size, emb_size
            )),
            "r",
        ) as f:
            data = json.load(f)

        # mapping, vocabulary = compute_index(data, mode=mode, co_indexing=args.co_indexing)
        mapping, vocabulary = compute_index_without_extra(data, mode=mode, co_indexing=args.co_indexing)
        
        with open(
            os.path.join(base_dir, 
            mode + "_computed_deduplicated_{}_{}_{}_graph_index.json".format(
                cluster_number, cluster_size, emb_size
            )),
            "w",
        ) as f:
            json.dump([mapping, vocabulary], f, indent=2)
    else:
        with open(
            os.path.join(base_dir,
            mode + "_computed_deduplicated_{}_{}_{}_graph_index.json".format(
                cluster_number, cluster_size, emb_size
            )),
            "r",
        ) as f:
            result = json.load(f)
            mapping = result[0]
            vocabulary = result[1]

            print(vocabulary[:5])
    
    print("Mapping length: {}".format(len(mapping.items())))
    
    return mapping, vocabulary


