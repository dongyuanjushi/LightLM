from sklearn.cluster import SpectralClustering
import random
import os
import json
import time
import math
import numpy as np
import torch
import gzip
from tqdm import tqdm
from CF_index import (
    construct_indices_from_cluster,
    within_category_spectral_clustering,
    construct_indices_from_cluster_optimal_width,
)

from graph_index import (
    construct_indices_from_graph
)
import argparse
from collections import defaultdict, Counter
from itertools import combinations
import warnings
import pickle as pkl

warnings.filterwarnings("ignore")

def no_tokenization(args):
    if args.data_order != "remapped_sequential":
        # has to be the remodeled tokenizer
        mapping = {
            str(i): "<extra_id_{}>".format(str(v)) for i, v in enumerate(range(30000))
        }
    else:
        mapping = {
            str(i): str(i) + "<extra_id_{}>".format(str(v))
            for i, v in enumerate(range(30000))
        }
    return lambda x: mapping[x]


def amazon_asin(args):
    id2item_dir = args.data_dir + args.task + "/datamaps.json"
    with open(id2item_dir, "r") as f:
        datamaps = json.load(f)
    id2item = datamaps["id2item"]

    return id2item


############### CF-based representation ###############
def CF_representation(x, mapping, mode="item"):
    x = str(int(x) - 1)
    if x in mapping:
        return mapping[x]
    # elif mode=="item":
    #     return "<i{}>".format(x)
    # else:
    #     assert mode=="user"
    #     return "<u{}>".format(x)


def create_CF_embedding(args, tokenizer, mode="item"):
    _, vocabulary = construct_indices_from_cluster(args,mode)
    new_tokens = set(vocabulary) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))

    return tokenizer


def CF_representation_optimal_width(x, mapping, mode="item"):
    x = str(int(x) - 1)
    if x in mapping:
        return mapping[x]
    elif mode=="item":
        return "<i{}>".format(x)
    else:
        assert mode=="user"
        return "<u{}>".format(x)

def create_CF_embedding_optimal_width(args, tokenizer, mode="item"):
    _, vocabulary = construct_indices_from_cluster_optimal_width(args, mode)
    new_tokens = set(vocabulary) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))

    return tokenizer

############### graph-based representation ###############
def graph_representation(x, mapping, mode="item"):
    x = str(int(x) - 1)
    # print(len(mapping.keys()))
    if x in mapping:
        return mapping[x]
    # elif mode=="item":
    #     return "<i{}>".format(x)
    # else:
    #     assert mode=="user"
    #     return "<u{}>".format(x)


def create_graph_embedding(args, tokenizer, mode="item"):
    _, vocabulary = construct_indices_from_graph(args,mode)
    new_tokens = set(vocabulary) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))

    return tokenizer


def CF_representation_optimal_width(x, mapping, mode="item"):
    x = str(int(x) - 1)
    if x in mapping:
        return mapping[x]
    elif mode=="item":
        return "<i{}>".format(x)
    else:
        assert mode=="user"
        return "<u{}>".format(x)

def create_CF_embedding_optimal_width(args, tokenizer, mode="item"):
    _, vocabulary = construct_indices_from_cluster_optimal_width(args, mode)
    new_tokens = set(vocabulary) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))

    return tokenizer


if __name__ == "__main__":

    random.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)
    torch.cuda.manual_seed_all(2022)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--task", type=str, default="beauty")
    parser.add_argument("--cluster_size", type=int, default=100)
    parser.add_argument("--cluster_number", type=int, default=20)

    parser.add_argument("--category_no_repetition", action="store_true")
    parser.add_argument(
        "--number_of_items",
        type=int,
        default=11925,
        help="number of items in each dataset, beauty 12102, toys 11925, sports 18358",
    )

    parser.add_argument(
        "--hybrid_order",
        type=str,
        default="CF_first",
        help="CF_first or category_first in concatenation",
    )

    args = parser.parse_args()


    if args.task != "yelp":
        meta_dir = "data/{}/meta.json.gz".format(args.task)
        id2item_dir = "data/{}/datamaps.json".format(args.task)
    else:
        meta_dir = "data/yelp/meta_data.pkl"
        id2item_dir = "data/yelp/datamaps.json"

    id2item_dir = args.data_dir + args.task + "/datamaps.json"

    def parse(path):
        g = gzip.open(path, "r")
        for l in g:
            yield eval(l)

    if args.task != "yelp":
        meta_data = []
        for meta in parse(meta_dir):
            meta_data.append(meta)

        meta_dict = {}
        for i, meta_item in enumerate(meta_data):
            meta_dict[meta_item["asin"]] = i
    else:
        with open(meta_dir, "rb") as f:
            meta_data = pkl.load(f)
        meta_dict = {}
        for i, meta_item in enumerate(meta_data):
            meta_dict[meta_item["business_id"]] = i

    with open(id2item_dir, "r") as f:
        datamaps = json.load(f)
    id2item = datamaps["id2item"]

    category_dict, level_categories = build_category_map(
        args, meta_data, meta_dict, id2item
    )

    reps = []
    for i in tqdm(range(1, args.number_of_items)):
        index = str(i)
        rep = content_based_representation(args, index, category_dict, level_categories)
        print(rep)
        reps.append(rep)
        time.sleep(5)
    print(len(reps))
    repsc = Counter(reps)
    for k, v in repsc.items():
        if v > 1:
            print((k, v))
    print(len(set(reps)))