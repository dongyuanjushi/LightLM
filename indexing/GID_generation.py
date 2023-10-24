#!/usr/bin/env python
# coding: utf-8

# In[23]:


import time
import networkx as nx
from collections import defaultdict, Counter
from itertools import combinations
from sklearn.cluster import SpectralClustering
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from numpy import linalg as LA
import scipy
from scipy.sparse import csgraph
import math
from scipy.sparse.linalg import eigsh
import random
import json
import os
from tqdm import trange
import torch
from sklearn.cluster import KMeans

def composite_function(f, g):
    return lambda x: f(g(x))

########### data preprocessing ###########
# base_dir = "/common/home/km1558/amazon_data/data"
base_dir = "/common/users/zl502/rec_data/data"

task = "toys"
# task = "beauty"
# task = "lastfm"
# task = "taobao"
# task = "yelp"

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

emb_size = 64

def normalize(matrix):
    # min-max normalization
    # min_vals = np.min(matrix, axis=0)
    # max_vals = np.max(matrix, axis=0)
    # normalized_matrix = (matrix - min_vals) / (max_vals - min_vals)
    # normalized_matrix = np.round(matrix * 100).astype(int)

    # z-score normalization
    mean_val = np.mean(matrix)
    std_dev = np.std(matrix)    
    normalized_matrix = (matrix - mean_val) / std_dev
    
    return normalized_matrix


# In[24]:


# graph = "LightGCN"
graph = "BPRMF"

# device = "cuda:2"

user_emb = np.load(os.path.join(base_dir, task, graph + "_user_emb_" + str(emb_size) + ".npy"))

user_emb = normalize(user_emb)
# user_emb = sigmoid(user_emb)
# user_emb = torch.tensor(user_emb).to(device)

num_of_users = user_emb.shape[0]

print(user_emb.shape)

item_emb = np.load(os.path.join(base_dir, task, graph + "_item_emb_" + str(emb_size) + ".npy"))

item_emb = normalize(item_emb)
# item_emb = sigmoid(item_emb)
# item_emb = torch.tensor(item_emb).to(device)

num_of_items = item_emb.shape[0]
# print(sorted_similarity[:10])

# emb_size = item_emb.shape[1]


# In[25]:


user_id_map = {}

item_id_map = {}


# In[31]:


begin_time = time.time()

number_of_clusters = 20

maximum_cluster_size = 100

mode = "user"
# mode = "item"

if mode == "user":
    data = user_emb
elif mode == "item":
    data = item_emb

clustering = KMeans(n_clusters=number_of_clusters, random_state=0, n_init="auto").fit(data)

end_time = time.time()
used_time = end_time - begin_time
print("used time to compute it is {} seconds".format(used_time))

labels = clustering.labels_.tolist()

print(len(labels))

graph_index_map = {i: [str(label)] for i, label in enumerate(labels)}

group_labels = []
for i in range(number_of_clusters):
    group_labels.append(labels.count(i))


# In[32]:


print(group_labels)
print(labels[:50])


# In[33]:


def one_further_indexing(
    which_group,
    group_labels,
    sub_data,
    number_of_clusters,
    graph_index_map,
    reverse_fcts,
    mode,
):
    # select items in this large cluster
    one_subcluster_items = [
        item for item, l in enumerate(group_labels) if l == which_group
    ]
    # edges within the subcluster
    # subcluster_pairs = [
    #     sorted_pair_freq
    #     for sorted_pair_freq in sorted_pair_freqs
    #     if sorted_pair_freq[0][0] in one_subcluster_items
    #     and sorted_pair_freq[0][1] in one_subcluster_items
    # ]
    # remap the item indices
    
    item_map = {
        old_item_index: i for i, old_item_index in enumerate(one_subcluster_items)
    }
    reverse_item_map = {
        i: old_item_index for i, old_item_index in enumerate(one_subcluster_items)
    }
    # modify the subcluster pairs by item_map
    # remapped_sub_index = [
    #     item_map[item] for item in sub_index
    # ]

    sub_group_data = []

    for i in one_subcluster_items:
        sub_group_data.append(sub_data[i])

    sub_group_data = np.array(sub_group_data)
    # create new matrix
    # sub_matrix_size = len(item_map)
    # sub_adjacency_matrix = np.zeros((sub_matrix_size, sub_matrix_size))
    # for pair, freq in remapped_subcluster_pairs:
    #     sub_adjacency_matrix[pair[0], pair[1]] = freq
    #     sub_adjacency_matrix[pair[1], pair[0]] = freq

    numberofclusters = number_of_clusters

    # clustering
    sub_clustering = KMeans(
        n_clusters=number_of_clusters, random_state=0, n_init="auto"
    ).fit(sub_group_data)
    sub_labels = sub_clustering.labels_.tolist()

    # remap the index to the actual item
    reversal = lambda x: x
    for reverse_fct in reverse_fcts:
        reversal = composite_function(
            reverse_fct, reversal
        )  # lambda x: reverse_fct(reversal(x))

    for i, label in enumerate(sub_labels):
        graph_index_map[reversal(reverse_item_map[i])].append(str(label))

    # concatenate the new reverse function
    new_reverse_fcts = [lambda y: reverse_item_map[y]] + reverse_fcts

    return sub_labels, sub_group_data, graph_index_map, new_reverse_fcts


# In[34]:


######### recursive application
level_one = labels
level_two = []
level_three = []
level_four = []
level_five = []
level_six = []
level_seven = []
level_eight = []
N = number_of_clusters
M = maximum_cluster_size
reverse_fcts = [lambda x: x]
for a in range(N):
    if level_one.count(a) > M:
        (
            a_labels,
            sub_a_data,
            graph_index_map,
            level_two_reverse_fcts,
        ) = one_further_indexing(
            a, labels, data, N, graph_index_map, reverse_fcts, 2
        )
        level_two.append((a, a_labels))

        for b in range(N):
            if a_labels.count(b) > M:
                (
                    b_labels,
                    sub_b_data,
                    graph_index_map,
                    level_three_reverse_fcts,
                ) = one_further_indexing(
                    b,
                    a_labels,
                    sub_a_data,
                    N,
                    graph_index_map,
                    level_two_reverse_fcts,
                    3,
                )
                level_three.append((a, b, b_labels))

                for c in range(N):
                    if b_labels.count(c) > M:
                        (
                            c_labels,
                            sub_c_data,
                            graph_index_map,
                            level_four_reverse_fcts,
                        ) = one_further_indexing(
                            c,
                            b_labels,
                            sub_b_data,
                            N,
                            graph_index_map,
                            level_three_reverse_fcts,
                            4,
                        )
                        level_four.append((a, b, c, c_labels))

                        for d in range(N):
                            if c_labels.count(d) > M:
                                (
                                    d_labels,
                                    sub_d_data,
                                    graph_index_map,
                                    level_five_reverse_fcts,
                                ) = one_further_indexing(
                                    d,
                                    c_labels,
                                    sub_c_data,
                                    N,
                                    graph_index_map,
                                    level_four_reverse_fcts,
                                    5,
                                )
                                level_five.append((a, b, c, d, d_labels))

                                for e in range(N):
                                    if d_labels.count(e) > M:
                                        (
                                            e_labels,
                                            sub_e_data,
                                            graph_index_map,
                                            level_six_reverse_fcts,
                                        ) = one_further_indexing(
                                            e,
                                            d_labels,
                                            sub_d_data,
                                            N,
                                            graph_index_map,
                                            level_five_reverse_fcts,
                                            6,
                                        )
                                        level_six.append((a, b, c, d, e, e_labels))

                                        for f in range(N):
                                            if e_labels.count(f) > M:
                                                (
                                                    f_labels,
                                                    sub_f_data,
                                                    graph_index_map,
                                                    level_seven_reverse_fcts,
                                                ) = one_further_indexing(
                                                    f,
                                                    e_labels,
                                                    sub_e_data,
                                                    N,
                                                    graph_index_map,
                                                    level_six_reverse_fcts,
                                                    7,
                                                )
                                                level_seven.append(
                                                    (a, b, c, d, e, f, f_labels)
                                                )

                                                for g in range(N):
                                                    if f_labels.count(g) > M:
                                                        (
                                                            g_labels,
                                                            sub_g_data,
                                                            graph_index_map,
                                                            level_eight_reverse_fcts,
                                                        ) = one_further_indexing(
                                                            g,
                                                            f_labels,
                                                            sub_f_data,
                                                            N,
                                                            graph_index_map,
                                                            level_seven_reverse_fcts,
                                                            8,
                                                        )
                                                        level_eight.append(
                                                            (
                                                                a,
                                                                b,
                                                                c,
                                                                d,
                                                                e,
                                                                f,
                                                                g,
                                                                g_labels,
                                                            )
                                                        )


# In[35]:


graph_index_path = os.path.join(base_dir, task, mode + "_graph_indices", mode + "_{}_{}_{}_graph_index.json".format(number_of_clusters, maximum_cluster_size, emb_size))
with open(
    graph_index_path, "w"
) as f:
    json.dump(graph_index_map, f, indent=2)


# In[38]:


import copy
user_graph_index_path = os.path.join(base_dir, task, "user_graph_indices", "user_{}_{}_{}_graph_index.json".format(number_of_clusters, maximum_cluster_size, emb_size))

with open(
    user_graph_index_path, "r"
) as f:
    user_graph_index_map = json.load(f)
    print("length of remapped users: {}".format(len(user_graph_index_map.items())))
    
item_graph_index_path = os.path.join(base_dir, task, "item_graph_indices", "item_{}_{}_{}_graph_index.json".format(number_of_clusters, maximum_cluster_size, emb_size))

with open(
    item_graph_index_path, "r"
) as f:
    item_graph_index_map = json.load(f)
    print("length of remapped items: {}".format(len(item_graph_index_map.items())))

# deduplicate_mode = "user"

deduplicate_mode = "item"
    
# deduplicate_mode = "useritem"

if deduplicate_mode == "user" or deduplicate_mode == "item":
    
    if deduplicate_mode == "user":
        duplicate_graph_index_map = copy.deepcopy(user_graph_index_map)
        size = num_of_users
    else:
        duplicate_graph_index_map = copy.deepcopy(item_graph_index_map)
        size = num_of_items
        
    duplicate_cluster = defaultdict(int)      

    deduplicated_graph_index_map = {}

    for i in range(0, size):
        id = " ".join(duplicate_graph_index_map[str(i)])

        duplicate_cluster[id] += 1

    id_count = {}
    for i in range(0, size):
        id = " ".join(duplicate_graph_index_map[str(i)])

        if duplicate_cluster[id] > 1:
            if id not in id_count.keys():
                id_count[id] = 0

            deduplicated_graph_index_map[i] = duplicate_graph_index_map[str(i)] + [str(id_count[id])]

            id_count[id] += 1
        else:
            deduplicated_graph_index_map[i] = duplicate_graph_index_map[str(i)]
    
    print("length of deduplicated items: {}".format(len(deduplicated_graph_index_map.items())))
    deduplicated_graph_index_path = os.path.join(base_dir, task, deduplicate_mode + "_graph_indices", deduplicate_mode + "_deduplicated_{}_{}_{}_graph_index.json".format(number_of_clusters, maximum_cluster_size, emb_size))
    
    with open(
        deduplicated_graph_index_path, "w"
    ) as f:
        json.dump(deduplicated_graph_index_map, f, indent=2)

elif deduplicate_mode == "useritem":
    
    duplicate_graph_index_map = copy.deepcopy(user_graph_index_map)
    
    for i in range(0, num_of_items):
        duplicate_graph_index_map[str(i + num_of_users)] = item_graph_index_map[str(i)]
    
    print(len(duplicate_graph_index_map.items()))
    
    duplicate_cluster = defaultdict(int)      

    deduplicated_graph_index_map = {}

    for i in range(0, num_of_users + num_of_items):
        id = " ".join(duplicate_graph_index_map[str(i)])

        duplicate_cluster[id] += 1

    id_count = {}
    for i in range(0, num_of_users + num_of_items):
        id = " ".join(duplicate_graph_index_map[str(i)])

        if duplicate_cluster[id] > 1:
            if id not in id_count.keys():
                id_count[id] = 0

            deduplicated_graph_index_map[str(i)] = duplicate_graph_index_map[str(i)] + [str(id_count[id])]

            id_count[id] += 1
            
        else:
            deduplicated_graph_index_map[str(i)] = duplicate_graph_index_map[str(i)]
    
    deduplicated_user_graph_index_map = {}
    deduplicated_item_graph_index_map = {}
    
    for i in range(0, num_of_users):
        deduplicated_user_graph_index_map[str(i)] = deduplicated_graph_index_map[str(i)]
        
    for i in range(0, num_of_items):
        deduplicated_item_graph_index_map[str(i)] = deduplicated_graph_index_map[str(i + num_of_users)]
    
    deduplicated_user_graph_index_path = os.path.join(
        base_dir, task, deduplicate_mode + "_graph_indices", "user_deduplicated_{}_{}_{}_graph_index.json".
        format(number_of_clusters, maximum_cluster_size, emb_size)
    )
    
    deduplicated_item_graph_index_path = os.path.join(
        base_dir, task, deduplicate_mode + "_graph_indices", "item_deduplicated_{}_{}_{}_graph_index.json".
        format(number_of_clusters, maximum_cluster_size, emb_size)
    )
    
    with open(
        deduplicated_user_graph_index_path, "w"
    ) as f:
        json.dump(deduplicated_user_graph_index_map, f, indent=2)
    
    with open(
        deduplicated_item_graph_index_path, "w"
    ) as f:
        json.dump(deduplicated_item_graph_index_map, f, indent=2)


# In[162]:


quantized_length = 4

steps = emb_size // quantized_length


# In[57]:


# for i in range(0, quantized_length):
#     data = user_emb[:,i * steps : (i+1) * steps]
#     kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(data)
#     print(i)
#     labels = kmeans.labels_.tolist()
#     for idx, label in enumerate(labels):
#         if str(idx) not in user_id_map.keys():
#             user_id_map[str(idx)] = []
#         user_id_map[str(idx)].append(str(label))

# for i in range(0, 10):
#     print(user_id_map[str(i)])


# In[58]:


for i in range(0, quantized_length):
    data = item_emb[:, i * steps : (i+1) * steps]
    kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(data)
    print(i)
    labels = kmeans.labels_.tolist()
    for idx, label in enumerate(labels):
        if str(idx) not in item_id_map.keys():
            item_id_map[str(idx)] = []
        item_id_map[str(idx)].append(str(label))

for i in range(0, 10):
    print(item_id_map[str(i)])


# In[ ]:


# print(sorted_similarity_seq[:50])


# In[ ]:


# def normalize(matrix):
#     min_vals = np.min(matrix, axis=0)
#     max_vals = np.max(matrix, axis=0)

# #     # Perform min-max normalization along the column axis
# #     # normalized_matrix = np.clip(np.round((matrix - min_vals) / (max_vals - min_vals) * 100, decimals = 0), a_min = 1, a_max = 99).astype(int)
#     normalized_matrix = np.clip(np.round((matrix - min_vals) / (max_vals - min_vals) * 50, decimals = 0), a_min = 1, a_max = 50).astype(int)
#     # normalized_matrix = np.round(matrix * 100).astype(int)
    
#     return normalized_matrix


# In[ ]:


# normalized_user_emb = normalize(user_emb)

# print(normalized_user_emb[-5:])

# normalized_item_emb = normalize(item_emb)

# print(normalized_item_emb[-5:])

# normalized_useritem_emb = normalize(np.concatenate((user_emb,item_emb),axis=0))

# print(normalized_useritem_emb.shape)

# print(normalized_useritem_emb[-5:])


# In[ ]:


# def quantize(id_map, normalized_emb, num, k):
#     for idx in range(0, num):
#         emb = normalized_emb[idx - 1]
#         quantized_index = []
#         for i in range(0, len(emb) // k):
#             q_i = ""
#             for j in range(0, k):
#                 q_i += str(emb[k * i + j])
#             quantized_index.append(q_i)
#         # if idx < 10:
#         #     print(quantized_index)
#         id_map[str(idx)] = quantized_index

# interval = 1
        
# quantize(user_id_map, normalized_user_emb, user_num, interval)
# quantize(item_id_map, normalized_item_emb, item_num, interval)


# In[ ]:


co_emb = np.concatenate((user_emb,item_emb),axis=0)


# In[61]:


duplicate_user_cluster = {}

duplicate_uids = []

for i in range(0, user_num):
    user_id = " ".join(user_id_map[str(i)])
    if not user_id in duplicate_user_cluster.keys():
        duplicate_user_cluster[user_id] = 1
    duplicate_user_cluster[user_id] += 1
    if duplicate_user_cluster[user_id] >= 100:
        if user_id not in duplicate_uids:
            duplicate_uids.append(user_id)
            print("Index exceeds 100")
            print(user_id)
            print(duplicate_user_cluster[user_id])
            print()
    
user_count = {} 

# for k,v in duplicate_user_cluster.items():
#     if v > 1:
#         print(k)
#         print(v)
    
for i in range(0, user_num):
    user_id = " ".join(user_id_map[str(i)])
    if duplicate_user_cluster[user_id] > 1:
        # print(i)
        if user_id not in user_count.keys():
            user_count[user_id] = 0
        user_id_map[str(i)] = user_id_map[str(i)] + [str(user_count[user_id])]
        user_count[user_id] += 1


# In[65]:


duplicate_item_cluster = {}
item_count = {}

duplicate_iids = []

for i in range(0, item_num):
    item_id = " ".join(item_id_map[str(i)])
    if not item_id in duplicate_item_cluster.keys():
        duplicate_item_cluster[item_id] = 1
    duplicate_item_cluster[item_id] += 1

    if duplicate_item_cluster[item_id] >= 100:
        if item_id not in duplicate_iids:
            duplicate_iids.append(item_id)
            print("Index exceeds 100")
            print(item_id)
            print(duplicate_item_cluster[item_id])
            print()

for i in range(0, item_num):
    item_id = " ".join(item_id_map[str(i)])
    if duplicate_item_cluster[item_id]:
        if item_id not in item_count.keys():
            item_count[item_id] = 0
        item_id_map[str(i)] = item_id_map[str(i)] + [str(item_count[item_id])]
        item_count[item_id] += 1

# quantized_len = emb_size // interval

# print(quantized_len)
print(quantized_length)


# In[66]:


save_dir = "/common/users/zl502/rec_data/data/"

user_save_dir = os.path.join(save_dir, task, "user_graph_indices")

item_save_dir = os.path.join(save_dir, task, "item_graph_indices")

if not os.path.exists(user_save_dir):
    os.makedirs(user_save_dir)

with open(os.path.join(user_save_dir, "user_"+ str(quantized_length) + "_index.json"), "w") as f:
    json.dump(user_id_map, f, indent=2)

if not os.path.exists(item_save_dir):
    os.makedirs(item_save_dir)

with open(os.path.join(item_save_dir, "item_"+ str(quantized_length) + "_index.json"), "w") as f:
    json.dump(item_id_map, f, indent=2)


# In[ ]:


co_user_id_map = {}

co_item_id_map = {}


# In[ ]:


# interval = 1

# def quantize_both(normalized_emb, num, k):
#     for idx in range(0, user_num):
#         emb = normalized_emb[idx - 1]
#         quantized_index = []
#         for i in range(0, len(emb) // k):
#             q_i = ""
#             for j in range(0, k):
#                 q_i += str(emb[k * i + j])
#             quantized_index.append(q_i)
#         # if idx < 10:
#         #     print(quantized_index)
#         co_user_id_map[str(idx)] = quantized_index
        
#     for idx in range(0, item_num):
#         emb = normalized_emb[idx + user_num - 1]
#         quantized_index = []
#         for i in range(0, len(emb) // k):
#             q_i = ""
#             for j in range(0, k):
#                 q_i += str(emb[k * i + j])
#             quantized_index.append(q_i)
#         # if idx < 10:
#         #     print(quantized_index)
#         co_item_id_map[str(idx)] = quantized_index

# quantize_both(normalized_useritem_emb, user_num + item_num, interval)


# In[ ]:


for i in range(0, quantized_length):
    data = co_emb[:,i * steps : (i+1) * steps]
    kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(data)
    print(i)
    labels = kmeans.labels_.tolist()
    for idx, label in enumerate(labels):
        if str(idx) not in user_id_map.keys():
            user_id_map[str(idx)] = []
        user_id_map[str(idx)].append(str(label))

for i in range(0, 10):
    print(user_id_map[str(i)])


# In[ ]:


duplicate_cluster = {}

count = {}

for i in range(0, user_num):
    user_id = " ".join(co_user_id_map[str(i)])
    if not user_id in duplicate_cluster.keys():
        duplicate_cluster[user_id] = 1
    duplicate_cluster[user_id] += 1

for i in range(0, item_num):
    item_id = " ".join(co_item_id_map[str(i)])
    if not item_id in duplicate_cluster.keys():
        duplicate_cluster[item_id] = 1
    duplicate_cluster[item_id] += 1
    
# for k,v in duplicate_cluster.items():
#     if v > 1:
#         print(k)
#         print(v)
    
for i in range(0, user_num):
    user_id = " ".join(co_user_id_map[str(i)])
    if duplicate_cluster[user_id] > 1:
        # print(i)
        if user_id not in count.keys():
            count[user_id] = 0
        co_user_id_map[str(i)] = co_user_id_map[str(i)] + [str(count[user_id])]
        count[user_id] += 1
    
for i in range(0, item_num):
    item_id = " ".join(co_item_id_map[str(i)])
    if duplicate_cluster[item_id] > 1:
        if item_id not in count.keys():
            count[item_id] = 0
        co_item_id_map[str(i)] = co_item_id_map[str(i)] + [str(count[item_id])]
        count[item_id] += 1

quantized_len = emb_size // interval

print(quantized_len)

for i in range(1, 10):
    print(co_user_id_map[str(i)])
    print(co_item_id_map[str(i)])

quantized_len = emb_size // interval

print(quantized_len)


# In[ ]:


print(interval)


# In[ ]:


co_save_dir = os.path.join(save_dir, task, "co_graph_indices")

if not os.path.exists(co_save_dir):
    os.makedirs(co_save_dir)

with open(os.path.join(co_save_dir, "user_"+ str(quantized_len) + "_index.json"), "w") as f:
    json.dump(user_id_map, f, indent=2)

with open(os.path.join(co_save_dir, "item_"+ str(quantized_len) + "_index.json"), "w") as f:
    json.dump(item_id_map, f, indent=2)


# In[ ]:




