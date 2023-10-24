#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from tqdm import tqdm
import json
import os

def composite_function(f, g):
    return lambda x: f(g(x))


########### data preprocessing ###########
# base_dir = "/common/home/km1558/amazon_data/data"
base_dir = "/common/users/km1558/rec_data/data" # the data directory needs to be changed

task = "toys"
# task = "beauty"
# task = "lastfm"
# task = "taobao"
# task = "yelp"

CF_mode = "user"
# CF_mode = "item"

user_data_path = os.path.join(base_dir, task, "user_CF_indices", "data.txt")
with open(user_data_path, "r") as f:
    user_data = f.read()

user_data = user_data.split("\n")[:-1]

user_data = [d.split(" ")[1:-2] for d in user_data]

user_words = [[int(one_d) - 1 for one_d in d] for d in user_data]

item_data_path = os.path.join(base_dir, task, "item_CF_indices", "data.txt")
with open(item_data_path, "r") as f:
    item_data = f.read()

item_data = item_data.split("\n")[:-1]

item_data = [d.split(" ")[1:-2] for d in item_data]

num_of_users = len(item_data)

num_of_items = len(user_data)

print(num_of_users)

print(num_of_items)

data_path = os.path.join(base_dir, task, CF_mode + "_CF_indices", "data.txt")
with open(data_path, "r") as f:
    data = f.read()

data = data.split("\n")[:-1]
data = [d.split(" ")[1:-2] for d in data]

words = [[int(one_d) - 1 for one_d in d] for d in data]
original_vocab = set([a for one_word in words for a in one_word])


# In[2]:


print(data[:5])
print(words[:5])


# In[4]:


print("compute pair frequency")
# compute pairs
pair_freqs = defaultdict(int)
for word in words:
    pairs = combinations(word, 2)
    for pair in pairs:
        if pair[0] != pair[1]:
            pair_freqs[tuple(pair)] += 1
sorted_pair_freqs = sorted(pair_freqs.items(), key=lambda x: x[1], reverse=True)

print("compute frequency")

print(sorted_pair_freqs[:10])
# compute single item occurrences


# In[5]:


occurrences = Counter([a for one_word in words for a in one_word])
occurrences = sorted(occurrences.items(), key=lambda x: x[1], reverse=True)

if CF_mode == "user":
    matrix_size = num_of_users

else:
    matrix_size = num_of_items

print(matrix_size)

print("compute adjacency matrix")
adjacency_matrix = np.zeros((matrix_size, matrix_size))
for pair, freq in sorted_pair_freqs:
    adjacency_matrix[pair[0], pair[1]] = freq
    adjacency_matrix[pair[1], pair[0]] = freq

print(sorted_pair_freqs[:10])

###### apply clustering for the first time, k and N are up to change
maximum_cluster_size = 100
number_of_clusters = 20
# print(occurrences[:10])


# In[39]:


# here adjacency matrix is the affinity matrix
begin_time = time.time()

clustering = SpectralClustering(
    n_clusters=number_of_clusters,
    assign_labels="cluster_qr",
    random_state=0,
    affinity="precomputed",
).fit(adjacency_matrix)

end_time = time.time()
used_time = end_time - begin_time
print("used time to compute it is {} seconds".format(used_time))

labels = clustering.labels_.tolist()

print(len(labels))

CF_index_map = {i: [str(label)] for i, label in enumerate(labels)}

group_labels = []
for i in range(number_of_clusters):
    group_labels.append(labels.count(i))

cluster_save_dir = os.path.join(base_dir, task, CF_mode+"_CF_indices", CF_mode + "_" + str(number_of_clusters) + "_cluster.npy")

print("saving " + CF_mode + " cluster labels.")

np.save(cluster_save_dir, np.array(labels))


# In[40]:


for i in range(number_of_clusters):
    print(labels.count(i))
print("length of labels: {}".format(len(labels)))


# In[41]:


def one_further_indexing(
    which_group,
    group_labels,
    sorted_pair_freqs,
    number_of_clusters,
    CF_index_map,
    reverse_fcts,
    mode,
):
    # select items in this large cluster
    one_subcluster_items = [
        item for item, l in enumerate(group_labels) if l == which_group
    ]
    # edges within the subcluster
    subcluster_pairs = [
        sorted_pair_freq
        for sorted_pair_freq in sorted_pair_freqs
        if sorted_pair_freq[0][0] in one_subcluster_items
        and sorted_pair_freq[0][1] in one_subcluster_items
    ]
    # remap the item indices
    item_map = {
        old_item_index: i for i, old_item_index in enumerate(one_subcluster_items)
    }
    reverse_item_map = {
        i: old_item_index for i, old_item_index in enumerate(one_subcluster_items)
    }
    # modify the subcluster pairs by item_map
    remapped_subcluster_pairs = [
        (
            (item_map[subcluster_pair[0][0]], item_map[subcluster_pair[0][1]]),
            subcluster_pair[1],
        )
        for subcluster_pair in subcluster_pairs
    ]

    # create new matrix
    sub_matrix_size = len(item_map)
    sub_adjacency_matrix = np.zeros((sub_matrix_size, sub_matrix_size))
    for pair, freq in remapped_subcluster_pairs:
        sub_adjacency_matrix[pair[0], pair[1]] = freq
        sub_adjacency_matrix[pair[1], pair[0]] = freq

    numberofclusters = number_of_clusters

    # clustering
    sub_clustering = SpectralClustering(
        n_clusters=numberofclusters,
        assign_labels="cluster_qr",
        random_state=0,
        affinity="precomputed",
    ).fit(sub_adjacency_matrix)
    sub_labels = sub_clustering.labels_.tolist()

    # remap the index to the actual item
    reversal = lambda x: x
    for reverse_fct in reverse_fcts:
        reversal = composite_function(
            reverse_fct, reversal
        )  # lambda x: reverse_fct(reversal(x))

    for i, label in enumerate(sub_labels):
        CF_index_map[reversal(reverse_item_map[i])].append(str(label))

    # concatenate the new reverse function
    new_reverse_fcts = [lambda y: reverse_item_map[y]] + reverse_fcts

    return sub_labels, remapped_subcluster_pairs, CF_index_map, new_reverse_fcts


# In[42]:


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
            remapped_a_cluster_pairs,
            CF_index_map,
            level_two_reverse_fcts,
        ) = one_further_indexing(
            a, labels, sorted_pair_freqs, N, CF_index_map, reverse_fcts, 2
        )
        level_two.append((a, a_labels))

        for b in range(N):
            if a_labels.count(b) > M:
                (
                    b_labels,
                    remapped_b_cluster_pairs,
                    CF_index_map,
                    level_three_reverse_fcts,
                ) = one_further_indexing(
                    b,
                    a_labels,
                    remapped_a_cluster_pairs,
                    N,
                    CF_index_map,
                    level_two_reverse_fcts,
                    3,
                )
                level_three.append((a, b, b_labels))

                for c in range(N):
                    if b_labels.count(c) > M:
                        (
                            c_labels,
                            remapped_c_cluster_pairs,
                            CF_index_map,
                            level_four_reverse_fcts,
                        ) = one_further_indexing(
                            c,
                            b_labels,
                            remapped_b_cluster_pairs,
                            N,
                            CF_index_map,
                            level_three_reverse_fcts,
                            4,
                        )
                        level_four.append((a, b, c, c_labels))

                        for d in range(N):
                            if c_labels.count(d) > M:
                                (
                                    d_labels,
                                    remapped_d_cluster_pairs,
                                    CF_index_map,
                                    level_five_reverse_fcts,
                                ) = one_further_indexing(
                                    d,
                                    c_labels,
                                    remapped_c_cluster_pairs,
                                    N,
                                    CF_index_map,
                                    level_four_reverse_fcts,
                                    5,
                                )
                                level_five.append((a, b, c, d, d_labels))

                                for e in range(N):
                                    if d_labels.count(e) > M:
                                        (
                                            e_labels,
                                            remapped_e_cluster_pairs,
                                            CF_index_map,
                                            level_six_reverse_fcts,
                                        ) = one_further_indexing(
                                            e,
                                            d_labels,
                                            remapped_d_cluster_pairs,
                                            N,
                                            CF_index_map,
                                            level_five_reverse_fcts,
                                            6,
                                        )
                                        level_six.append((a, b, c, d, e, e_labels))

                                        for f in range(N):
                                            if e_labels.count(f) > M:
                                                (
                                                    f_labels,
                                                    remapped_f_cluster_pairs,
                                                    CF_index_map,
                                                    level_seven_reverse_fcts,
                                                ) = one_further_indexing(
                                                    f,
                                                    e_labels,
                                                    remapped_e_cluster_pairs,
                                                    N,
                                                    CF_index_map,
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
                                                            remapped_g_cluster_pairs,
                                                            CF_index_map,
                                                            level_eight_reverse_fcts,
                                                        ) = one_further_indexing(
                                                            g,
                                                            f_labels,
                                                            remapped_f_cluster_pairs,
                                                            N,
                                                            CF_index_map,
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


# In[43]:


print(len(CF_index_map.items()))


# In[44]:


########### save results here
CF_index_path = os.path.join(base_dir, task, CF_mode + "_CF_indices", CF_mode + "_c{}_{}_CF_index.json".format(number_of_clusters, maximum_cluster_size))
with open(
    CF_index_path, "w"
) as f:
    json.dump(CF_index_map, f, indent=2)


# In[6]:


import copy
user_CF_index_path = os.path.join(base_dir, task, "user_CF_indices", "user_c{}_{}_CF_index.json".format(number_of_clusters, maximum_cluster_size))

with open(
    user_CF_index_path, "r"
) as f:
    user_CF_index_map = json.load(f)
    print("length of remapped users: {}".format(len(user_CF_index_map.items())))
    
item_CF_index_path = os.path.join(base_dir, task, "item_CF_indices", "item_c{}_{}_CF_index.json".format(number_of_clusters, maximum_cluster_size))

with open(
    item_CF_index_path, "r"
) as f:
    item_CF_index_map = json.load(f)
    print("length of remapped items: {}".format(len(item_CF_index_map.items())))

deduplicate_mode = "user"

# deduplicate_mode = "item"
    
# deduplicate_mode = "useritem"

if deduplicate_mode == "user" or deduplicate_mode == "item":
    
    if deduplicate_mode == "user":
        duplicate_CF_index_map = copy.deepcopy(user_CF_index_map)
        size = num_of_users
    else:
        duplicate_CF_index_map = copy.deepcopy(item_CF_index_map)
        size = num_of_items
        
    duplicate_cluster = defaultdict(int)      

    deduplicated_CF_index_map = {}

    for i in range(0, size):
        id = " ".join(duplicate_CF_index_map[str(i)])

        duplicate_cluster[id] += 1

    id_count = {}
    for i in range(0, size):
        id = " ".join(duplicate_CF_index_map[str(i)])

        if duplicate_cluster[id] > 1:
            if id not in id_count.keys():
                id_count[id] = 0

            deduplicated_CF_index_map[i] = duplicate_CF_index_map[str(i)] + [str(id_count[id])]

            id_count[id] += 1
        else:
            deduplicated_CF_index_map[i] = duplicate_CF_index_map[str(i)]
    
    print("length of deduplicated items: {}".format(len(deduplicated_CF_index_map.items())))
    deduplicated_CF_index_path = os.path.join(base_dir, task, deduplicate_mode + "_CF_indices", deduplicate_mode + "_deduplicated_c{}_{}_CF_index.json".format(number_of_clusters, maximum_cluster_size))
    
    with open(
        deduplicated_CF_index_path, "w"
    ) as f:
        json.dump(deduplicated_CF_index_map, f, indent=2)

elif deduplicate_mode == "useritem":
    
    duplicate_CF_index_map = copy.deepcopy(user_CF_index_map)
    
    for i in range(0, num_of_items):
        duplicate_CF_index_map[str(i + num_of_users)] = item_CF_index_map[str(i)]
    
    print(len(duplicate_CF_index_map.items()))
    
    duplicate_cluster = defaultdict(int)      

    deduplicated_CF_index_map = {}

    for i in range(0, num_of_users + num_of_items):
        id = " ".join(duplicate_CF_index_map[str(i)])

        duplicate_cluster[id] += 1

    id_count = {}
    for i in range(0, num_of_users + num_of_items):
        id = " ".join(duplicate_CF_index_map[str(i)])

        if duplicate_cluster[id] > 1:
            if id not in id_count.keys():
                id_count[id] = 0

            deduplicated_CF_index_map[str(i)] = duplicate_CF_index_map[str(i)] + [str(id_count[id])]

            id_count[id] += 1
            
        else:
            deduplicated_CF_index_map[str(i)] = duplicate_CF_index_map[str(i)]
    
    deduplicated_user_CF_index_map = {}
    deduplicated_item_CF_index_map = {}
    
    for i in range(0, num_of_users):
        deduplicated_user_CF_index_map[str(i)] = deduplicated_CF_index_map[str(i)]
        
    for i in range(0, num_of_items):
        deduplicated_item_CF_index_map[str(i)] = deduplicated_CF_index_map[str(i + num_of_users)]
    
    deduplicated_user_CF_index_path = os.path.join(
        base_dir, task, deduplicate_mode + "_CF_indices", "user_deduplicated_c{}_{}_CF_index.json".
        format(number_of_clusters, maximum_cluster_size)
    )
    
    deduplicated_item_CF_index_path = os.path.join(
        base_dir, task, deduplicate_mode + "_CF_indices", "item_deduplicated_c{}_{}_CF_index.json".
        format(number_of_clusters, maximum_cluster_size)
    )
    
    with open(
        deduplicated_user_CF_index_path, "w"
    ) as f:
        json.dump(deduplicated_user_CF_index_map, f, indent=2)
    
    with open(
        deduplicated_item_CF_index_path, "w"
    ) as f:
        json.dump(deduplicated_item_CF_index_map, f, indent=2)


# In[ ]:




