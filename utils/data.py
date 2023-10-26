from sklearn.cluster import SpectralClustering
import random
import argparse
import json
from torch.utils.data import Dataset, DataLoader
from utils.prompt import (
    task_subgroup_1,
    task_subgroup_2,
    task_subgroup_3,
    task_subgroup_4,
    task_subgroup_5,
)
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import math
from torch.utils.data.distributed import DistributedSampler
from indexing.rep_method import (
    CF_representation,
    CF_representation_optimal_width,
    create_CF_embedding,
    create_CF_embedding_optimal_width,
    graph_representation,
    create_graph_embedding
)
import time
import numpy as np
# from utils import create_category_embedding
from indexing.CF_index import (
    construct_indices_from_cluster,
    construct_indices_from_cluster_optimal_width,
)

from indexing.graph_index import (
    construct_indices_from_graph
)
import os

def calculate_whole_word_ids(tokenized_text, input_ids):
    whole_word_ids = []
    curr = 0
    for i in range(len(tokenized_text)):
        if tokenized_text[i] == "<pad>":
            curr = 0
        if tokenized_text[i].startswith("▁"):
            curr += 1
            whole_word_ids.append(curr)
        else:
            whole_word_ids.append(curr)
    return whole_word_ids[: len(input_ids) - 1] + [0]  # [0] for </s>

def load_data(args, tokenizer):
    # set_seed(args)
    if args.data_order == "remapped_sequential":
        if args.remapped_data_order == "original":
            data_dir = os.path.join(args.data_dir, args.task, args.data_order, "data.txt")
            with open(data_dir, "r") as f:
                data = f.read()
            data = data.split("\n")[:-1]
    else:
        data_dir = os.path.join(args.data_dir, args.task, args.data_order, "data.txt")
        with open(data_dir, "r") as f:
            data = f.read()
        data = data.split("\n")[:-1]
        
    users = []
    all_items = []
    remapped_all_items = []
    train_sequence = []
    val_sequence = []
    test_sequence = []

    item_remap_fct = None
    user_remap_fct = None
    
    # indexing for item representation

    if args.item_representation == "no_tokenization":
        item_remap_fct = no_tokenization(args)

    elif args.item_representation == "CID":
        assert args.data_order == "remapped_sequential"
        if not args.optimal_width_in_CF:
            print("--- do not use optimal width in CF ---")
            item_mapping, _ = construct_indices_from_cluster(args, mode="item")
            item_remap_fct = lambda x: CF_representation(x, item_mapping, mode="item")
        else:
            print("--- do use optimal width in CF ---")
            item_mapping, _ = construct_indices_from_cluster_optimal_width(args, mode="item")
            item_remap_fct = lambda x: CF_representation_optimal_width(x, item_mapping)
        # print("---finish loading CF mapping---")
    
    elif args.item_representation == "GID":
        assert args.data_order == "remapped_sequential"
        item_mapping, _ = construct_indices_from_graph(args, mode="item")
        item_remap_fct = lambda x: graph_representation(x, item_mapping, mode="item")
    
    # indexing for user representation
    if args.user_representation == "CID":
        user_mapping, _ = construct_indices_from_cluster(args, mode="user")
        user_remap_fct = lambda x: CF_representation(x, user_mapping, mode="user")
        
        user2idpath = os.path.join(args.data_dir, args.task, "user_CF_indices", "userid_map.json")
        with open(user2idpath, "r") as f:
            user2id = json.load(f)
    
    elif args.user_representation == "GID":
        user_mapping, _ = construct_indices_from_graph(args, mode="user")
        user_remap_fct = lambda x: graph_representation(x, user_mapping, mode="user")
        
    
    splittion_c = " "
    # remap data
    for one_user in tqdm(data):
        splittion_point = one_user.index(splittion_c)
        user = one_user[:splittion_point]
        items = one_user[splittion_point + 1 :].split(" ")
        
        # print(args.item_representation)
        if args.item_representation != "None":
            remapped_items = []
            for item in items:
                remapped_item = item_remap_fct(item)
                if remapped_item is None:
                    print("user id: ", user)
                    print("item id: ", item)
                remapped_items.append(remapped_item)
        else:
            # remapped_items = items
            remapped_items = [str(int(item) + 1000) for item in items]
        if args.user_representation != "None":
            if args.user_representation == "CID":
                remapped_user = user_remap_fct(user2id[user])
            elif args.user_representation == "GID":
                remapped_user = user_remap_fct(user)
        else:
            remapped_user = user
        users.append(remapped_user)
        train_sequence.append(remapped_items[:-2]) # 1 to n-2 as training samples
        val_sequence.append(remapped_items[:-1]) # n-1 as validation samples
        test_sequence.append(remapped_items) # n-2 as test samples
        all_items += items
        remapped_all_items += remapped_items

    remove_extra_items = list(
        set([(a, b) for a, b in zip(all_items, remapped_all_items)])
    )
    remove_extra_items = sorted(remove_extra_items, key=lambda x: x[0])
    all_items = [pair[0] for pair in remove_extra_items]
    remapped_all_items = [pair[1] for pair in remove_extra_items]
    
    print("Number of remapped items: {}".format(len(remapped_all_items)))

    return (
        users,
        all_items,
        train_sequence,
        val_sequence,
        test_sequence,
        remapped_all_items,
    )
    

class direct_straightforward_dataset(Dataset):
    def __init__(self, args, users, all_items, test_history, task_group):
        super().__init__()
        self.args = args
        self.users = users
        self.all_items = all_items
        self.all_history = test_history
        self.templates = task_group
        self.num_template = len(self.templates)
        self.train_history = [d[:-2] for d in self.all_history]
        self.number_of_interactions()

    def number_of_interactions(self):
        self.user_interaction_code = []
        total_num = 0
        for k, v in zip(self.users, self.train_history):
            number = len(v)
            # number = 3 # make dataset more balanced
            # number = min(20, len(v))
            total_num += number
            for _ in range(number):
                self.user_interaction_code.append(k)
        return total_num

    def __len__(self):
        length = self.number_of_interactions() * self.num_template
        return length

    def __getitem__(self, index):
        index = index // self.num_template

        user_idx = self.user_interaction_code[index]
        the_number_of_datapoint = self.users.index(user_idx)
        sequence = self.train_history[the_number_of_datapoint]
    
        # target_item = random.choice(sequence[:-2])
        # length = min(8, len(sequence))
        length = len(sequence)
        target_item_index = random.randint(0, length-1)
        
        target_item = sequence[target_item_index]
        
        template_idx = index % self.num_template
        template = self.templates[template_idx]
        
#         items = []
#         for item_idx in range(length):
#             if item_idx != target_item_index:
#                 items.append(sequence[item_idx])
        
#         items = random.sample(items, k=length-1)
        
#         assert target_item not in items
        
        if template["input_first"] == "user":
            input_sent = template["source"].format(
                user_idx, 
                # " , ".join(["item_" + item_idx for item_idx in items])
            )
            # input_sent += ("purchased " + "item_".join(items))
            output_sent = template["target"].format("item_" + target_item)
        else:
            input_sent = template["source"].format(user_idx)
            output_sent = template["target"].format("item_" + target_item)
        
        # print(sequence)
        return input_sent, output_sent
        # return input_sent, output_sent, sequence


class Collator:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_texts = [input_text[0] for input_text in batch]
        output_texts = [input_text[1] for input_text in batch]
        # purchase_sequence = [input_text[2] for input_text in batch]
        # max_history_length = self.args.max_history_length
        # print(purchase_sequence)
        # padded_purchase_sequence = []
        # for p_s in purchase_sequence:
            # padded_purchase_sequence.append([int(p) for p in p_s] + [-1] * (max_history_length - len(p_s)))
        
        # purchase_sequence = padded_purchase_sequence
        inputs = self.tokenizer.batch_encode_plus(
            input_texts, padding="longest", truncation=True, max_length=512
        )
        input_ids = inputs["input_ids"]
        whole_word_ids = []
        for input_id in input_ids:
            tokenized_text = self.tokenizer.convert_ids_to_tokens(input_id)
            if (
                self.args.whole_word_embedding == "shijie"
                or self.args.whole_word_embedding == "None"
            ):
                if self.args.item_representation != "title":
                    whole_word_id = calculate_whole_word_ids(tokenized_text, input_id)
                    # whole_word_id = calculate_whole_word_ids_remove_number_begin(
                    #    tokenized_text, input_id
                    # )
                else:
                    whole_word_id = calculate_whole_word_ids_title(
                        tokenized_text, input_id
                    )
            else:
                assert self.args.whole_word_embedding == "position_embedding"
                whole_word_id = position_embedding(self.args, tokenized_text, input_id)
            whole_word_ids.append(whole_word_id)
        input_attention = inputs["attention_mask"]
        outputs = self.tokenizer.batch_encode_plus(
            output_texts, padding="longest", truncation=True, max_length=512
        )
        output_ids = outputs["input_ids"]
        output_attention = outputs["attention_mask"]
        
        # print(torch.tensor(purchase_sequence).shape)

        return (
            torch.tensor(input_ids),
            torch.tensor(input_attention),
            torch.tensor(whole_word_ids),
            torch.tensor(output_ids),
            torch.tensor(output_attention),
            # torch.tensor(purchase_sequence)
        )


def load_train_dataloaders(
    args, tokenizer, users, all_items, train_sequence, test_sequence
):
    # load data
    # users, all_items, train_sequence, val_sequence, test_sequence = load_data(
    #    args, tokenizer
    # )

    task_data_lengths = []

    # create direct straightforward dataset
    train_direct_straightforward_dataset = direct_straightforward_dataset(
        args, users, all_items, test_sequence, task_subgroup_5
    )
    train_direct_straightforward_dataset_length = math.ceil(
        len(train_direct_straightforward_dataset)
        / args.train_direct_straightforward_batch
    )
    task_data_lengths.append(train_direct_straightforward_dataset_length)

    # create collator
    collator = Collator(args, tokenizer)

    # create sampler
    if args.distributed:
        sampler = DistributedSampler(train_direct_straightforward_dataset)
    else:
        sampler = None
    train_direct_straightforward_dataloader = DataLoader(
        train_direct_straightforward_dataset,
        batch_size=args.train_direct_straightforward_batch,
        collate_fn=collator,
        shuffle=(sampler is None),
        sampler=sampler,
    )

    return (
        train_direct_straightforward_dataloader,
        task_data_lengths,
        all_items,
    )


############### evaluation dataset ###############

class evaluation_direct_straightforward_dataset(Dataset):
    def __init__(self, args, users, eval_sequence, task_group, mode):
        super().__init__()
        self.args = args
        self.mode = mode
        self.evaluation_template_id = self.args.evaluation_template_id
        self.users = users
        self.eval_history = eval_sequence
        self.template = task_group[self.evaluation_template_id]

    def __len__(self):
        number = len(self.users)
        return number

    def __getitem__(self, index):
        user_idx = self.users[index]
        sequence = self.eval_history[index]
        if self.args.remove_last_item:
            if self.mode == "validation":
                purchase_history = sequence[:-1]
            else:
                assert self.mode == "test"
                purchase_history = sequence[:-2]
        elif self.args.remove_first_item:
            if self.mode == "validation":
                purchase_history = sequence[:-1]
            else:
                assert self.mode == "test"
                purchase_history = sequence[1:-1]
        else:
            purchase_history = sequence[:-1]
            # if self.mode == "validation":
            #     purchase_history = sequence[:-2]
            # else:
            #     purchase_history = sequence[:-1]
        
        # purchase_history = random.sample(purchase_history, k=min(5, len(purchase_history)))
        # if self.mode == "test":
        #     target_item = sequence[-1]
        # else:
        #     target_item = sequence[-2]
        target_item = sequence[-1]
        
        # assert target_item not in purchase_history
        
        if self.template["input_first"] == "user":
            input_sent = self.template["source"].format(
                user_idx,
                # " , ".join(["item_" + item_idx for item_idx in purchase_history]),
            )
        # else:
        #     input_sent = self.template["source"].format(
        #         " , ".join(["item_" + item_idx for item_idx in purchase_history]),
        #         user_idx,
        #     )
        output_sent = self.template["target"].format("item_" + target_item)

        return input_sent, output_sent


def load_eval_dataloaders(
    args, tokenizer, method, mode, users, all_items, val_sequence, test_sequence,
):
    # load data
    # users, all_items, _, val_sequence, test_sequence = load_data(args, tokenizer)

    if mode == "validation":
        eval_sequence = val_sequence
    else:
        eval_sequence = test_sequence

    collator = Collator(args, tokenizer)
    
    if method == "direct_straightforward":
        assert method == "direct_straightforward"
        eval_direct_straightforward_dataset = evaluation_direct_straightforward_dataset(
            args, users, eval_sequence, task_subgroup_5, mode
        )
        # create sampler
        if args.distributed:
            sampler = DistributedSampler(eval_direct_straightforward_dataset)
        else:
            sampler = None
        dataloader = DataLoader(
            eval_direct_straightforward_dataset,
            batch_size=args.eval_direct_straightforward_batch,
            collate_fn=collator,
            shuffle=(sampler is None),
            sampler=sampler,
        )

    return dataloader