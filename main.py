from sklearn.cluster import SpectralClustering
import argparse
from transformers import AutoTokenizer,T5Config, T5ForConditionalGeneration
from utils.data import load_train_dataloaders, load_eval_dataloaders, load_data
import copy
from tqdm import tqdm
from utils.utils import (
    set_seed,
    Logger,
    create_optimizer_and_scheduler,
    exact_match,
    prefix_allowed_tokens_fn,
    load_model,
    random_initialization,
    # create_category_embedding,
    # create_category_embedding_yelp,
    # content_category_embedding_modified_yelp,
    # content_based_representation_non_hierarchical,
)
from indexing.rep_method import (
    create_CF_embedding,
    create_CF_embedding_optimal_width,
    create_graph_embedding
)
from modeling.modeling_p5 import P5
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# from transformers import T5Config

import os

# os.environ["TRANSFORMERS_CACHE"]="/common/users/km1558/huggingface/cache/"

import transformers
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import time
from utils.generation_trie import Trie
import os
from collections import OrderedDict

transformers.logging.set_verbosity_error()

import warnings

warnings.filterwarnings("ignore")


def predict_outputs(args, batch, model, tokenizer, logger, k=20, prefix_allowed_tokens=None):
    input_ids = batch[0].to(args.gpu)
    attn = batch[1].to(args.gpu)
    whole_input_ids = batch[2].to(args.gpu)
    output_ids = batch[3].to(args.gpu)

    batch_size = batch[0].shape[0]

    if args.whole_word_embedding == "None":
        if args.distributed:
            prediction = model.module.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_length=8,
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                num_beams=20,
                num_return_sequences=20,
                # whole_word_embedding_type=args.whole_word_embedding,
                output_scores=True,
                return_dict_in_generate=True,
            )
        else:
            prediction = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_length=8,
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                num_beams=20,
                num_return_sequences=20,
                # whole_word_embedding_type=args.whole_word_embedding,
                output_scores=True,
                return_dict_in_generate=True,
            )
    else:
        # k = 1
        if args.distributed:
            prediction = model.module.generate(
                input_ids=input_ids,
                attention_mask=attn,
                whole_word_ids=whole_input_ids,
                max_length=8,
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                num_beams=20,
                num_return_sequences=20,
                # whole_word_embedding_type=args.whole_word_embedding,
                output_scores=True,
                return_dict_in_generate=True,
            )
        else:
            prediction = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                whole_word_ids=whole_input_ids,
                max_length=8,
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                num_beams=20,
                num_return_sequences=20,
                # whole_word_embedding_type=args.whole_word_embedding,
                output_scores=True,
                return_dict_in_generate=True,
            )

    prediction_ids = prediction["sequences"]
    prediction_scores = prediction["sequences_scores"]
    
    # print("Prediction ID's shape: ", prediction_ids.shape)
    # print(prediction_scores.shape)

    if args.item_representation not in [
        "no_tokenization",
        "item_resolution",
    ]:
        gold_sents = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        generated_sents = tokenizer.batch_decode(
            prediction_ids, skip_special_tokens=True
        )
        
        
        assert len(generated_sents) == batch_size * 20
        assert len(prediction_scores) == batch_size * 20
            
    else:
        gold_sents = [
            a.replace("<pad>", "").replace("</s>", "").replace(" ", "")
            for a in tokenizer.batch_decode(output_ids)
        ]
        generated_sents = [
            a.replace("<pad>", "").replace("</s>", "").replace(" ", "")
            for a in tokenizer.batch_decode(prediction_ids)
        ]
    hit_5, hit_10, hit_20, ncdg_5, ncdg_10, ncdg_20 = exact_match(
        logger, generated_sents, gold_sents, prediction_scores, 20
    )

    return hit_5, hit_10, hit_20, ncdg_5, ncdg_10, ncdg_20


def trainer(
    args,
    rank,
    train_loaders,
    val_loader,
    test_loader,
    remapped_all_items,
    batch_per_epoch,
    tokenizer,
    logger,
):

    # if rank == 0:
    #     logger.log("loading model ...")
        # logger.log("using only sequential data, and all possible sequences are here")
    # use default T5 config
    config = T5Config.from_pretrained(
        args.model_type,
        # cache_dir="/common/users/zl502/huggingface/cache"
    )
    
    config.dropout_rate = args.dropout
    if args.no_pretrain:
        if rank == 0:
            logger.log("do not use pretrained weights")
        model = P5(config=config)
    else:
        if rank == 0:
            logger.log("use pretrained weights")
        
        # model = P5.from_pretrained(
        #     pretrained_model_name_or_path=args.model_type,
        #     config=config,
        #     # **model_args,  # , args=args
        # )  # .to(args.gpu)
        
        t5_model = P5.from_pretrained(
            pretrained_model_name_or_path='t5-small',
            config=config,
            # cache_dir="/common/users/zl502/huggingface/cache"
            # **model_args,  # , args=args
        )  # .to(args.gpu)
        # print(t5_model)

        state_dict = t5_model.state_dict()

        # print(state_dict.items())

        customized_config = T5Config.from_pretrained('t5-small')
        customized_config.d_ff = args.ffn_width

        model = P5(config=customized_config)

        if not args.load_checkpoint:
            reduced_state_dict = {k: v for k, v in state_dict.items() if "Dense" not in k}

            model.load_state_dict(reduced_state_dict, strict=False)
        
        # for k in model.state_dict().items
        # print("Reduced model parameter numbers: {}".format(model.num_parameters()))
        # print(model)
        
    if not args.eval_only:
        if args.random_initialization_embedding and not args.load_checkpoint:
            # if rank == 0:
            #     logger.log("randomly initialize number-related embeddings only")
            model = random_initialization(model, tokenizer)

    model.resize_token_embeddings(len(tokenizer))
    model.to(args.gpu)

    optimizer, scheduler = create_optimizer_and_scheduler(
        args, logger, model, batch_per_epoch
    )

    if args.distributed:
        dist.barrier()

    if args.multiGPU:
        # if rank == 0:
        #     logger.log("model dataparallel set")
        if args.distributed:
            model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)

    if rank == 0:
        logger.log("Start Training.")
    model.zero_grad()
    logging_step = 0
    logging_loss = 0
    best_validation_recall = 0
    best_test_recall = 0
    
    number_of_tasks = 1
    
    start_epoch = 0
        
    if args.load_checkpoint:
        print("load model weights from checkpoint.")
        state = torch.load(args.model_dir + "_latest.pt")
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        start_epoch = state['epoch'] + 1
        best_validation_recall = state['best_test_recall']
        print("training from epoch: {}".format(start_epoch))
        logging_step = start_epoch * args.warmup_steps
    
    for epoch in range(start_epoch, args.epochs):
        if rank == 0:
            logger.log("---------- training epoch {} ----------".format(epoch))
        if args.distributed:
            for loader in train_loaders:
                loader.sampler.set_epoch(epoch)

        if not args.eval_only:
            model.train()

            for batch in tqdm(train_loaders[-1]):

                input_ids = batch[0].to(args.gpu)
                attn = batch[1].to(args.gpu)
                whole_input_ids = batch[2].to(args.gpu)
                output_ids = batch[3].to(args.gpu)
                output_attention = batch[4].to(args.gpu)

                if args.whole_word_embedding == "None":
                    if args.distributed:
                        output = model.module(
                            input_ids=input_ids,
                            attention_mask=attn,
                            labels=output_ids,
                            alpha=args.alpha,
                            return_dict=True,
                            whole_word_embedding_type=args.whole_word_embedding,
                        )
                    else:
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attn,
                            labels=output_ids,
                            alpha=args.alpha,
                            return_dict=True,
                            whole_word_embedding_type=args.whole_word_embedding,
                        )
                else:
                    if args.distributed:
                        output = model.module(
                            input_ids=input_ids,
                            whole_word_ids=whole_input_ids,
                            attention_mask=attn,
                            labels=output_ids,
                            alpha=args.alpha,
                            return_dict=True,
                            whole_word_embedding_type=args.whole_word_embedding,
                        )
                    else:
                        output = model(
                            input_ids=input_ids,
                            whole_word_ids=whole_input_ids,
                            attention_mask=attn,
                            labels=output_ids,
                            alpha=args.alpha,
                            return_dict=True,
                            whole_word_embedding_type=args.whole_word_embedding,
                        )
                
                # if logging_step % args.logging_step == 0:
                #     print("Input ID: ", tokenizer.convert_ids_to_tokens(input_ids[0]))
                #     logits = output['logits']
                #     print("Ground truth ID: ", tokenizer.convert_ids_to_tokens(output_ids[0]))
                #     print("Prediction ID: ", tokenizer.convert_ids_to_tokens(torch.argmax(logits, dim=-1)[0]))

                # compute loss masking padded tokens
                loss = output["loss"]
                lm_mask = output_attention != 0
                lm_mask = lm_mask.float()
                B, L = output_ids.size()
                loss = loss.view(B, L) * lm_mask
                loss = (loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()

                logging_loss += loss.item()

                # update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                logging_step += 1

                if logging_step % args.logging_step == 0 and rank == 0:
                    logger.log(
                        "Total loss for {} steps : {}".format(
                            logging_step, logging_loss
                        )
                    )
                    logging_loss = 0
            if args.distributed:
                dist.barrier()

        if rank == 0:
            logger.log(
                "---------- start evaluation after epoch {} ----------".format(epoch)
            )
        for idx, remapped_item in enumerate(remapped_all_items):
            if remapped_item is None:
                print("ID: ", idx, "Invalid item")
        # if args.evaluation_method == "direct_straightforward":
        candidates = []
        for idx, candidate in enumerate(remapped_all_items):
            encoded_item = tokenizer.encode("{}".format("item_" + candidate))
            if encoded_item is None:
                print("ID: ", idx, " invalid_items")
            else:
                candidates.append([0] + encoded_item)
            
        candidate_trie = Trie(candidates)
        prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
        
        state = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_test_recall': best_test_recall
        }
        save_dir = args.model_dir + "_latest.pt"
        
        print("Save model to {}".format(save_dir))
        
        torch.save(state, save_dir)
        
        model.eval()
        correct_validation_5 = 0
        correct_validation_10 = 0
        correct_validation_20 = 0
        ncdg_validation_5 = 0
        ncdg_validation_10 = 0
        ncdg_validation_20 = 0

        validation_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                output_ids = batch[3]

                (
                    one_hit_5,
                    one_hit_10,
                    one_hit_20,
                    one_ncdg_5,
                    one_ncdg_10,
                    one_ncdg_20,
                ) = predict_outputs(
                    args, batch, model, tokenizer, logger, prefix_allowed_tokens=prefix_allowed_tokens
                )

                correct_validation_5 += one_hit_5
                correct_validation_10 += one_hit_10
                correct_validation_20 += one_hit_20
                ncdg_validation_5 += one_ncdg_5
                ncdg_validation_10 += one_ncdg_10
                ncdg_validation_20 += one_ncdg_20

                validation_total += output_ids.size(0)

            recall_validation_5 = correct_validation_5 / validation_total
            recall_validation_10 = correct_validation_10 / validation_total
            recall_validation_20 = correct_validation_20 / validation_total
            ncdg_validation_5 = ncdg_validation_5 / validation_total
            ncdg_validation_10 = ncdg_validation_10 / validation_total
            ncdg_validation_20 = ncdg_validation_20 / validation_total
            logger.log("validation hit @ 5 is {}".format(recall_validation_5))
            logger.log("validation hit @ 10 is {}".format(recall_validation_10))
            logger.log("validation hit @ 20 is {}".format(recall_validation_20))
            logger.log("validation ncdg @ 5 is {}".format(ncdg_validation_5))
            logger.log("validation ncdg @ 10 is {}".format(ncdg_validation_10))
            logger.log("validation ncdg @ 20 is {}".format(ncdg_validation_20))
        
        correct_test_5 = 0
        correct_test_10 = 0
        correct_test_20 = 0
        ncdg_test_5 = 0
        ncdg_test_10 = 0
        ncdg_test_20 = 0
        test_total = 0
        with torch.no_grad():
            for batch in tqdm(test_loader):
                output_ids = batch[3].to(args.gpu)

                (
                    one_hit_5,
                    one_hit_10,
                    one_hit_20,
                    one_ncdg_5,
                    one_ncdg_10,
                    one_ncdg_20,
                ) = predict_outputs(
                    args, batch, model, tokenizer, logger, prefix_allowed_tokens=prefix_allowed_tokens
                )

                correct_test_5 += one_hit_5
                correct_test_10 += one_hit_10
                correct_test_20 += one_hit_20
                ncdg_test_5 += one_ncdg_5
                ncdg_test_10 += one_ncdg_10
                ncdg_test_20 += one_ncdg_20

                test_total += output_ids.size(0)

            recall_test_5 = correct_test_5 / test_total
            recall_test_10 = correct_test_10 / test_total
            recall_test_20 = correct_test_20 / test_total
            ncdg_test_5 = ncdg_test_5 / test_total
            ncdg_test_10 = ncdg_test_10 / test_total
            ncdg_test_20 = ncdg_test_20 / test_total
            logger.log("test hit @ 5 is {}".format(recall_test_5))
            logger.log("test hit @ 10 is {}".format(recall_test_10))
            logger.log("test hit @ 20 is {}".format(recall_test_20))
            logger.log("test ncdg @ 5 is {}".format(ncdg_test_5))
            logger.log("test ncdg @ 10 is {}".format(ncdg_test_10))
            logger.log("test ncdg @ 20 is {}".format(ncdg_test_20))
            
        if recall_validation_20 > best_validation_recall:
            model_dir = "best_" + args.model_dir + ".pt"
            logger.log(
                "recall increases from {} ----> {} at epoch {}".format(
                    best_validation_recall, recall_validation_20, epoch
                )
            )
            if rank == 0:
                logger.log("save current best model to {}".format(model_dir))
                if args.distributed:
                    torch.save(model.module.state_dict(), model_dir)
                else:
                    torch.save(model.state_dict(), model_dir)
            best_validation_recall = recall_validation_20

        if args.distributed:
            dist.barrier()
            

def main_worker(local_rank, args, logger):
    set_seed(args)

    args.gpu = local_rank
    args.rank = local_rank
    logger.log(f"Process Launching at GPU {args.gpu}")

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(
            backend="nccl", world_size=args.world_size, rank=args.rank
        )

    logger.log(f"Building train loader at GPU {args.gpu}")

    if local_rank == 0:
        logger.log("loading data ...")

    # build tokenizers and new model embeddings
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_type,
        # cache_dir="/common/users/km1558/huggingface/cache" # needs to be replaced by your own cache directory
    )
    number_of_items = args.number_of_items

    if args.item_representation == "no_tokenization":
        if local_rank == 0:
            logger.log(
                "*** use no tokenization setting, highest resolution, extend vocab ***"
            )
        new_tokens = []
        for x in range(number_of_items):
            new_token = "<extra_id_{}>".format(x)
            new_tokens.append(new_token)
        new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
        tokenizer.add_tokens(list(new_tokens))

    elif args.item_representation == "CID":
        if local_rank == 0:
            logger.log(
                "*** use collaborative_filtering_based representation, extend vocab ***"
            )
            # print("Item CF setting: cluster number: {}, cluster size: {}".format(args.item_cluster_number, args.item_cluster_size))
        if not args.optimal_width_in_CF:
            tokenizer = create_CF_embedding(args, tokenizer, mode="item")
        else:
            tokenizer = create_CF_embedding_optimal_width(args, tokenizer, mode="item")
    
    elif args.item_representation == "GID":
        if local_rank == 0:
            logger.log(
                "*** use graph representation, extend vocab ***"
            )
        if not args.optimal_width_in_CF:
            tokenizer = create_graph_embedding(args, tokenizer, mode="item")

    # elif args.item_representation == "hybrid":
    #     if local_rank == 0:
    #         logger.log(
    #             "*** use hybrid_based representation using metadata and CF, extend vocab ***"
    #         )
    #     _, vocabulary = load_hybrid(args)
    #     tokenizer = create_hybrid_embedding(vocabulary, tokenizer)

    if args.item_representation == "remapped_sequential":
        if local_rank == 0:
            logger.log("*** use remapped sequential data ***")
        assert args.random_initialization_embedding
    
    if args.user_representation == "CID":
        if local_rank == 0:
            logger.log(
                "*** use CID representation ***"
            )
            print("User CF setting: cluster number: {}, cluster size: {}".format(args.user_cluster_number, args.user_cluster_size))
            
        if not args.optimal_width_in_CF:
            tokenizer = create_CF_embedding(args, tokenizer, mode="user")
        else:
            tokenizer = create_CF_embedding_optimal_width(args, tokenizer, mode="user")
    
    elif args.user_representation == "GID":
        if local_rank == 0:
            logger.log(
                "*** use GID representation ***"
            )
        if not args.optimal_width_in_CF:
            tokenizer = create_graph_embedding(args, tokenizer, mode="user")
        # else:
        #     tokenizer = create_CF_embedding_optimal_width(args, tokenizer, mode="user")
    
    (
        users,
        all_items,
        train_sequence,
        val_sequence,
        test_sequence,
        remapped_all_items,
    ) = load_data(args, tokenizer)

    (
        train_direct_straightforward_dataloader,
        task_data_lengths,
        remapped_all_items,
    ) = load_train_dataloaders(
        args, tokenizer, users, remapped_all_items, train_sequence, test_sequence
    )

    batch_per_epoch = len(train_direct_straightforward_dataloader)  # * 2
    train_loaders = [
        train_direct_straightforward_dataloader,
    ]

    if local_rank == 0:
        logger.log("finished loading data")
        logger.log("length of training data is {}".format(batch_per_epoch))

    val_loader = load_eval_dataloaders(
        args,
        tokenizer,
        args.evaluation_method,
        "validation",
        users,
        remapped_all_items,
        val_sequence,
        test_sequence,
    )

    test_loader = load_eval_dataloaders(
        args,
        tokenizer,
        args.evaluation_method,
        "test",
        users,
        remapped_all_items,
        test_sequence,
        test_sequence,
    )


    trainer(
        args,
        local_rank,
        train_loaders,
        val_loader,
        test_loader,
        remapped_all_items,
        batch_per_epoch,
        tokenizer,
        logger,
    )


def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--logging_dir", type=str, default="beauty.log")
    parser.add_argument("--model_dir", type=str, default="pretrain_t5_small_beauty.pt")
    parser.add_argument("--task", type=str, default="beauty")

    parser.add_argument("--max_history", type=int, default=20)
    parser.add_argument("--sequential_num", type=int, default=10)
    parser.add_argument("--negative_sample", type=int, default=2)
    parser.add_argument("--direct_item_proportion", type=int, default=2)

    parser.add_argument("--train_direct_straightforward_batch", type=int, default=48)
    
    parser.add_argument("--eval_direct_straightforward_batch", type=int, default=64)

    # learning hyperparameters
    parser.add_argument("--model_type", type=str, default="t5-small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("--logging_step", type=int, default=100)
    parser.add_argument("--warmup_prop", type=float, default=0.05)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=2)

    # CPU/GPU
    parser.add_argument("--multiGPU", action="store_const", default=False, const=True)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("--evaluation_method", type=str, default="direct_straightforward")
    parser.add_argument("--evaluation_template_id", type=int, default=0)
    
    parser.add_argument("--ffn_width", type=int, default=16)

    parser.add_argument(
        "--number_of_items",
        type=int,
        default=12102,
        help="number of items in each dataset, beauty 12102, toys 11925, sports 18358",
    )

    # item representation experiment setting
    parser.add_argument(
        "--item_representation",
        type=str,
        default="None",
        help="CID,GID,None",
    )
    
    parser.add_argument(
        "--user_representation",
        type=str,
        default="None",
        help="CID,GID,None",
    )
    
    # arguments for collaborative indexing
    
    parser.add_argument(
        "--co_indexing",
        action="store_true"
    )
    
    parser.add_argument(
        "--separate_indexing",
        action="store_true"
    )
    
    parser.add_argument(
        "--user_cluster_number",
        type=int,
        default=5,
        help="number of clusters to divide every step when user representation method is CF",
    )
    
    parser.add_argument(
        "--item_cluster_number",
        type=int,
        default=5,
        help="number of clusters to divide every step when item representation method is CF",
    )
    
    parser.add_argument(
        "--item_cluster_size",
        type=int,
        default=30,
        help="number of items in the largest clusters",
    )
    
    parser.add_argument(
        "--user_cluster_size",
        type=int,
        default=30,
        help="number of users in the largest clusters",
    )
    
    parser.add_argument(
        "--optimal_width_in_CF",
        action="store_true",
        help="whether to use eigengap heuristics to find the optimal width in CF, all repetition",
    )
    
    parser.add_argument(
        "--last_token_no_repetition",
        action="store_true"
    )

    parser.add_argument(
        "--data_order",
        type=str,
        default="random",
        help="random or remapped_sequential (excluding validation and test)",
    )
    
    # arguments for GID
    parser.add_argument(
        "--user_quantized_len",
        type=int,
        default=4,
        help="quantization length for user indexing",
    )
    
    parser.add_argument(
        "--item_quantized_len",
        type=int,
        default=4,
        help="quantization length for item indexing",
    )
    
    parser.add_argument(
        "--embedding_length",
        type=int,
        default=64,
        help="length for embedding",
    )
    
    # arguments for sequential indexing
    parser.add_argument(
        "--remapped_data_order",
        type=str,
        default="original",
        help="original (original file), short_to_long, long_to_short, randomize, used when item_representation == remapped_sequential",
    )

    # for None or remapped sequential
    parser.add_argument(
        "--random_initialization_embedding",
        action="store_true",
        help="randomly initialize number related tokens, use only for random_number setting",
    )

    # whether to use whole word embedding and how
    parser.add_argument(
        "--whole_word_embedding",
        type=str,
        default="shijie",
        help="shijie, None, position_embedding",
    )

    # whether to use pretrain
    parser.add_argument(
        "--no_pretrain", action="store_true", help="does not use pretrained weights of T5"
    )
    
    # whether modify the evaluation setting
    parser.add_argument(
        "--remove_last_item",
        action="store_true",
        help="remove last item in a sequence in test time",
    )
    parser.add_argument(
        "--remove_first_item",
        action="store_true",
        help="remove first item in a sequence in test time",
    )

    parser.add_argument(
        "--check_data",
        action="store_true",
        help="check whether data are correctly formated and whether consistent across GPUs",
    )
    
    parser.add_argument(
        "--load_checkpoint",
        action="store_true",
        help="whether to load trained checkpoint",
    )
    
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="whether to evaluate only",
    )
    

    args = parser.parse_args()

    if args.task == "beauty":
        args.number_of_items = 12101
        args.max_history_length = 204
        
    elif args.task == "toys":
        args.number_of_items = 11924
        args.max_history_length = 550
        
    elif args.task == "sports":
        args.number_of_items = 18358
        args.max_history_length = 296
        
    elif args.task == "ml-1m":
        args.number_of_items = 3707
        args.max_history_length = 2314
        
    elif args.task == "lastfm":
        args.number_of_items = 3646
        args.max_history_length = 899
        
    elif args.task == "taobao":
        args.number_of_items = 4193
        args.max_history_length = 259
        
    else:
        assert args.task == "yelp"
        args.number_of_items = 20034
        args.max_history_length = 350

    return args


if __name__ == "__main__":
    transformers.logging.set_verbosity_error()

    cudnn.benchmark = True
    args = parse_argument()

    set_seed(args)
    logger = Logger(args.logging_dir, True)
    logger.log(str(args))

    # number of visible gpus set in os[environ]
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    if args.distributed:
        mp.spawn(
            main_worker, args=(args, logger), nprocs=args.world_size, join=True,
        )
    else:
        main_worker(local_rank=0, args=args, logger=logger)
