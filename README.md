# LightLM: A Lightweight Deep and Narrow Language Model for Generative Recommendation

## Environment
``conda create -n LightLM python=3.9 && conda activate LightLM``  
``pip install -r requirements.txt``

## Usage 
Here we use the Toys dataset as an example.\\
Spectral Collaborative Indexing (CoUI)  
``
CUDA_VISIBLE_DEVICES=0 python \ \\
   main.py \
    --task toys \
    --seed 2022 \
    --warmup_prop 0.05 \
    --lr 1e-3 \
    --clip 1.0 \
    --model_type 't5-small' \
    --epochs 8 \  
    --gpu '0' \  
    --data_dir data \  
    --logging_step 100 \  
    --logging_dir 'log/pretrain_dn_t5_small_toys_co_useritem_CF_50.log' \  
    --model_dir 'model/pretrain_dn_t5_small_toys_co_useritem_CF_50' \  
    --train_direct_straightforward_batch 64 \  
    --eval_direct_straightforward_batch 32 \  
    --ffn_width 16 \  
    --whole_word_embedding shijie \  
    --random_initialization_embedding \  
    --item_representation CID \  
    --user_representation CID \  
    --random_initialization_embedding \  
    --data_order remapped_sequential \  
    --remapped_data_order original \  
    --co_indexing \  
    --user_cluster_num 50 \  
    --user_cluster_size 100 \  
    --item_cluster_num 50 \  
    --item_cluster_size 100
``

Graph Collaborative Indexing (CoUI)
``
CUDA_VISIBLE_DEVICES=0 python \  
    main.py \  
    --task toys \  
    --seed 2022 \  
    --warmup_prop 0.05 \  
    --lr 1e-3 \  
    --clip 1.0 \  
    --model_type 't5-small' \  
    --epochs 20 \  
    --gpu '0' \  
    --data_dir data/ \  
    --logging_step 100 \  
    --logging_dir 'log/pretrain_dn_t5_small_toys_co_useritem_graph.log' \  
    --model_dir 'model/pretrain_dn_t5_small_toys_co_useritem_graph' \  
    --train_direct_straightforward_batch 64 \  
    --eval_direct_straightforward_batch 32 \  
    --ffn_width 16 \  
    --whole_word_embedding shijie \  
    --random_initialization_embedding \  
    --item_representation GID \  
    --user_representation GID \  
    --random_initialization_embedding \  
    --data_order remapped_sequential \  
    --remapped_data_order original \  
    --co_indexing \  
    --user_quantized_len 4 \  
    --item_quantized_len 4
``
