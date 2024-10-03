#!/bin/bash

# export TRANSFORMERS_CACHE=$(pwd)/cache

# for i in {0..3}; do
# 	CUDA_VISIBLE_DEVICES=${i} python3 pairwise_rerank.py \
#         --runfile ColBERT/experiments/official-neg63-point-mini-pair-r2-4-colbertv2/train/2024-01-21_20.34.09/x0${i}.mini \
#         --collection data/msmarco/collection.tsv \
#         --queries data/msmarco/queries.dev.small.tsv \
#         --n_pairs 16 \
#         --topk 20 \
#         --from_point \
#         --output ColBERT/experiments/official-neg63-point-mini-pair-r2-4-colbertv2/train/2024-01-21_20.34.09/x0${i}.mini.duo &
# done
        # --output ColBERT/experiments/official-neg63-point-mini-pair/retrieve/2023-12-29_12.04.01/x0${i}.mini.duo &

# for i in {0..3}; do
# 	CUDA_VISIBLE_DEVICES=${i} python3 pointwise_rerank.py \
#         --runfile ColBERT/experiments/official-neg63-point-mini-pair-r2-4-colbertv2/train/2024-01-21_20.34.09/x0${i} \
#         --collection data/msmarco/collection.tsv \
#         --queries data/msmarco/queries.dev.small.tsv \
#         --n_passages 100 \
#         --generation_model \
#         --output ColBERT/experiments/official-neg63-point-mini-pair-r2-4-colbertv2/train/2024-01-21_20.34.09/x0${i}.mono &
# done

        # --model "cross-encoder/ms-marco-MiniLM-L-6-v2" \
        # --prev_rerank_file "ColBERT/experiments/official-neg63-point-mini-pair/retrieve/2023-12-29_12.04.01/msmarco.mini" \
        # --queries data/msmarco/queries.train.tsv \
        # --runfile ColBERT/experiments/official-neg63/retrieve/2023-12-22_08.49.47/msmarco.train.nbits=2.ranking.tsv.${i} \
        # --output ColBERT/experiments/official-neg63/retrieve/2023-12-22_08.49.47/msmarco.train.nbits=2.ranking.tsv.${i}.mini &

# for i in {0..1}; do
#     CUDA_VISIBLE_DEVICES=$((i+2)) HF_HOME=$(pwd)/cache python3 pointwise_rerank_instupr.py \
#         --runfile ColBERT/experiments/colbertv2/index/2024-02-03_12.33.04/climate-fever.nbits=2.ranking.tsv.0${i} \
#         --output ColBERT/experiments/colbertv2/index/2024-02-03_12.33.04/climate-fever.train.nbits=2.ranking.tsv.0${i}.instupr.point \
#         --collection data/beir/climate-fever_sampled/collection.tsv \
#         --queries data/beir/climate-fever_sampled/queries.train.tsv \
#         --n_passages 100 \
#         --use_title --batch_size 16 --max_length 360 &
# done

for i in {0..3}; do
    CUDA_VISIBLE_DEVICES=$((i+4)) HF_HOME=$(pwd)/cache python3 pairwise_rerank_instupr.py \
        --runfile ColBERT/experiments/colbertv2/index/2024-02-03_12.33.04/fiqa.train.nbits=2.ranking.tsv.instupr.point.0${i} \
        --output ColBERT/experiments/colbertv2/index/2024-02-03_12.33.04/fiqa.train.nbits=2.ranking.tsv.instupr.pair.0${i} \
        --collection data/beir/fiqa/collection.tsv \
        --queries data/beir/fiqa/queries.train.tsv \
        --topk 64 \
        --n_pairs 128 \
        --use_title --batch_size 16 --max_length 1024 --from_point &
done

