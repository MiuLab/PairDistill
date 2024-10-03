
DATASETS="android english gaming gis mathematica physics programmers stats tex unix webmasters wordpress"
# Associative array of name-directory pairs where the directory stores the ranking files, we'll evaluate all of them
declare -A RETRIEVE_DIRS
RETRIEVE_DIRS["colbertv2-pair3.0-cont2-2"]="experiments/official-neg63-point-mini-pair-r2-4-colbertv2-pair3.0-cont2-2/index/2024-02-07_15.41.29/"
RETRIEVE_DIRS["colbertv2"]="experiments/colbertv2/index/2024-02-08_15.46.57/"

# First, print out the names of the models in one line, tab separated
printf "Dataset\t"
for key in "${!RETRIEVE_DIRS[@]}"; do
    printf "%s\t" ${key}
done
printf "\n"

for dataset in ${DATASETS}; do
    printf "%s\t" ${dataset}

    for key in "${!RETRIEVE_DIRS[@]}"; do
        RETRIEVE_DIR=${RETRIEVE_DIRS[$key]}

        # If the file ${RETRIEVE_DIR}/${dataset}.run does not exist, then convert the ranking file to TREC format
        if [ ! -f ${RETRIEVE_DIR}/cqadupstack-${dataset}.run ]; then
            python3 utility/evaluate/convert_to_trec.py \
                --input ${RETRIEVE_DIR}/cqadupstack-${dataset}.nbits=2.ranking.tsv \
                --output ${RETRIEVE_DIR}/cqadupstack-${dataset}.run
        fi

        # Evaluate the ranking file and print out the NDCG@10 score
        SCORE=$(../anserini-tools/eval/trec_eval.9.0.4/trec_eval -c -m ndcg_cut.10 \
            ../data/beir/cqadupstack/${dataset}/qrels.test.tsv \
            ${RETRIEVE_DIR}/cqadupstack-${dataset}.run | grep ndcg_cut_10 | awk '{print $3}')

        printf "%s\t" ${SCORE}
    done

    printf "\n"
done
