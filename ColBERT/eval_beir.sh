
DATASETS="arguana bioasq climate-fever fever fiqa hotpotqa nfcorpus nq quora robust04 scidocs scifact signal1m trec-covid trec-news webis-touche2020"
# Associative array of name-directory pairs where the directory stores the ranking files, we'll evaluate all of them
declare -A RETRIEVE_DIRS
RETRIEVE_DIRS["colbertv2-pair3.0-cont2-2-len300"]="experiments/official-neg63-point-mini-pair-r2-4-colbertv2-pair3.0-cont2-2/index/2024-02-03_00.04.06/"
RETRIEVE_DIRS["colbertv2-pair3.0-cont2-2"]="experiments/official-neg63-point-mini-pair-r2-4-colbertv2-pair3.0-cont2-2/index/2024-02-02_16.15.31"
RETRIEVE_DIRS["colbertv2-len300"]="experiments/colbertv2/index/2024-02-03_12.33.04/"
RETRIEVE_DIRS["colbertv2"]="experiments/colbertv2/index/2024-01-19_14.33.06/"

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

        if [ ${dataset} == "arguana" ]; then
            rm ${RETRIEVE_DIR}/${dataset}.run
        fi

        # If the file ${RETRIEVE_DIR}/${dataset}.run does not exist, then convert the ranking file to TREC format
        if [ ! -f ${RETRIEVE_DIR}/${dataset}.run ]; then
            if [ ${dataset} == "arguana" ]; then
                # create a temp file
                TMPFILE=$(mktemp)

                python3 ../filter_duplicate_ids.py \
                    --input_file ${RETRIEVE_DIR}/${dataset}.nbits=2.ranking.tsv \
                    --output_file ${TMPFILE} \
                    --duplicate_ids ../data/beir/${dataset}/duplicate_qid_pid.json
                mv ${TMPFILE} ${RETRIEVE_DIR}/${dataset}.nbits=2.ranking.tsv
            fi

            python3 utility/evaluate/convert_to_trec.py \
                --input ${RETRIEVE_DIR}/${dataset}.nbits=2.ranking.tsv \
                --output ${RETRIEVE_DIR}/${dataset}.run
        fi

        # Evaluate the ranking file and print out the NDCG@10 score
        SCORE=$(../anserini-tools/eval/trec_eval.9.0.4/trec_eval -c -m ndcg_cut.10 \
            ../data/beir/${dataset}/qrels.test.tsv \
            ${RETRIEVE_DIR}/${dataset}.run | grep ndcg_cut_10 | awk '{print $3}')

        printf "%s\t" ${SCORE}
    done

    printf "\n"
done
