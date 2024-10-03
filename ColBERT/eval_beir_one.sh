#!/bin/bash

# Usage: bash eval_beir_one.sh <runfile> <qrels>
if [ "$#" -ne 2 ]; then
    echo "Usage: bash eval_beir_one.sh <runfile> <qrels>"
    exit
fi

runfile=$1
qrels=$2

tempfile=$(mktemp)
python3 utility/evaluate/convert_to_trec.py \
    --input ${runfile} \
    --output ${tempfile}

../anserini-tools/eval/trec_eval.9.0.4/trec_eval -c -m ndcg_cut.10 \
    ${qrels} \
    ${tempfile}