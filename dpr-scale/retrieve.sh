
DATA_DIR="/work/cwhuang0921/fine-grained-distillation/data/msmarco"
SUFFIX="_pair_r1"
PATH_TO_RUNFILE=${DATA_DIR}/embeddings/query${SUFFIX}/run.txt

PYTHONPATH=.:$PYTHONPATH python dpr_scale/run_retrieval_pytorch.py \
--ctx_embeddings_dir=${DATA_DIR}/embeddings/ctx${SUFFIX} \
--query_emb_path=${DATA_DIR}/embeddings/query${SUFFIX}/query_reps.pkl \
--questions_tsv_path=${DATA_DIR}/queries.dev.small.tsv \
--passages_tsv_path=${DATA_DIR}/collection_dpr.tsv \
--output_runfile_path=${PATH_TO_RUNFILE} \
--trec_format \
--topk=1000

../anserini-tools/eval/trec_eval.9.0.4/trec_eval -c -M 10 -m recip_rank ${DATA_DIR}/qrels.dev.small.tsv $PATH_TO_RUNFILE
../anserini-tools/eval/trec_eval.9.0.4/trec_eval -c -m ndcg_cut.10 ${DATA_DIR}/qrels.dev.small.tsv $PATH_TO_RUNFILE
../anserini-tools/eval/trec_eval.9.0.4/trec_eval -c -m recall.100,1000 ${DATA_DIR}/qrels.dev.small.tsv $PATH_TO_RUNFILE
