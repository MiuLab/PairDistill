
DATA_DIR="/work/cwhuang0921/fine-grained-distillation/data/msmarco"
CHECKPOINT="/work/cwhuang0921/fine-grained-distillation/dpr-scale/multirun/2024-02-06/13-01-47/0/lightning_logs/version_0/checkpoints/last.ckpt"
SUFFIX="_pair_r1"

PYTHONPATH=.:$PYTHONPATH HYDRA_FULL_ERROR=1 python3 dpr_scale/generate_embeddings.py -m \
    datamodule=generate \
    datamodule.test_path=${DATA_DIR}/collection_dpr.tsv \
    datamodule.test_batch_size=256 \
    datamodule.use_title=False \
    task.transform.max_seq_len=128 \
    +task.ctx_embeddings_dir=${DATA_DIR}/embeddings/ctx${SUFFIX} \
    +task.checkpoint_path=${CHECKPOINT} \
    +trainer.precision=16 \
    trainer.gpus=6


PYTHONPATH=.:$PYTHONPATH HYDRA_FULL_ERROR=1 python3 dpr_scale/generate_query_embeddings.py -m \
    datamodule=generate_query_emb \
    +datamodule.test_path=${DATA_DIR}/queries.dev.small.tsv \
    +datamodule.trec_format=True \
    +task.ctx_embeddings_dir=${DATA_DIR}/embeddings/query${SUFFIX} \
    datamodule.test_batch_size=128 \
    task.transform.max_seq_len=64 \
    +task.checkpoint_path=${CHECKPOINT} \
    trainer.gpus=1
