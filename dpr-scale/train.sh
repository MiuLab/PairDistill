
DATA_DIR="/work/cwhuang0921/fine-grained-distillation/data/msmarco/"

PYTHONPATH=.:$PYTHONPATH HYDRA_FULL_ERROR=1 python3 dpr_scale/main.py -m --config-name msmarco_pair_distill.yaml \
    task.transform.max_seq_len=128