PairDistill: Pairwise Relevance Distillation for Dense Retrieval
===

<p align="center">
📃 <a href="https://arxiv.org/abs/2410.01383" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/collections/chaoweihuang/pairdistill-66fe6b0cfa6eae4704df9f5e" target="_blank">Huggingface Collection</a> • <a href="https://drive.google.com/drive/folders/19jlgIBNHjuSsx5iQa3Mt7b_g3ZSZjJuh?usp=drive_link" target="_blank">Model & Dataset (Google Drive)</a>
</p>

Source code, trained models, and data of our paper **"PairDistill: Pairwise Relevance Distillation for Dense Retrieval"**, accepted to **EMNLP 2024 Main Conference**.

Please cite the following reference if you find our code, models, and datasets useful.

```
@inproceedings{huang2024pairdistill,
      title={PairDistill: Pairwise Relevance Distillation for Dense Retrieval}, 
      author={Chao-Wei Huang and Yun-Nung Chen},
      year={2024},
      booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024)}
}
```

<img width="1380" alt="image" src="https://github.com/user-attachments/assets/41c4c6d4-2934-4631-8135-58ca18336715">


## Overview
**PairDistill** is a pairwise relevance distillation framework designed to enhance the retrieval performance of dense retrieval models. PairDistill leverages the pairwise relevance signals to guide the distillation process. **PairDistill** achieves superior performance on [MS MARCO](https://microsoft.github.io/msmarco/), [BEIR](https://github.com/beir-cellar/beir), and [LoTTE](https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md).

![image](https://github.com/user-attachments/assets/1184c300-4767-4ad6-98d0-c631e1f64eca)

## Train PairDistill

### Preparation
#### Install Dependencies

Make a new Python 3.9+ environment using `virtualenv` or `conda`.

```bash
conda create -n pair-distill python=3.10
conda activate pair-distill
# Install python dependencies. We specify the versions in the requirements.txt file, but newer versions should work generally okay.
pip install -r requirements.txt
```

PairDistill supports two dense retrieval models: [ColBERT](https://github.com/stanford-futuredata/ColBERT) and [DPR through the dpr-scale library](https://github.com/facebookresearch/dpr-scale). Please install the corresponding dependencies for the model you want to use.

```bash
# Install ColBERT dependencies
pip install -r ColBERT/requirements.txt

# Install DPR dependencies
pip install -r dpr-scale/requirements.txt
```

#### Download Checkpoints and Datasets
In order to train PairDistill, please download the following checkpoints and datasets:
* [Pretrained ColBERTv2 checkpoint](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz): unzip to `ColBERT/colbertv2.0`
* [Preprocessed training dataset](https://drive.google.com/drive/folders/1sSFajkdZLChfF7aCPID2pEa6NrTT9hpB?usp=drive_link): put the files into `data/msmarco/`


### Training
Please navigate to `ColBERT` for ColBERT training. You could directly run
```python
python3 train.py
```
to launch the training. Adjust the number of GPUs according to your setup.


## Inference with PairDistill
PairDistill is directly compatible with ColBERT. Most of the instructions for running inference can be found in the [original ColBERT repo](https://github.com/stanford-futuredata/ColBERT).

In this repo, we provide instructions on how to run inference on the MSMARCO dev set and perform evaluation.

### Pretrained Checkpoints
Please download our pretrained PairDistill checkpoint from [Google Drive](https://drive.google.com/drive/folders/1AmaNrDbQf4Got6pTDygIZOiD8er4yFze?usp=drive_link) or [Huggingface](https://huggingface.co/chaoweihuang/PairDistill-colbertv2). Put it in `ColBERT/PairDistill`

### Running Inference
Please navigate to `ColBERT` for inference. Run
```
python3 index.py
```
to run the indexing with the trained checkpoint. You might need to adjust the path to the trained model.

Then, run
```
python3 retrieve.py
```
to run retrieval. The retrieval results will be saved at `ColBERT/PairDistill/index/{DATETIME}/msmarco.nbits=2.ranking.tsv`.

### Evaluation
You can use the provided script for evaluation. Run
```
python3 -m utility.evaluate.msmarco_passages \
      --ranking ColBERT/PairDistill/index/{DATETIME}/msmarco.nbits=2.ranking.tsv \
      --qrels ../data/msmarco/qrels.dev.small.tsv
```
