PairDistill: Pairwise Relevance Distillation for Dense Retrieval
===

<p align="center">
📃 <a href="https://arxiv.org/abs/2410.01383" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/collections/chaoweihuang/pairdistill-66fe6b0cfa6eae4704df9f5e" target="_blank">Models & Datasets</a>
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

## Training
Please navigate to `ColBERT` for ColBERT training. You could directly run
```python
python3 train.py
```
to launch the training.
