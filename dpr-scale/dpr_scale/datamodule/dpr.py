#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import mmap
from typing import Any, Dict, List

import torch
import torch.nn as nn
from dpr_scale.transforms.dpr_distill_transform import DPRDistillTransform
from dpr_scale.transforms.dpr_transform import (
    DPRCrossAttentionTransform,
    DPRTransform,
    DPRPairDistillTransform
)

from dpr_scale.transforms.hf_transform import HFTransform
from dpr_scale.utils.utils import (
    ContiguousDistributedSampler,
    ContiguousDistributedSamplerForTest,
    maybe_add_title,
    PathManager,
)
from pytorch_lightning import LightningDataModule
import random


class MemoryMappedDataset(torch.utils.data.Dataset):
    """
    A memory mapped dataset.
    """

    def __init__(self, path, header=False):
        local_path = PathManager.get_local_path(path)
        self.file = open(local_path, mode="r")
        self.mm = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)
        if header:
            line = self.mm.readline()
        self.offset_dict = {0: self.mm.tell()}
        line = self.mm.readline()
        self.count = 0
        while line:
            self.count += 1
            offset = self.mm.tell()
            self.offset_dict[self.count] = offset
            line = self.mm.readline()

    def __len__(self):
        return self.count

    def process_line(self, line):
        return line

    def __getitem__(self, index):
        offset = self.offset_dict[index]
        self.mm.seek(offset)
        line = self.mm.readline()
        return self.process_line(line)


class MultiSourceDataset(torch.utils.data.Dataset):
    """
    A method to maintain a list of memory mapped datasets. Given a index, 
    it will randomly sample a memory mapped dataset and output the data point
    with the data index from the sampled memory mapped dataset. Note that we 
    now assume the data size the minimun size among the datasets; thus, the data 
    point with a higher index in the larger dataset will not be sampled.
    """

    def __init__(self, paths, header=False):
        self.datasets = []
        self.data_size = None
        self.datasets = [MemoryMappedDataset(path, header) for path in paths]
        self.data_size = min(len(dataset) for dataset in self.datasets)
        assert self.data_size > 0, "One of the path in datamodule.train_path is empty"

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        dataset = random.choice(self.datasets)
        return dataset[index]


class CSVDataset(MemoryMappedDataset):
    """
    A memory mapped dataset for csv files
    """

    def __init__(self, path, sep="\t"):
        super().__init__(path, header=True)
        self.sep = sep
        self.columns = self._get_header()

    def _get_header(self):
        self.mm.seek(0)
        return self._parse_line(self.mm.readline())

    def _parse_line(self, line):
        """Implementation of csv quoting."""
        row = line.decode().rstrip("\r\n").split(self.sep)
        for i, val in enumerate(row):
            if val and val[0] == '"' and val[-1] == '"':
                row[i] = val.strip('"').replace('""', '"')
        return row

    def process_line(self, line):
        vals = self._parse_line(line)
        if len(self.columns) == len(vals):
            return dict(zip(self.columns, vals))
        else:  # hack
            self.__getitem__(0)


class QueryCSVDataset(MemoryMappedDataset):
    """
    A memory mapped dataset for query csv files (such as the test set)
    """

    def __init__(self, path, sep="\t"):
        super().__init__(path, header=False)
        self.sep = sep

    def _parse_line(self, line):
        """Implementation of csv quoting."""
        row = line.decode().rstrip("\r\n").split(self.sep)
        for i, val in enumerate(row):
            if val and val[0] == '"' and val[-1] == '"':
                row[i] = val.strip('"').replace('""', '"')
        return row

    def process_line(self, line):
        vals = self._parse_line(line)
        return {
            "question": vals[0],
            # This unsafe eval call is needed because how the DPR data format
            # is designed: https://github.com/facebookresearch/DPR/blob/a31212dc0a54dfa85d8bfa01e1669f149ac832b7/dpr/data/retriever_data.py#L110
            "answers": eval(vals[1]),
        }


class QueryTSVDataset(MemoryMappedDataset):
    """
    A memory mapped dataset for query tsv files with the format qid\tquery_text\n
    """

    def __init__(self, path, sep="\t"):
        super().__init__(path, header=False)
        self.sep = sep

    def _parse_line(self, line):
        """Implementation of csv quoting."""
        row = line.decode().rstrip("\r\n").split(self.sep)
        for i, val in enumerate(row):
            if val and val[0] == '"' and val[-1] == '"':
                row[i] = val.strip('"').replace('""', '"')
        return row

    def process_line(self, line):
        vals = self._parse_line(line)
        return {
                "id": vals[0],
                "question": vals[1],
            }


class DenseRetrieverDataModuleBase(LightningDataModule):
    """
    Parent class for data modules.
    """

    def __init__(self, transform, *args, **kwargs):
        super().__init__()
        self.text_transform = transform

    def _transform(self, texts):
        if not isinstance(self.text_transform, HFTransform):
            result = self.text_transform({"text": texts})["token_ids"]
        else:
            result = self.text_transform(texts)
        return result

    def train_dataloader(self):
        sampler = None
        if (
            self.trainer
            and hasattr(self.trainer, "world_size")
            and self.trainer.world_size > 1
        ):
            sampler = ContiguousDistributedSampler(
                self.datasets["train"], num_replicas_per_node=self.trainer.gpus
            )

        return torch.utils.data.DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_train,
            sampler=sampler,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["valid"],
            shuffle=False,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_eval,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["test"],
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_test,
        )

    def collate_eval(self, batch):
        return self.collate(batch, "eval")

    def collate_test(self, batch):
        return self.collate(batch, "test")

    def collate_train(self, batch):
        return self.collate(batch, "train")


class DPRDistillJsonlDataModule(DenseRetrieverDataModuleBase):
    """
    This reads a jsonl file with json objects from the dpr distillation data
    """

    def __init__(
        self,
        transform,
        # Dataset args
        train_path: str,
        val_path: str,
        test_path: str,
        batch_size: int = 2,
        val_batch_size: int = 0,
        test_batch_size: int = 0,
        pos_ctx_sample: bool = True,  # defaults to use positive context sampling
        drop_last: bool = False,  # drop last batch if len(dataset) not multiple of bs
        num_workers: int = 0,  # increasing this bugs out right now
        *args,
        **kwargs,
    ) -> None:
        super().__init__(transform)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size else batch_size
        self.test_batch_size = (
            test_batch_size if test_batch_size else self.val_batch_size
        )
        transform_class = DPRDistillTransform
        self.drboost_distill_transform = transform_class(
            transform,
            pos_ctx_sample,
            **kwargs,
        )
        self.num_workers = num_workers
        self.datasets = {
            "train": MemoryMappedDataset(train_path),
            "valid": MemoryMappedDataset(val_path),
            "test": MemoryMappedDataset(test_path),
        }

    def collate(self, batch: Dict[str, Any], stage: str) -> nn.Module:
        return self.drboost_distill_transform(batch, stage)


class DenseRetrieverJsonlDataModule(DenseRetrieverDataModuleBase):
    """
    This reads a jsonl file with json objects from the original DPR data obtained from
    https://github.com/facebookresearch/DPR/blob/master/data/download_data.py.
    """

    def __init__(
        self,
        transform,
        # Dataset args
        train_path: str,
        val_path: str,
        test_path: str,
        batch_size: int = 2,
        val_batch_size: int = 0,  # defaults to batch_size
        test_batch_size: int = 0,  # defaults to val_batch_size
        num_positive: int = 1,  # currently, like the original paper only 1 is supported
        num_negative: int = 7,
        neg_ctx_sample: bool = True,
        pos_ctx_sample: bool = False,
        num_val_negative: int = 7,  # num negatives to use in validation
        num_test_negative: int = 0,  # defaults to num_val_negative
        drop_last: bool = False,  # drop last batch if len(dataset) not multiple of bs
        num_workers: int = 0,  # increasing this bugs out right now
        use_title: bool = False,  # use the title for context passages
        sep_token: str = " ",  # sep token between title and passage
        use_cross_attention: bool = False,  # Use cross attention model
        rel_sample: bool = False,  # Use relevance scores to sample ctxs
        *args,
        **kwargs,
    ):
        super().__init__(transform)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size else batch_size
        self.test_batch_size = (
            test_batch_size if test_batch_size else self.val_batch_size
        )
        transform_class = DPRTransform
        if use_cross_attention:
            transform_class = DPRCrossAttentionTransform
        self.dpr_transform = transform_class(
            transform,
            num_positive,
            num_negative,
            neg_ctx_sample,
            pos_ctx_sample,
            num_val_negative,
            num_test_negative,
            use_title,
            sep_token,
            rel_sample,
            **kwargs,
        )
        self.num_workers = num_workers
        self.datasets = {
            "train": MemoryMappedDataset(train_path),
            "valid": MemoryMappedDataset(val_path),
            "test": MemoryMappedDataset(test_path),
        }

    def collate(self, batch, stage):
        return self.dpr_transform(batch, stage)


class DenseRetrieverMultiJsonlDataModule(DenseRetrieverDataModuleBase):
    """
    There are two main changes in the datamodule compared to DenseRetrieverJsonlDataModule:
    (1) This reads multiple train jsonl files with json objects from the original DPR. 
    (2) DPRTransform directly fetches text from given corpus through 'docidx'
        
    If you input corpus_path, you can only save a light train jsonl file with docidx 
    (the doc position in the corpus and start from 0) for postive_ctxs and negative_ctxs 
    instead of the whole passage title and text.
    
    The example of train json file:
    {"query_id": "3", "question": " Another name for the primary visual 
    cortex is", "positive_ctxs": [{"docidx": 1142685, "score": "1818.663208"}], 
    "hard_negative_ctxs": [{"docidx": 6711744]}
    """

    def __init__(
        self,
        transform,
        # Dataset args
        train_path: List[str],
        val_path: str,
        test_path: str,
        batch_size: int = 2,
        val_batch_size: int = 0,  # defaults to batch_size
        test_batch_size: int = 0,  # defaults to val_batch_size
        num_positive: int = 1,  # currently, like the original paper only 1 is supported
        num_negative: int = 7,
        neg_ctx_sample: bool = True,
        pos_ctx_sample: bool = False,
        num_val_negative: int = 7,  # num negatives to use in validation
        num_test_negative: int = 0,  # defaults to num_val_negative
        drop_last: bool = False,  # drop last batch if len(dataset) not multiple of bs
        num_workers: int = 0,  # increasing this bugs out right now
        use_title: bool = False,  # use the title for context passages
        sep_token: str = " ",  # sep token between title and passage
        use_cross_attention: bool = False, # Use cross attention model
        rel_sample: bool = False,  # Use relevance scores to sample ctxs
        corpus_path: str = None, # if corpus is None, we assume the training data includes passage text
        *args,
        **kwargs,
    ):
        
        super().__init__(transform)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size else batch_size
        self.test_batch_size = (
            test_batch_size if test_batch_size else self.val_batch_size
        )
        transform_class = DPRTransform
        if use_cross_attention:
            transform_class = DPRCrossAttentionTransform
        self.num_workers = num_workers

        corpus = None
        if corpus_path is not None:
            corpus = MemoryMappedDataset(corpus_path, header=True)
        
        self.dpr_transform = transform_class(
            transform,
            num_positive,
            num_negative,
            neg_ctx_sample,
            pos_ctx_sample,
            num_val_negative,
            num_test_negative,
            use_title,
            sep_token,
            rel_sample,
            corpus,
            **kwargs,
        )
        self.datasets = {
            "train": MultiSourceDataset(train_path),
            "valid": MemoryMappedDataset(val_path),
            "test": MemoryMappedDataset(test_path),
        }

    def collate(self, batch, stage):
        return self.dpr_transform(batch, stage)
    

class DenseRetrieverMultiJsonlPairDistillDataModule(DenseRetrieverDataModuleBase):
    def __init__(
        self,
        transform,
        # Dataset args
        train_path: List[str],
        val_path: str,
        test_path: str,
        queries_path: str,
        batch_size: int = 2,
        num_passages: int = 64,
        num_workers: int = 0,  # increasing this bugs out right now
        val_batch_size: int = 0,  # defaults to batch_size
        test_batch_size: int = 0,  # defaults to val_batch_size
        use_title: bool = False,  # use the title for context passages
        corpus_path: str = None, # if corpus is None, we assume the training data includes passage text
        *args,
        **kwargs,
    ):
        
        super().__init__(transform)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size else batch_size
        self.test_batch_size = (
            test_batch_size if test_batch_size else self.val_batch_size
        )
        self.num_workers = num_workers

        corpus = None
        if corpus_path is not None:
            corpus = MemoryMappedDataset(corpus_path, header=True)
        
        self.dpr_transform = DPRPairDistillTransform(
            transform,
            corpus,
            queries_path,
            use_title,
            num_passages
        )
        self.datasets = {
            "train": MultiSourceDataset(train_path),
            "valid": MemoryMappedDataset(val_path),
            "test": MemoryMappedDataset(test_path),
        }

    def collate(self, batch, stage):
        return self.dpr_transform(batch, stage)


class DenseRetrieverPassagesDataModule(DenseRetrieverDataModuleBase):
    """
    This reads a csv file of passages for embedding creation.
    """

    def __init__(
        self,
        transform,
        test_path: str,
        test_batch_size: int = 128,  # defaults to val_batch_size
        num_workers: int = 0,  # increasing this bugs out right now
        use_title: bool = False,  # use the title for context passages
        sep_token: str = " [SEP] ",  # sep token between title and passage
        *args,
        **kwargs,
    ):
        super().__init__(transform)
        self.test_path = test_path
        self.test_batch_size = test_batch_size
        self.use_title = use_title
        self.sep_token = sep_token
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.datasets = {
            "test": CSVDataset(self.test_path),
        }

    def collate(self, batch, stage):
        ctx_tensors = self._transform(
            [
                maybe_add_title(
                    row["text"], row["title"], self.use_title, self.sep_token
                )
                for row in batch
            ]
        )
        if "id" in batch[0]:
            return {
                "contexts_ids": ctx_tensors,
                "corpus_ids": [row["id"] for row in batch],
            }
        return {"contexts_ids": ctx_tensors}

    def val_dataloader(self):
        return self.test_dataloader()

    def train_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        sampler = None
        if (
            self.trainer
            and hasattr(self.trainer, "world_size")
            and self.trainer.world_size > 1
        ):
            sampler = ContiguousDistributedSamplerForTest(self.datasets["test"])

        return torch.utils.data.DataLoader(
            self.datasets["test"],
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_test,
            sampler=sampler,
        )


class DenseRetrieverQueriesDataModule(DenseRetrieverDataModuleBase):
    """
    This reads a csv file of questions for query embedding creation.
    """

    def __init__(
        self,
        transform,
        test_path: str,
        test_batch_size: int = 128,  # defaults to val_batch_size
        num_workers: int = 0,  # increasing this bugs out right now
        trec_format: bool = False, # the format is used for msmarco query
        *args,
        **kwargs,
    ):
        super().__init__(transform)
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.datasets = {"test": QueryTSVDataset(test_path) if trec_format else QueryCSVDataset(test_path) }

    def collate(self, batch, stage):
        ctx_tensors = self._transform([row["question"] for row in batch])
        return {"query_ids": ctx_tensors}

    def val_dataloader(self):
        return self.test_dataloader()

    def train_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        sampler = None
        if (
            self.trainer
            and hasattr(self.trainer, "world_size")
            and self.trainer.world_size > 1
        ):
            sampler = ContiguousDistributedSamplerForTest(self.datasets["test"])

        return torch.utils.data.DataLoader(
            self.datasets["test"],
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_test,
            sampler=sampler,
        )