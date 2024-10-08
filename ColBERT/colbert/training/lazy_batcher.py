import os
import ujson

from functools import partial
from colbert.infra.config.config import ColBERTConfig
from colbert.utils.utils import print_message, zipstar
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from colbert.evaluation.loaders import load_collection

from colbert.data.collection import Collection
from colbert.data.queries import Queries
from colbert.data.examples import Examples

# from colbert.utils.runs import Run


class LazyBatcher():
    def __init__(self, config: ColBERTConfig, triples, queries, collection, rank=0, nranks=1):
        self.bsize, self.accumsteps = config.bsize, config.accumsteps
        self.nway = config.nway

        self.query_tokenizer = QueryTokenizer(config)
        self.doc_tokenizer = DocTokenizer(config)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        self.triples = Examples.cast(triples, nway=self.nway).tolist(rank, nranks)
        self.queries = Queries.cast(queries)
        self.collection = Collection.cast(collection)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __next__(self):
        offset, endpos = self.position, min(self.position + self.bsize, len(self.triples))
        self.position = endpos

        if offset + self.bsize > len(self.triples):
            raise StopIteration

        all_queries, all_passages, all_scores, all_pair_scores = [], [], [], []

        for position in range(offset, endpos):
            query, *pids = self.triples[position]
            pids = pids[:self.nway]

            query = self.queries[query]

            if isinstance(pids[0], int):
                scores, pair_scores = [], []
            elif len(pids[0]) == 2:
                pids, scores = zipstar(pids)
                pair_scores = []
            elif len(pids[0]) == 3:
                pids, scores, pair_scores = zipstar(pids)

            passages = [self.collection[pid] for pid in pids]

            all_queries.append(query)
            all_passages.extend(passages)
            all_scores.extend(scores)
            all_pair_scores.append(pair_scores)
        
        assert len(all_scores) in [0, len(all_passages)], len(all_scores)

        return self.collate(all_queries, all_passages, all_scores, all_pair_scores)

    def collate(self, queries, passages, scores, pair_scores):
        assert len(queries) == self.bsize
        assert len(passages) == self.nway * self.bsize

        return self.tensorize_triples(queries, passages, scores, self.bsize // self.accumsteps, self.nway, pair_scores)

