from colbert.data import Queries
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer, Indexer, Searcher


def train():
#     with Run().context(RunConfig(nranks=6, root="experiments", experiment="official-neg63-point-mini-pair-colbertv2-pair3.0")):
#         # triples = '../data/rankgpt/marco-train-100k-triples-r50-k16.jsonl'
#         # triples = '../data/msmarco/official_train_colbert_neg63.jsonl'
#         triples = '../data/msmarco/official_neg63_point_mini_pair.jsonl'
#         # triples = "../data/msmarco/official_neg63_point_mini_pair_r2-4_onlypair.jsonl"
#         queries = '../data/msmarco/queries.train.tsv'
#         collection = '../data/msmarco/collection.tsv'

#         config = ColBERTConfig(
#             bsize=48,
#             lr=1e-05,
#             warmup=3_000,
#             maxsteps=30_000,
#             doc_maxlen=180,
#             dim=128,
#             nway=64,
#             accumsteps=2,
#             use_ib_negatives=True,
#             save_every=10_000,
#             use_pair_scores=True,
#             pair_loss_alpha=3.0,
#             only_pair_loss=False,
#         )
#         trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)

#         # trainer.train(checkpoint='bert-base-uncased')
#         trainer.train(checkpoint="colbertv2.0")
#         # trainer.train(checkpoint="experiments/official-neg63-point-mini-pair-r2-4-colbertv2-pair3.0-cont/none/2024-01-31_14.16.16/checkpoints/colbert-15000")

#         # Index
#         config = ColBERTConfig(
#             nbits=2,
#             root="experiments",
#         )
#         indexer = Indexer(
#             checkpoint=trainer.best_checkpoint_path(), config=config
#         )
#         indexer.index(name="msmarco.nbits=2", collection="../data/msmarco/collection.tsv")

#         # Search
#         config = ColBERTConfig(
#             root="experiments",
#         )
#         searcher = Searcher(index="msmarco.nbits=2", config=config)
#         queries = Queries("../data/msmarco/queries.dev.small.tsv")
#         ranking = searcher.search_all(queries, k=100)
#         ranking.save("msmarco.nbits=2.ranking.tsv")

    with Run().context(RunConfig(nranks=4, root="experiments", experiment="colbertv2-fiqa-instupr-point-temp5.0")):
        triples = "../data/beir/fiqa/train_instupr_point.jsonl"
        queries = '../data/beir/fiqa/queries.train.tsv'
        collection = '../data/beir/fiqa/collection.tsv'

        config = ColBERTConfig(
            bsize=16,
            lr=1e-05,
            warmup=400,
            maxsteps=4_000,
            doc_maxlen=300,
            query_maxlen=32,
            dim=128,
            nway=64,
            accumsteps=2,
            use_ib_negatives=True,
            save_every=1_000,
            use_pair_scores=False,
            pair_loss_alpha=0.0,
            only_pair_loss=False,
            temperature=5.0
        )
        trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)

        # trainer.train(checkpoint='bert-base-uncased')
        trainer.train(checkpoint="colbertv2.0")
        # trainer.train(checkpoint="experiments/official-neg63-point-mini-pair-r2-4-colbertv2-pair3.0-cont/none/2024-01-31_14.16.16/checkpoints/colbert-15000")

        # Index
        config = ColBERTConfig(
            nbits=2,
            root="experiments",
            # kmeans_niters=5
        )
        indexer = Indexer(
            checkpoint=trainer.best_checkpoint_path(), config=config
        )
        indexer.index(name="fiqa.nbits=2", collection="../data/beir/fiqa/collection.tsv")

        # Search
        config = ColBERTConfig(
            root="experiments",
        )
        searcher = Searcher(index="fiqa.nbits=2", config=config)
        queries = Queries("../data/beir/fiqa/queries.test.tsv")
        ranking = searcher.search_all(queries, k=100)
        ranking.save("fiqa.nbits=2.ranking.tsv")


if __name__ == '__main__':
    train()
