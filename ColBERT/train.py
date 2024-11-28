from colbert.data import Queries
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer, Indexer, Searcher


def train():
    with Run().context(RunConfig(nranks=4, root="experiments", experiment="PairDistill")):
        triples = "../data/msmarco/pairs.train.jsonl"
        queries = '../data/msmarco/queries.train.tsv'
        collection = '../data/msmarco/collection.tsv'

        config = ColBERTConfig(
            bsize=32,
            lr=1e-05,
            warmup=3_000,
            maxsteps=30_000,
            doc_maxlen=180,
            dim=128,
            nway=64,
            accumsteps=2,
            use_ib_negatives=True,
            save_every=10_000,
            use_pair_scores=True,
            pair_loss_alpha=3.0,
            only_pair_loss=False,
        )
        trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)

        trainer.train(checkpoint="colbertv2.0")

        # Index
        config = ColBERTConfig(
            nbits=2,
            root="experiments",
        )
        indexer = Indexer(
            checkpoint=trainer.best_checkpoint_path(), config=config
        )
        indexer.index(name="msmarco.nbits=2", collection="../data/msmarco/collection.tsv")

        # Search
        config = ColBERTConfig(
            root="experiments",
        )
        searcher = Searcher(index="msmarco.nbits=2", config=config)
        queries = Queries("../data/msmarco/queries.dev.small.tsv")
        ranking = searcher.search_all(queries, k=100)
        ranking.save("msmarco.nbits=2.ranking.tsv")


if __name__ == '__main__':
    train()
