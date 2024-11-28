from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher


if __name__ == '__main__':
    with Run().context(RunConfig(nranks=4, experiment="PairDistill")):
        # Index
        config = ColBERTConfig(
            nbits=2,
            root="experiments",
            kmeans_niters=4
        )
        indexer = Indexer(
            checkpoint="PairDistill", config=config
        )
        indexer.index(name="msmarco.nbits=2", collection="../data/msmarco/collection.tsv")
