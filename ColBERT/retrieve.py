from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher


if __name__ == '__main__':
    with Run().context(RunConfig(nranks=4, experiment="PairDistill")):

        config = ColBERTConfig(
            root="experiments",
        )
        searcher = Searcher(index="msmarco.nbits=2", collection="../data/msmarco/collection.tsv", config=config)
        queries = Queries("../data/msmarco/queries.train.tsv")
        ranking = searcher.search_all(queries, k=100)
        ranking.save("msmarco.train.nbits=2.ranking.tsv")
