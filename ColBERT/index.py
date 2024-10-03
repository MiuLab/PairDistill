from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher


if __name__ == '__main__':
    # with Run().context(RunConfig(nranks=8, experiment="official-neg63-point-mini-pair-r2-4-colbertv2-pair3.0-cont2-2")):
    #     # for dataset in ["arguana", "climate-fever", "fever", "fiqa", "hotpotqa", "nfcorpus", "nq", "quora", "robust04", "scidocs", "scifact", "signal1m", "trec-covid", "trec-news", "webis-touche2020"]:
    #     # for dataset in ["bioasq", "cqadupstack/android", "cqadupstack/english", "cqadupstack/gaming", "cqadupstack/gis", "cqadupstack/mathematica", "cqadupstack/physics", "cqadupstack/programmers", "cqadupstack/stats", "cqadupstack/tex", "cqadupstack/unix", "cqadupstack/webmasters", "cqadupstack/wordpress"]:
    #     for dataset in ["dbpedia-entity", "wiki"]:
    #     # for dataset in ["arguana", "fiqa", "nfcorpus", "quora", "robust04", "scidocs", "scifact", "trec-covid", "trec-news", "webis-touche2020"]:
    #     # for dataset in ["climate-fever", "fever", "hotpotqa", "nq", "signal1m"]:
    #         dataset_name = dataset.replace('/', '-')
    #         config = ColBERTConfig(
    #             nbits=2,
    #             root="experiments",
    #             query_maxlen={
    #                 "arguana": 300,
    #                 "climate-fever": 64,
    #             }.get(dataset, 32),
    #             doc_maxlen=300,
    #             kmeans_niters=4 if dataset == "wiki" else 20
    #         )
    #         indexer = Indexer(
    #             checkpoint=f"experiments/official-neg63-point-mini-pair-r2-4-colbertv2-pair3.0-cont2-2/none/2024-02-02_00.40.15/checkpoints/colbert-30000",
    #             config=config
    #         )
    #         if dataset == "wiki":
    #             indexer.index(name=f"{dataset_name}.len.nbits=2", collection=f"../data/{dataset}/collection.tsv")
    #         else:
    #             indexer.index(name=f"{dataset_name}.len.nbits=2", collection=f"../data/beir/{dataset}/collection.tsv")

    #         # Search
    #         if dataset != "wiki":
    #             config = ColBERTConfig(
    #                 root="experiments",
    #             )
    #             searcher = Searcher(index=f"{dataset_name}.len.nbits=2", config=config)
    #             queries = Queries(f"../data/beir/{dataset}/queries.test.tsv")
    #             ranking = searcher.search_all(queries, k=100)
    #             ranking.save(f"{dataset_name}.nbits=2.ranking.tsv")

    # with Run().context(RunConfig(nranks=4, experiment="colbertv2")):
    #     # for dataset in ["arguana", "climate-fever", "fever", "fiqa", "hotpotqa", "nfcorpus", "nq", "quora", "robust04", "scidocs", "scifact", "signal1m", "trec-covid", "trec-news", "webis-touche2020"]:
    #     for dataset in ["bioasq", "cqadupstack/android", "cqadupstack/english", "cqadupstack/gaming", "cqadupstack/gis", "cqadupstack/mathematica", "cqadupstack/physics", "cqadupstack/programmers", "cqadupstack/stats", "cqadupstack/tex", "cqadupstack/unix", "cqadupstack/webmasters", "cqadupstack/wordpress"]:
    #     # for dataset in ["arguana", "fiqa", "nfcorpus", "quora", "robust04", "scidocs", "scifact", "trec-covid", "trec-news", "webis-touche2020"]:
    #     # for dataset in ["climate-fever", "fever", "hotpotqa", "nq", "signal1m"]:
    #         dataset_name = dataset.replace('/', '-')
    #         config = ColBERTConfig(
    #             nbits=2,
    #             root="experiments",
    #             query_maxlen={
    #                 "arguana": 300,
    #                 "climate-fever": 64,
    #             }.get(dataset, 32),
    #             doc_maxlen=300,
    #             kmeans_niters=5 if dataset == "bioasq" else 20
    #         )
    #         indexer = Indexer(
    #             checkpoint="colbertv2.0",
    #             config=config
    #         )
    #         indexer.index(name=f"{dataset_name}.len.nbits=2", collection=f"../data/beir/{dataset}/collection.tsv")

    #         # Search
    #         config = ColBERTConfig(
    #             root="experiments",
    #         )
    #         searcher = Searcher(index=f"{dataset_name}.len.nbits=2", config=config)
    #         queries = Queries(f"../data/beir/{dataset}/queries.test.tsv")
    #         ranking = searcher.search_all(queries, k=100)
    #         ranking.save(f"{dataset_name}.nbits=2.ranking.tsv")

    # with Run().context(RunConfig(nranks=4, experiment="official-neg63-point-mini-pair-r2-4-colbertv2-pair3.0-cont2-2")):
    #     for dataset in ["writing", "recreation", "science", "technology", "lifestyle", "pooled"]:
    #         for split in ["dev", "test"]:
    #             config = ColBERTConfig(
    #                 nbits=2,
    #                 root="experiments",
    #                 query_maxlen=32,
    #                 doc_maxlen=300
    #             )
    #             indexer = Indexer(
    #                 checkpoint=f"experiments/official-neg63-point-mini-pair-r2-4-colbertv2-pair3.0-cont2-2/none/2024-02-02_00.40.15/checkpoints/colbert-30000",
    #                 config=config
    #             )
    #             indexer.index(name=f"{dataset}.{split}.nbits=2", collection=f"../data/lotte/{dataset}/{split}/collection.tsv")

    #             # Search
    #             for query_type in ["forum", "search"]:
    #                 config = ColBERTConfig(
    #                     root="experiments",
    #                 )
    #                 searcher = Searcher(index=f"{dataset}.{split}.nbits=2", config=config)
    #                 queries = Queries(f"../data/lotte/{dataset}/{split}/questions.{query_type}.tsv")
    #                 ranking = searcher.search_all(queries, k=100)
    #                 ranking.save(f"{dataset}.{split}.{query_type}.nbits=2.ranking.tsv")

    # with Run().context(RunConfig(nranks=4, experiment="colbertv2")):
    #     for dataset in ["writing", "recreation", "science", "technology", "lifestyle", "pooled"]:
    #         for split in ["dev", "test"]:
    #             config = ColBERTConfig(
    #                 nbits=2,
    #                 root="experiments",
    #                 query_maxlen=32,
    #                 doc_maxlen=300
    #             )
    #             indexer = Indexer(
    #                 checkpoint="colbertv2.0",
    #                 config=config
    #             )
    #             indexer.index(name=f"{dataset}.{split}.nbits=2", collection=f"../data/lotte/{dataset}/{split}/collection.tsv")

    #             # Search
    #             for query_type in ["forum", "search"]:
    #                 config = ColBERTConfig(
    #                     root="experiments",
    #                 )
    #                 searcher = Searcher(index=f"{dataset}.{split}.nbits=2", config=config)
    #                 queries = Queries(f"../data/lotte/{dataset}/{split}/questions.{query_type}.tsv")
    #                 ranking = searcher.search_all(queries, k=100)
    #                 ranking.save(f"{dataset}.{split}.{query_type}.nbits=2.ranking.tsv")

    # with Run().context(RunConfig(nranks=2, experiment="colbertv2-climate-fever-instupr-point")):
    #     for dataset in ["climate-fever"]:
    #         config = ColBERTConfig(
    #             nbits=2,
    #             root="experiments",
    #             query_maxlen={
    #                 "arguana": 300,
    #                 "climate-fever": 64,
    #             }.get(dataset, 32),
    #             doc_maxlen=300,
    #             kmeans_niters=5 if dataset == "bioasq" else 20
    #         )
    #         indexer = Indexer(
    #             checkpoint="experiments/colbertv2-climate-fever-instupr-point/none/2024-02-11_01.34.04/checkpoints/colbert-1000/",
    #             config=config
    #         )
    #         indexer.index(name=f"{dataset}.len.nbits=2", collection=f"../data/beir/{dataset}/collection.tsv")
            
    #         config = ColBERTConfig(
    #             root="experiments",
    #             query_maxlen=32,
    #             doc_maxlen=300
    #         )
    #         searcher = Searcher(index=f"{dataset}.len.nbits=2", config=config)
    #         queries = Queries(f"../data/beir/{dataset}/queries.test.tsv")
    #         ranking = searcher.search_all(queries, k=100)
    #         ranking.save(f"{dataset}.nbits=2.ranking.tsv")

    # with Run().context(RunConfig(nranks=8, experiment="official-neg63-point-mini-pair-r2-4-colbertv2-pair3.0-cont2-2")):
    with Run().context(RunConfig(nranks=4, experiment="colbertv2")):
        # Index
        config = ColBERTConfig(
            nbits=2,
            root="experiments",
            kmeans_niters=4
        )
        indexer = Indexer(
            checkpoint="colbertv2.0", config=config
        )
        indexer.index(name="wiki2021_infobox.nbits=2", collection="../data/wiki/atlas/collection_infobox.tsv")

        # config = ColBERTConfig(
        #     root="experiments",
        #     query_maxlen=32
        # )
        # searcher = Searcher(index=f"msmarco.nbits=2", config=config)
        # # searcher = Searcher(index=f"wiki.len.nbits=2", config=config)
        # queries = Queries(f"../data/msmarco/queries.dev.small.tsv")
        # ranking = searcher.search_all(queries, k=1000)
        # ranking.save(f"msmarco.nbits=2.ranking.tsv")
