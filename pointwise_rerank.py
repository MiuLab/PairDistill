import argparse

import torch

from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSequenceClassification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pointwise reranking with monoT5')
    parser.add_argument('--runfile', type=str, help='path to the run file', required=True)
    parser.add_argument('--output', type=str, help='path to the output file', required=True)
    parser.add_argument('--collection', type=str, help='path to collection.tsv', required=True)
    parser.add_argument('--queries', type=str, help='path to queries.tsv', required=True)
    parser.add_argument('--model', type=str, default="castorini/monot5-3b-msmarco")
    parser.add_argument('--generation_model', action='store_true', help='whether to use generation model')
    parser.add_argument('--n_passages', type=int, default=100, help='number of passages to rerank')
    parser.add_argument('--prev_rerank_file', type=str, default=None, help='path to the previous rerank file')
    args = parser.parse_args()

    # Read the collection
    print("Reading the collection...")
    collection = {}
    with open(args.collection, 'r') as f:
        for line in f:
            docid, text = line.strip().split('\t')
            collection[docid] = text

    # Read the queries
    print("Reading the queries...")
    queries = {}
    with open(args.queries, 'r') as f:
        for line in f:
            qid, text = line.strip().split('\t')
            queries[qid] = text

    # Read the run file
    print("Reading the run file...")
    run = {}
    with open(args.runfile, 'r') as f:
        for line in f:
            qid, docid, rank, score = line.strip().split('\t')
            if qid not in run:
                run[qid] = []
            run[qid].append(docid)

    # Read the previous rerank file
    prev_rerank = {}
    if args.prev_rerank_file:
        print("Reading the previous rerank file...")
        with open(args.prev_rerank_file, 'r') as f:
            for line in f:
                qid, docid, score_true, score_false = line.strip().split('\t')
                if qid not in prev_rerank:
                    prev_rerank[qid] = {}
                prev_rerank[qid][docid] = [float(score_true), float(score_false)]

    # Load the model
    print("Loading the model...")
    if args.generation_model:
        model = T5ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16)
        tokenizer = T5Tokenizer.from_pretrained(args.model, legacy=True, use_fast=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    model.eval()
    model = model.cuda()

    all_passages = []
    all_scores = []
    for qid in tqdm(run):
        # Only rerank top n_passages passages
        passages_scored, passages = [], []
        scores = []
        for docid in run[qid][:args.n_passages]:
            if docid in prev_rerank.get(qid, {}):
                scores.append(prev_rerank[qid][docid])
                passages_scored.append(docid)
            else:
                passages.append(docid)

        # Rerank the passages
        if len(passages) > 0:
            if args.generation_model:
                inputs = []
                for docid in passages:
                    inputs.append(f"Query: {queries[qid]} Document: {collection[docid]} Relevant:")
                inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=180)

                # Generate the outputs with scores
                outputs = model.generate(inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda(), return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
                scores.extend(outputs.scores[0][:, [1176, 6136]].cpu().tolist())
            else:
                input_queries = [queries[qid]] * len(passages)
                input_passages = [collection[docid] for docid in passages]
                inputs = tokenizer(input_queries, input_passages, return_tensors="pt", padding='longest', truncation=True, max_length=180).to('cuda')

                with torch.inference_mode():
                    outputs = model(**inputs).logits.flatten().cpu().tolist()
                    scores.extend([[score, 0.0] for score in outputs])
        
        passages = passages_scored + passages
        passages = [(qid, docid) for docid in passages]
        all_passages.extend(passages)
        all_scores.extend(scores)

    # Write the output
    with open(args.output, 'w') as f:
        for (qid, docid), (score_true, score_false) in zip(all_passages, all_scores):
            f.write(f"{qid}\t{docid}\t{round(score_true, 6)}\t{round(score_false, 6)}\n")