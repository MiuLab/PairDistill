import argparse
import random
import math

import torch
import torch.nn.functional as F

from tqdm import tqdm
from openai import OpenAI
from transformers import T5ForConditionalGeneration, T5Tokenizer


"""
This script takes a run file and run pairwise reranking on n pairs sampled from all pairs using InstUPR.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pairwise reranking with duoT5')
    parser.add_argument('--runfile', type=str, help='path to the run file', required=True)
    parser.add_argument('--output', type=str, help='path to the output file', required=True)
    parser.add_argument('--collection', type=str, help='path to collection.tsv', required=True)
    parser.add_argument('--queries', type=str, help='path to queries.tsv', required=True)
    parser.add_argument('--use_title', action='store_true', help='use the title of the passage as context')
    parser.add_argument('--model', type=str, default="google/flan-t5-xl")
    parser.add_argument('--openai', action='store_true', help='whether to use the openai api')
    parser.add_argument('--n_pairs', type=int, default=100, help='number of pairs to rerank')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for inference')
    parser.add_argument('--max_length', type=int, default=1024, help='max length for inference')
    parser.add_argument('--topk', type=int, default=100, help='number of topk passages for pairwise reranking')
    parser.add_argument('--from_point', action='store_true', help='whether to load from pointwise reranking')
    parser.add_argument('--softmax', action='store_true', help='whether to use softmax')
    args = parser.parse_args()

    # Read the collection
    print("Reading the collection...")
    collection = {}
    with open(args.collection, 'r') as f:
        for line in f:
            if args.use_title:
                docid, text, title = line.split('\t')
                collection[docid] = title + " " + text
            else:
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
        if args.from_point:
            for line in f:
                qid, docid, score_true, score_false = line.strip().split('\t')
                if qid not in run:
                    run[qid] = []
                if args.softmax:
                    log_prob = F.softmax(torch.tensor([float(score_true), float(score_false)]), dim=0)[0].log().item()
                else:
                    log_prob = float(score_true)
                run[qid].append((docid, log_prob))
        else:
            for line in f:
                qid, docid, rank, score = line.strip().split('\t')
                if qid not in run:
                    run[qid] = []
                run[qid].append((docid, float(score)))

    if args.openai:
        print("Using the OpenAI API...")
        client = OpenAI()
    else:
        # Load the model
        print("Loading the model...")
        model = T5ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16)
        tokenizer = T5Tokenizer.from_pretrained(args.model, legacy=True, use_fast=True)
        model.eval()
        model = model.cuda()

    # Filter the qids in run that are not in queries
    qids = [qid for qid in run if qid in queries]

    all_pairs = []
    all_scores = []
    for q_index, qid in enumerate(tqdm(qids)):
        run[qid] = sorted(run[qid], key=lambda x: x[1], reverse=True)[:args.topk]
        run[qid] = [docid for docid, _ in run[qid]]

        pairs = []
        for i in range(len(run[qid])):
            # for j in range(i + 1, len(run[qid])):
            for j in range(i + 1, min(i + 8, len(run[qid]))):
                pairs.append((qid, run[qid][i], run[qid][j]))
        
        # Sample n_pairs pairs
        sampled_pairs = random.sample(pairs, args.n_pairs)
        # sampled_pairs = pairs

        # Rerank the pairs
        inputs = []
        for qid, docid1, docid2 in sampled_pairs:
            inputs.append(f"Which document is more relevant to the query? Answer only 'A' or 'B'.\n\nQuery: {queries[qid]}\n\nDocument A: {collection[docid1]}\n\nDocument B: {collection[docid2]}")

        if args.openai:
            for prompt in inputs:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer the question directly with 'A' or 'B'."},
                        {"role": "user", "content": prompt}
                    ]
                )
                import ipdb; ipdb.set_trace()
        else:
            # batch inference
            # batch_size = args.batch_size
            for i in range(0, len(inputs), args.batch_size):
                batch_inputs = inputs[i : i + args.batch_size]
                batch_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
                outputs = model.generate(batch_inputs['input_ids'].cuda(), attention_mask=batch_inputs['attention_mask'].cuda(), return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
                all_pairs.extend(sampled_pairs[i : i + args.batch_size])
                all_scores.extend(outputs.scores[0][:, [71, 272]].cpu().tolist())

    # Write the output
    with open(args.output, 'w') as f:
        for (qid, docid1, docid2), (score_true, score_false) in zip(all_pairs, all_scores):
            f.write(f"{qid}\t{docid1}\t{docid2}\t{round(score_true, 6)}\t{round(score_false, 6)}\n")