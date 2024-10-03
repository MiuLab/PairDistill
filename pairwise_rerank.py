import argparse
import random
import math
import asyncio

import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from huggingface_hub import AsyncInferenceClient


"""
This script takes a run file and run pairwise reranking on n pairs sampled from all pairs using duoT5.
"""


def generate(client: AsyncInferenceClient, prompt: str):
    return client.text_generation(
        prompt=prompt, max_new_tokens=1, do_sample=False, details=True
    )


async def tgi_inference(client: AsyncInferenceClient, inputs: list):
    tasks = []
    for i in range(0, len(inputs)):
        tasks.append(generate(client, inputs[i]))

    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pairwise reranking with duoT5')
    parser.add_argument('--runfile', type=str, help='path to the run file', required=True)
    parser.add_argument('--output', type=str, help='path to the output file', required=True)
    parser.add_argument('--collection', type=str, help='path to collection.tsv', required=True)
    parser.add_argument('--queries', type=str, help='path to queries.tsv', required=True)
    parser.add_argument('--model', type=str, default="castorini/duot5-3b-msmarco")
    parser.add_argument('--n_pairs', type=int, default=100, help='number of pairs to rerank')
    parser.add_argument('--topk', type=int, default=100, help='number of topk passages for pairwise reranking')
    parser.add_argument('--from_point', action='store_true', help='whether to load from pointwise reranking')
    parser.add_argument('--softmax', action='store_true', help='whether to use softmax')
    parser.add_argument('--tgi_server', type=str, default=None, help='Set this to the URL to TGI server if you want to use TGI for inference')
    parser.add_argument('--tgi_batch_size', type=int, default=1000, help='Batch size for TGI inference')
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

    if args.tgi_server is None:
        # Load the model
        print("Loading the model...")
        model = T5ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16)
        tokenizer = T5Tokenizer.from_pretrained(args.model, legacy=True, use_fast=True)
        model.eval()
        model = model.cuda()
    else:
        print("Loading the TGI client...")
        client = AsyncInferenceClient(model=args.tgi_server)
        current_inputs = []
        current_pairs = []

    all_pairs = []
    all_scores = []
    for q_index, qid in enumerate(tqdm(run)):
        run[qid] = sorted(run[qid], key=lambda x: x[1], reverse=True)[:args.topk]
        run[qid] = [docid for docid, _ in run[qid]]

        pairs = []
        for i in range(len(run[qid])):
            # for j in range(i + 1, len(run[qid])):
            for j in range(i + 1, min(i + 6, len(run[qid]))):
                pairs.append((qid, run[qid][i], run[qid][j]))
        
        # Sample n_pairs pairs
        sampled_pairs = random.sample(pairs, args.n_pairs)
        # sampled_pairs = pairs

        # Rerank the pairs
        inputs = []
        for qid, docid1, docid2 in sampled_pairs:
            inputs.append(f"Query: {queries[qid]} Document0: {collection[docid1]} Document1: {collection[docid2]} Relevant:")

        if args.tgi_server is None:
            # batch inference
            # batch_size = args.n_pairs
            for i in range(0, len(inputs), args.n_pairs):
                batch_inputs = inputs[i : i + args.n_pairs]
                batch_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=360)
                outputs = model.generate(batch_inputs['input_ids'].cuda(), attention_mask=batch_inputs['attention_mask'].cuda(), return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
                all_pairs.extend(sampled_pairs[i : i + args.n_pairs])
                all_scores.extend(outputs.scores[0][:, [1176, 6136]].cpu().tolist())

            # inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=360)

            # # Generate the outputs with scores
            # outputs = model.generate(inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda(), return_dict_in_generate=True, output_scores=True, max_new_tokens=1)

            # all_pairs.extend(sampled_pairs)
            # all_scores.extend(outputs.scores[0][:, [1176, 6136]].cpu().tolist())
        else:
            current_inputs.extend(inputs)
            current_pairs.extend(sampled_pairs)
            if len(current_inputs) >= args.tgi_batch_size or q_index == len(run) - 1:
                results = asyncio.run(tgi_inference(client, current_inputs))
                all_pairs.extend(current_pairs)
                for r in results:
                    prob = math.exp(r.details.tokens[0].logprob)
                    prob = prob if r.details.tokens[0].id == 1176 else 1 - prob
                    all_scores.append([math.log(min(prob + 1e-8, 1.0)), math.log(min(1 - prob + 1e-8, 1.0))])
                current_inputs = []
                current_pairs = []

    # Write the output
    with open(args.output, 'w') as f:
        for (qid, docid1, docid2), (score_true, score_false) in zip(all_pairs, all_scores):
            f.write(f"{qid}\t{docid1}\t{docid2}\t{round(score_true, 6)}\t{round(score_false, 6)}\n")