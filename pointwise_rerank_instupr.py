import argparse
import math

import torch

from tqdm import tqdm
from openai import OpenAI
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSequenceClassification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pointwise reranking with InstUPR (flan-t5)')
    parser.add_argument('--runfile', type=str, help='path to the run file', required=True)
    parser.add_argument('--output', type=str, help='path to the output file', required=True)
    parser.add_argument('--collection', type=str, help='path to collection.tsv', required=True)
    parser.add_argument('--queries', type=str, help='path to queries.tsv', required=True)
    parser.add_argument('--model', type=str, default="google/flan-t5-xl")
    parser.add_argument('--openai', action='store_true', help='whether to use the openai api')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for inference')
    parser.add_argument('--max_length', type=int, default=1024, help='max length for inference')
    parser.add_argument('--use_title', action='store_true', help='use the title of the passage as context')
    parser.add_argument('--n_passages', type=int, default=100, help='number of passages to rerank')
    parser.add_argument('--prev_rerank_file', type=str, default=None, help='path to the previous rerank file')
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
                docid, text = line.split('\t')
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

    if args.openai:
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

    all_passages = []
    all_scores = []
    for qid in tqdm(qids):
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
            inputs = []
            for docid in passages:
                # inputs.append(f"Given a query and a context, answer whether the context is relevant to the query (Yes or No).\n\nQuery: {queries[qid]}\n\nContext: {collection[docid]}")
                inputs.append(f"Is the document relevant to the query (Yes or No)?\n\nQuery: {queries[qid]}\n\nDocument: {collection[docid]}")

            if args.openai:
                for prompt in inputs:
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            # {"role": "system", "content": "You are a helpful assistant. Answer the question directly with 'Yes' or 'No'."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1,
                        logprobs=True,
                        top_logprobs=4
                    )

                    logprob = -999.0
                    for top_logprob in response.choices[0].logprobs.content[0].top_logprobs:
                        if top_logprob.token == "Yes":
                            logprob = top_logprob.logprob
                            break
                    
                    prob_yes = torch.tensor(logprob).exp()
                    logit = (prob_yes.log() - (1 - prob_yes).log()).item()
                    scores.append([logit, 0.0])
            else:
                for i in range(0, len(inputs), args.batch_size):
                    batch_inputs = inputs[i : i + args.batch_size]
                    batch_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)

                    # Generate the outputs with scores
                    outputs = model.generate(batch_inputs['input_ids'].cuda(), attention_mask=batch_inputs['attention_mask'].cuda(), return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
                    probs = outputs.scores[0][:, [2163, 465]].softmax(dim=-1)
                    probs_yes = probs[:, 0]

                    # Inverse sigmoid
                    logits = (probs_yes.log() - (1 - probs_yes).log()).cpu().tolist()
                    scores.extend([[logit, 0.0] for logit in logits])

        passages = passages_scored + passages
        passages = [(qid, docid) for docid in passages]
        all_passages.extend(passages)
        all_scores.extend(scores)

    # Write the output
    with open(args.output, 'w') as f:
        for (qid, docid), (score_true, score_false) in zip(all_passages, all_scores):
            f.write(f"{qid}\t{docid}\t{round(score_true, 6)}\t{round(score_false, 6)}\n")