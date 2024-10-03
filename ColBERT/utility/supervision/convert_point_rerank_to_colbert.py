import argparse
import json
import random

import torch
import torch.nn.functional as F

from tqdm import tqdm

from convert_dragon_to_colbert import convert_line as convert_dragon_line


def convert(qid, rerank_scores, n_neg):
    converted_line = [qid]

    rerank_scores = random.sample(rerank_scores, n_neg + 1)
    rerank_scores = sorted(rerank_scores, key=lambda x: x[1], reverse=True)

    for docid, log_prob in rerank_scores:
        converted_line.append([docid, log_prob, []])

    return converted_line


def convert_point_rerank_to_colbert(rerank_file, output_file, orig_file, n_neg, repeat, shuffle, softmax):
    if orig_file is not None:
        print("Reading the orig file...")
        with open(orig_file, 'r') as f:
            data = f.readlines()
    
    rerank_data = {}
    print("Reading the rerank file...")
    with open(rerank_file, 'r') as f:
        for line in tqdm(f):
            qid, docid, score_true, score_false = line.strip().split('\t')
            qid = int(qid)
            if qid not in rerank_data:
                rerank_data[qid] = []
            if softmax:
                prob = F.softmax(torch.tensor([float(score_true), float(score_false)]), dim=0)[0].log().item()
            else:
                prob = float(score_true)
            rerank_data[qid].append((int(docid), prob))

    print("Converting the data...")
    converted_data = []

    if orig_file is not None:
        for line in tqdm(data):
            line = json.loads(line)
            for _ in range(repeat):
                # Perform the conversion logic here
                converted_line = convert_dragon_line(line, n_neg)
                converted_data.append(converted_line)

    for qid in tqdm(rerank_data):
        for _ in range(repeat):
            # Perform the conversion logic here
            converted_line = convert(qid, rerank_data[qid], n_neg)
            converted_data.append(converted_line)
        
    if shuffle:
        random.shuffle(converted_data)

    with open(output_file, 'w') as f:
        for line in converted_data:
            f.write(json.dumps(line) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert pairwise reranking data to Colbert format')
    parser.add_argument('rerank_file', type=str, help='path to the rerank file')
    parser.add_argument('output_file', type=str, help='path to the output file')
    parser.add_argument('--orig_file', type=str, default=None, help='path to the original training data file')
    parser.add_argument('--n_neg', type=int, default=10, help='number of negatives per query')
    parser.add_argument('--repeat', type=int, default=1, help='number of times to repeat each query')
    parser.add_argument('--softmax', action='store_true', help='whether to use softmax')
    parser.add_argument('--shuffle', action='store_true', help='shuffle the data')
    args = parser.parse_args()

    convert_point_rerank_to_colbert(args.rerank_file, args.output_file, args.orig_file, args.n_neg, args.repeat, args.shuffle, args.softmax)
