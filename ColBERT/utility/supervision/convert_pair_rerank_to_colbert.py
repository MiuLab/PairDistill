import argparse
import json
import random

import torch
import torch.nn.functional as F

from tqdm import tqdm

from convert_dragon_to_colbert import convert_line as convert_dragon_line


def convert(qid, rerank_pairs, rerank_scores, n_neg):
    converted_line = [qid]

    n_neg += 1
    ctxs = list(set([docid for docid1, docid2, _ in rerank_pairs for docid in [docid1, docid2]]))
    ctxs = random.sample(ctxs, min(n_neg, len(ctxs)))
    n_neg -= len(ctxs)

    if n_neg > 0:
        # sample n_neg more contexts
        ctxs.extend(random.sample([docid for docid, _ in rerank_scores if docid not in ctxs], n_neg))

    rerank_pairs_dict = {}
    for docid1, docid2, score in rerank_pairs:
        if docid1 not in rerank_pairs_dict:
            rerank_pairs_dict[docid1] = {}
        rerank_pairs_dict[docid1][docid2] = score
    
    rerank_scores_dict = {}
    for docid, score in rerank_scores:
        rerank_scores_dict[docid] = score

    # sort the contexts by their rerank scores
    ctxs = sorted(ctxs, key=lambda x: rerank_scores_dict.get(x, -999), reverse=True)

    for ctx in ctxs:
        pair_scores = []
        for docid2 in rerank_pairs_dict.get(ctx, []):
            score = rerank_pairs_dict[ctx][docid2]
            pair_scores.append([ctxs.index(docid2), score])
        if ctx not in rerank_scores_dict:
            print("Warning: context {} not in rerank_scores_dict".format(ctx))
        converted_line.append([ctx, rerank_scores_dict.get(ctx, 0.0), pair_scores])

    return converted_line


def convert_pair_rerank_to_colbert(rerank_file, point_rerank_file, output_file, orig_file, n_neg, repeat, shuffle, softmax):
    if orig_file is not None:
        print("Reading the orig file...")
        with open(orig_file, 'r') as f:
            data = f.readlines()
    
    rerank_data = {}
    print("Reading the rerank file...")
    with open(rerank_file, 'r') as f:
        for line in tqdm(f):
            qid, docid1, docid2, score_true, score_false = line.strip().split('\t')
            qid = int(qid)
            if qid not in rerank_data:
                rerank_data[qid] = []
            prob = F.softmax(torch.tensor([float(score_true), float(score_false)]), dim=0)[0].log().item()
            rerank_data[qid].append((int(docid1), int(docid2), prob))
    
    if point_rerank_file is not None:
        point_rerank_data = {}
        print("Reading the point rerank file...")
        with open(point_rerank_file, 'r') as f:
            for line in tqdm(f):
                qid, docid, score_true, score_false = line.strip().split('\t')
                qid = int(qid)
                if qid not in point_rerank_data:
                    point_rerank_data[qid] = []
                if softmax:
                    prob = F.softmax(torch.tensor([float(score_true), float(score_false)]), dim=0)[0].log().item()
                else:
                    prob = float(score_true)
                point_rerank_data[qid].append((int(docid), prob))

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
            converted_line = convert(qid, rerank_data[qid], point_rerank_data.get(qid, None), n_neg)
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
    parser.add_argument('--point_rerank_file', type=str, default=None, help='path to the pointwise rerank file')
    parser.add_argument('--n_neg', type=int, default=10, help='number of negatives per query')
    parser.add_argument('--repeat', type=int, default=1, help='number of times to repeat each query')
    parser.add_argument('--softmax', action='store_true', help='whether to use softmax')
    parser.add_argument('--shuffle', action='store_true', help='shuffle the data')
    args = parser.parse_args()

    convert_pair_rerank_to_colbert(args.rerank_file, args.point_rerank_file, args.output_file, args.orig_file, args.n_neg, args.repeat, args.shuffle, args.softmax)
