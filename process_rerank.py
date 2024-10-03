import argparse
import time
import random

import torch
import torch.nn.functional as F

from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pointwise reranking with monoT5')
    parser.add_argument('--runfile', type=str, help='path to the run file', required=True)
    parser.add_argument('--output', type=str, help='path to the output file', required=True)
    parser.add_argument('--softmax', action='store_true', help='whether to use softmax')
    args = parser.parse_args()

    # Read the run file
    print("Reading the run file...")
    run = {}
    with open(args.runfile, 'r') as f:
        for line in f:
            qid, docid, score_true, score_false = line.strip().split('\t')
            if qid not in run:
                run[qid] = []
            if args.softmax:
                prob = F.softmax(torch.tensor([float(score_true), float(score_false)]), dim=0)[0].log().item()
            else:
                prob = float(score_true)
            run[qid].append((docid, prob))

    with open(args.output, 'w') as f:
        for qid in tqdm(run):
            run[qid] = sorted(run[qid], key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(run[qid]):
                f.write(f'{qid}\t{docid}\t{rank + 1}\t{score}\n')