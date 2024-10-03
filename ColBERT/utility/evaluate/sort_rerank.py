import argparse

import torch
import torch.nn.functional as F


def main(input_file, output_file, mono_file=None, mono_sorted=False, softmax=False):
    with open(input_file if mono_file is None else mono_file, 'r') as f:
        lines = f.readlines()

    scores = {}
    for line in lines:
        qid, pid, score_true, score_false = line.strip().split('\t')
        qid, pid, score_true, score_false = int(qid), int(pid), float(score_true), float(score_false)

        if softmax:
            score = F.softmax(torch.tensor([score_true, score_false]), dim=0)[0].item()
        elif mono_sorted:
            score = score_false
        else:
            score = score_true

        if qid not in scores:
            scores[qid] = []
        scores[qid].append((pid, score))
        scores[qid] = sorted(scores[qid], key=lambda x: x[1], reverse=True)
    
    if mono_file is not None:
        # read input file as duo reranked file
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        scores_duo = {}
        for line in lines:
            qid, pid1, pid2, score_true, score_false = line.strip().split('\t')
            qid, pid1, pid2, score_true, score_false = int(qid), int(pid1), int(pid2), float(score_true), float(score_false)

            if qid not in scores_duo:
                scores_duo[qid] = {}

            score = F.softmax(torch.tensor([score_true, score_false]), dim=0)[0].item()
            
            scores_duo[qid][pid1] = scores_duo[qid].get(pid1, 0.0) + score
            scores_duo[qid][pid2] = scores_duo[qid].get(pid2, 0.0) + (1 - score)

        for qid in scores_duo:
            duo_topk = len(scores_duo[qid])
            scores_duo_list = [
                (pid, score*100) for pid, score in scores_duo[qid].items()
            ]
            scores_duo_list = sorted(scores_duo_list, key=lambda x: x[1], reverse=True)
            scores[qid][:duo_topk] = scores_duo_list            

    with open(output_file, 'w') as f:
        for qid in scores:
            for rank, (pid, score) in enumerate(scores[qid]):
                f.write(f'{qid}\t{pid}\t{rank}\t{score}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--mono_file', type=str, default=None)
    parser.add_argument('--softmax', action='store_true')
    parser.add_argument('--mono_sorted', action='store_true')
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.mono_file, args.mono_sorted, args.softmax)