import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--duplicate_ids", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.duplicate_ids, 'r') as jsonfile:
        duplicate = json.load(jsonfile)

    predictions = {}
    for line in open(args.input_file, 'r'):
        qid, pid, _, score = line.strip().split('\t')
        score = float(score)
        if duplicate.get(qid, None) == pid:
            continue
        
        if qid not in predictions:
            predictions[qid] = []
        predictions[qid].append((pid, score))
    
    with open(args.output_file, 'w') as fout:
        for qid, scores in predictions.items():
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            for rank, (pid, score) in enumerate(scores):
                fout.write(f"{qid}\t{pid}\t{rank+1}\t{score}\n")
    

if __name__ == "__main__":
    main()