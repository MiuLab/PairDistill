import argparse
import json
import random

from tqdm import tqdm


def convert_line(line, n_neg):
    """
    Original line:
    {"query_id": 906689, "question": "what to do in quito for one day", "positive_ctxs": [{"docidx": 6306009}], "hard_negative_ctxs": [{"docidx": 6029312}, {"docidx": 1775617}, ...]}

    Desired line:
    [906689, 6306009, 6029312, 1775617]
    """
    query_id = line['query_id']
    positive_ctxs = line['positive_ctxs']
    hard_negative_ctxs = line['hard_negative_ctxs']

    converted_line = [query_id]

    # sample one positive context
    positive_ctx = random.choice(positive_ctxs)
    converted_line.append([positive_ctx['docidx'], 1.0, []])
    
    if len(hard_negative_ctxs) < n_neg:
        hard_negative_ctxs = hard_negative_ctxs * (n_neg // len(hard_negative_ctxs) + 1)

    # sample n_neg negative contexts
    hard_negative_ctxs = random.sample(hard_negative_ctxs, n_neg)
    for hard_negative_ctx in hard_negative_ctxs:
        converted_line.append([hard_negative_ctx['docidx'], -999., []])
    
    return converted_line


def convert_dragon_to_colbert(input_file, output_file, n_neg, repeat):
    with open(input_file, 'r') as f:
        data = f.readlines()

    converted_data = []
    for line in tqdm(data):
        line = json.loads(line)
        for _ in range(repeat):
            # Perform the conversion logic here
            converted_line = convert_line(line, n_neg)
            converted_data.append(converted_line)

    with open(output_file, 'w') as f:
        for line in converted_data:
            f.write(json.dumps(line) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Dragon format to Colbert format')
    parser.add_argument('input_file', type=str, help='path to the input file')
    parser.add_argument('output_file', type=str, help='path to the output file')
    parser.add_argument('--n_neg', type=int, default=10, help='number of negatives per query')
    parser.add_argument('--repeat', type=int, default=1, help='number of times to repeat each query')
    args = parser.parse_args()

    convert_dragon_to_colbert(args.input_file, args.output_file, args.n_neg, args.repeat)
