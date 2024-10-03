import argparse
import csv


def convert_to_trec(input_file, output_file):
    with open(input_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        with open(output_file, 'w') as g:
            writer = csv.writer(g, delimiter=' ')
            for row in reader:
                writer.writerow([row[0], 'Q0', row[1], row[2], row[3], 'ColBERT'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert run file to TREC format')
    parser.add_argument('--input', type=str, help='path to the input run file', required=True)
    parser.add_argument('--output', type=str, help='path to the output file', required=True)
    args = parser.parse_args()

    convert_to_trec(args.input, args.output)