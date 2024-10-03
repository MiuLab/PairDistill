from tqdm import tqdm
import argparse
import json
import os
from typing import List, Optional

def json_to_tsv(
    in_path: str, 
    out_path: str, 
    meta_list: List[str],
    id_column: str, 
    head: Optional[List[str]] = None,
    ids: Optional[List[str]] = None
):
    id2index = {}
    with open(out_path, 'w') as fout:
        with open(in_path, 'r') as fin:
            for i, line in tqdm(enumerate(fin)):
                content = json.loads(line)
                if (i == 0) and (head is not None):
                    # write head
                    fout.write('\t'.join(head) + '\n')

                if content[id_column] in id2index:
                    print('duplicate id: {}'.format(content[id_column]))

                if ids is not None and content[id_column] not in ids:
                    continue

                id2index[content[id_column]] = str(i)
                content[id_column] = str(i)

                text_list = []
                for item in meta_list:
                    if item == "text" or item == "title":
                        content[item] = ' '.join(content[item].split()) # avoid '\t' and '\n' in text and title to impact file reader
                    text_list.append(content[item])
                fout.write('\t'.join(text_list) + '\n')

    return id2index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    
    qrel_tsv_path = os.path.join(args.data_dir, 'qrels', '{split}.tsv')
    query_json_path = os.path.join(args.data_dir, 'queries.jsonl')
    query_tsv_path = os.path.join(args.data_dir, 'queries.{split}.tsv')
    corpus_json_path = os.path.join(args.data_dir, 'corpus.jsonl')
    corpus_dpr_tsv_path = os.path.join(args.data_dir, 'collection_dpr.tsv')
    corpus_tsv_path = os.path.join(args.data_dir, 'collection.tsv')
    
    print('output collection dpr tsv')
    json_to_tsv(corpus_json_path, corpus_dpr_tsv_path, ["_id", "text", "title"], "_id", ["id", "text", "title"])

    print('output collection tsv')
    pid2index = json_to_tsv(corpus_json_path, corpus_tsv_path, ["_id", "text", "title"], "_id", head=None)

    for split in ['train', 'dev', 'test']:
        if not os.path.exists(qrel_tsv_path.format(split=split)):
            continue
        
        print('output query tsv for split {}'.format(split))
        qids = [
            line.split('\t')[0]
            for line in open(qrel_tsv_path.format(split=split), 'r').readlines()[1:]
        ]
        qid2index = json_to_tsv(query_json_path, query_tsv_path.format(split=split), ["_id", "text"], "_id", ids=qids)

        print('output qrel tsv for split {}'.format(split))
        with open(os.path.join(args.data_dir, f"qrels.{split}.tsv"), 'w') as fout:
            with open(qrel_tsv_path.format(split=split), 'r') as fin:
                for i, line in tqdm(enumerate(fin)):
                    if (i == 0): 
                        continue #skip head
                    qid, pid, rel = line.split('\t')
                    if qid not in qid2index:
                        print('qid {} not in qid2index'.format(qid))
                        qid2index[qid] = str(len(qid2index))
                    if pid not in pid2index:
                        print('pid {} not in pid2index'.format(pid))
                        pid2index[pid] = str(len(pid2index))
                    fout.write('{} {} {} {}'.format(qid2index[qid], 0, pid2index[pid], rel))
    
    print('output duplicate qid and pid')
    with open(os.path.join(args.data_dir, 'duplicate_qid_pid.json'), 'w') as jsonfile:
        duplicate = {}
        for qid in qid2index:
            if qid in pid2index:
                duplicate[qid2index[qid]] = pid2index[qid]
        json.dump(duplicate, jsonfile, indent=4)
            
    
if __name__ == "__main__":
	main()
