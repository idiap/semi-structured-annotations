# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Convert filtered data from BERT-tokenized IDs back to text. """

import json
import os

import nltk
import torch
import tqdm

from data_schema import SchemaFactory
from data_utils import SPLIT_SYMBOL
from preprocess import ReferenceFormat

BERT_SEP_TOKEN_ID = 102


def group_sentences(ids, eos_id=BERT_SEP_TOKEN_ID):
    """ Groups sentences in `ids` by ending sentences at `eos_id`. """
    sents = [[]]
    for i in ids:
        sents[-1].append(i)
        if i == eos_id:
            sents.append([])  # start a new sentence
    assert sents[-1] == []
    return sents[:-1]


def get_original_sentences(data_formatter, example, text_dir):
    """ Gets the original source sentences that appear in example (in preprocessed version). """
    # read original sentences
    with open(os.path.join(text_dir, f"{example['name']}.src.txt"), 'r') as f:
        original_text = f.read().strip()
    original_sentences = nltk.sent_tokenize(original_text)

    # apply same preprocessing, then match ids
    original_sent_ids = data_formatter.preprocess_source(original_sentences)
    selected_sents = group_sentences(example['src'])
    selected_sent_ids = set([repr(sent) for sent in selected_sents])
    source_sents = []
    source_sents_added = set()
    for i, sent_ids in enumerate(original_sent_ids):
        if repr(sent_ids) in selected_sent_ids and repr(sent_ids) not in source_sents_added:
            source_sents.append(original_sentences[i])
            source_sents_added.add(repr(sent_ids))
    assert len(source_sents) == len(selected_sent_ids), "Not all sentences found in original."
    return source_sents


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    annotation_schema = SchemaFactory.get_schema(args.dataset)
    data_formatter = ReferenceFormat(
        annotation_schema, max_src_tokens=args.max_src_tokens, max_tgt_tokens=args.max_tgt_tokens
    )
    for filter_model in ['filterbert', 'oracle']:
        for split in ['train', 'valid', 'test']:
            outputs = []
            source_sents = {}  # only get source sentences once for each refdoc
            encoded_data = torch.load(os.path.join(args.data_dir, f'{args.dataset}.{filter_model}.{split}.pt'))
            for example in tqdm.tqdm(encoded_data):
                # get original source sentences
                if example['name'] not in source_sents:
                    source_sents[example['name']] = get_original_sentences(data_formatter, example, args.text_dir)

                # get target
                with open(os.path.join(args.text_dir, f"{example['name']}.tgt.txt"), 'r') as f:
                    target_text = f.readlines()[example['tgt_i']].strip().replace(SPLIT_SYMBOL, ' ')
                outputs.append({
                    'src_sents': source_sents[example['name']],
                    'tgt': target_text,
                    'name': example['name'],
                    'tgt_i': example['tgt_i'],
                })
            output_path = os.path.join(args.output_dir, f'{args.dataset}.{filter_model}.{split}.json')
            with open(output_path, 'w') as f:
                json.dump(outputs, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Converts BERT-encoded data to text.')
    parser.add_argument('--dataset', default='fomc', choices=['fomc'], help='Dataset name')
    parser.add_argument('--max_src_tokens', type=int, default=512, help='Maximum number of source tokens')
    parser.add_argument('--max_tgt_tokens', type=int, default=512, help='Maximum number of target tokens')
    parser.add_argument('--text_dir', required=True, help='Path to raw text dir')
    parser.add_argument('--data_dir', required=True, help='Path to filtered data dir')
    parser.add_argument('--output_dir', required=True, help='Path to output dir')
    main(parser.parse_args())
