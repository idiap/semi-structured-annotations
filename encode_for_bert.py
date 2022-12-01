# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Encode text data for BERT generation. """

import json
import os

import torch
import tqdm

from data_utils import BertData, SPLIT_SYMBOL


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    encoder = BertData(args.dataset, tgt_format='presumm', max_src_tokens=args.max_src_tokens)
    for filter_model in ['filterbert', 'oracle', 'lead']:
        for split in ['train', 'valid', 'test']:
            data_path = os.path.join(args.text_dir, f'{args.dataset}.{filter_model}.{split}.json')
            if os.path.exists(data_path):
                with open(data_path, 'r') as f:
                    data = json.load(f)
                outputs = []
                for example in tqdm.tqdm(data):
                    outputs.append(encoder.encode(example, sent_sep=SPLIT_SYMBOL))
                torch.save(outputs, os.path.join(args.output_dir, f'{args.dataset}.{filter_model}.{split}.pt'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Encode text for BERT generation.')
    parser.add_argument('--tokenizer_name', default='bert-base-uncased', help='Tokenizer name or path to dir')
    parser.add_argument('--dataset', default='fomc', help='Dataset name')
    parser.add_argument('--max_src_tokens', type=int, default=512, help='Maximum number of source tokens')
    parser.add_argument('--max_tgt_tokens', type=int, default=512, help='Maximum number of target tokens')
    parser.add_argument('--text_dir', required=True, help='Path to text dir')
    parser.add_argument('--output_dir', required=True, help='Path to output dir')
    main(parser.parse_args())
