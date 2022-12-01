# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Encode text data for BART generation. """

import json
import os

import torch
import tqdm
from transformers import BartTokenizer, AddedToken

from data_schema import SchemaFactory
from data_utils import BartData, SPLIT_SYMBOL


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    special_tokens = SchemaFactory.get_schema(args.dataset).get_special_text_tokens()
    special_tokens = [AddedToken(t) for t in special_tokens]
    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_name, additional_special_tokens=special_tokens)
    encoder = BartData(tokenizer, args.separate_sentences, args.max_src_tokens, args.max_tgt_tokens)
    for filter_model in ['filterbert', 'oracle', 'lead']:
        for split in ['train', 'valid', 'test']:
            data_path = os.path.join(args.text_dir, f'{args.dataset}.{filter_model}.{split}.json')
            if os.path.exists(data_path):
                with open(data_path, 'r') as f:
                    data = json.load(f)
                outputs = []
                for example in tqdm.tqdm(data):
                    example['tgt'] = example['tgt'].replace(SPLIT_SYMBOL, ' ')
                    outputs.append(encoder.encode(example))
                torch.save(outputs, os.path.join(args.output_dir, f'{args.dataset}.{filter_model}.{split}.pt'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Encode text for BART generation.')
    parser.add_argument('--tokenizer_name', default='facebook/bart-large', help='Tokenizer name or path to dir')
    parser.add_argument('--dataset', default='fomc', help='Dataset name')
    parser.add_argument('--separate_sentences', action='store_true',
                        help='Encode source sentences separately, surrounded by BOS/EOS tokens.')
    parser.add_argument('--max_src_tokens', type=int, default=1024, help='Maximum number of source tokens')
    parser.add_argument('--max_tgt_tokens', type=int, default=512, help='Maximum number of target tokens')
    parser.add_argument('--text_dir', required=True, help='Path to text dir')
    parser.add_argument('--output_dir', required=True, help='Path to output dir')
    main(parser.parse_args())
