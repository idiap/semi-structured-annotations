# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Script that filters source documents with oracle/lead strategy adhering to a length limit for a given tokenizer. """

import json
import os
from abc import ABC, abstractmethod

import nltk
import tqdm
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AddedToken

from data_schema import SchemaFactory
from data_utils import SPLIT_SYMBOL


class Selector(ABC):

    def __init__(self, dataset, tokenizer_name, max_src_tokens):
        annotation_schema = SchemaFactory.get_schema(dataset)
        special_tokens = annotation_schema.get_special_text_tokens()
        special_tokens = [AddedToken(t) for t in special_tokens]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, additional_special_tokens=special_tokens)
        self.max_src_tokens = max_src_tokens

    @abstractmethod
    def select_sentences(self, source_sentences, reference_ststs):
        pass


class LeadSelector(Selector):

    def select_sentences(self, source_sentences, reference_ststs):
        selected_sentence_indices = {0}
        selected_sentence_texts = set()
        num_tokens = len(self.tokenizer.encode(source_sentences[0]))
        for i in range(1, len(source_sentences)):
            # skip duplicate sentences
            if source_sentences[i] in selected_sentence_texts:
                continue

            new_tokens = len(self.tokenizer.encode(source_sentences[i]))
            if num_tokens + new_tokens > self.max_src_tokens:
                break
            selected_sentence_indices.add(i)
            selected_sentence_texts.add(source_sentences[i])
            num_tokens += new_tokens
        return selected_sentence_indices


class RougeSelector(Selector):

    def __init__(self, dataset, tokenizer_name, max_src_tokens):
        super().__init__(dataset, tokenizer_name, max_src_tokens)
        self.scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)

    def select_sentences(self, source_sentences, targets):
        """ Greedily selects source sentences with the highest ROUGE-2 recall, until the token limit is reached. """
        references = '\n'.join(targets)
        selected_sentence_indices = set()
        possible_sentence_indices = set(range(len(source_sentences)))
        selected_sentence_texts = set()
        cur_recall = 0
        cur_length = 0

        # compute sentence lengths
        sentence_lengths = [len(self.tokenizer.encode(s)) for s in source_sentences]

        # add source sentences until above the token limit or the ROUGE recall does not increase anymore
        while True:
            # compute recall values of each source sentence when combined with the already selected sentences
            max_recall_increase = 0
            best_recall = 0
            best_idx = -1
            indices_to_remove = set()
            for i in possible_sentence_indices:
                # skip and remove sentences that are too long
                if cur_length + sentence_lengths[i] > self.max_src_tokens:
                    indices_to_remove.add(i)
                    continue

                # skip and remove duplicate sentences
                if source_sentences[i] in selected_sentence_texts:
                    indices_to_remove.add(i)
                    continue

                sentence_indices = sorted(list(selected_sentence_indices) + [i])
                candidates = '\n'.join([source_sentences[j] for j in sentence_indices])
                recall = self.scorer.score(references, candidates)['rouge2'].recall

                # compute recall increases normalized by sentence length
                recall_increase = (recall - cur_recall) / sentence_lengths[i]
                if recall_increase == 0:
                    # no more recall increases from this sentence: remove
                    indices_to_remove.add(i)
                elif recall_increase > max_recall_increase:
                    max_recall_increase = recall_increase
                    best_recall = recall
                    best_idx = i

            possible_sentence_indices -= indices_to_remove
            cur_recall = best_recall
            if max_recall_increase == 0:
                # no further increase in recall: stop
                break
            else:
                selected_sentence_indices.add(best_idx)
                possible_sentence_indices.remove(best_idx)
                selected_sentence_texts.add(source_sentences[best_idx])
                cur_length += sentence_lengths[best_idx]

        # if length limit has not been reached yet, add sentences from the top of the document
        if cur_length < self.max_src_tokens:
            for i, length in enumerate(sentence_lengths):
                if i in selected_sentence_indices or source_sentences[i] in selected_sentence_texts:
                    continue
                if cur_length + length > self.max_src_tokens:
                    break
                selected_sentence_indices.add(i)
                cur_length += length

        # if no sentence was selected, add the first one even if it exceeds the length limit
        if not selected_sentence_indices:
            selected_sentence_indices = {0}
        return selected_sentence_indices


def clean_target_text(text, special_tokens):
    """ Removes special tokens from target text. """
    text = text.replace(SPLIT_SYMBOL, ' ')
    for token in special_tokens:
        text = text.replace(token, ' ')
    return ' '.join(text.split())  # remove multiple white-space


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.filter_model == 'oracle':
        filter_model = RougeSelector(args.dataset, args.tokenizer_name, args.max_src_tokens)
    else:
        filter_model = LeadSelector(args.dataset, args.tokenizer_name, args.max_src_tokens)
    annotation_schema = SchemaFactory.get_schema(args.dataset)
    special_text_tokens = annotation_schema.get_special_text_tokens()

    for split in ['train', 'valid', 'test']:
        print(f'Processing {split} files...')
        outputs = []
        with open(os.path.join(args.input_dir, f'{split}.txt'), 'r') as f:
            filenames = list(map(str.strip, f.readlines()))
        for filename in tqdm.tqdm(filenames):
            with open(os.path.join(args.input_dir, f'{filename}.src.txt'), 'r') as f:
                source_text = f.read().strip()
            with open(os.path.join(args.input_dir, f'{filename}.tgt.txt'), 'r') as f:
                targets = list(map(str.strip, f.readlines()))
            target_texts = list(map(lambda t: clean_target_text(t, special_text_tokens), targets))

            # split source into sentences
            source_sentences = nltk.sent_tokenize(source_text)

            # select source sentences
            selected_sentence_indices = filter_model.select_sentences(source_sentences, target_texts)
            source_sentences = [s for i, s in enumerate(source_sentences) if i in selected_sentence_indices]

            # add filtered example to output
            for i, target in enumerate(targets):
                outputs.append({
                    'src_sents': source_sentences,
                    'tgt': target,
                    'name': filename,
                    'tgt_i': i,
                })

        with open(os.path.join(args.output_dir, f'{args.dataset}.{args.filter_model}.{split}.json'), 'w') as f:
            json.dump(outputs, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Filters the refdoc sentences.')
    parser.add_argument('--dataset', default='fomc', help='Dataset name')
    parser.add_argument('--input_dir', default='data_fomc_txt', help='Path to input text data directory')
    parser.add_argument('--output_dir', default='data_fomc_bert', help='Path to output directory')
    parser.add_argument('--filter_model', default='oracle', choices=['oracle', 'lead'],
                        help='Model to select refdoc sentences')
    parser.add_argument('--tokenizer_name', default='bert-base-uncased', help='Tokenizer model name or path to dir')
    parser.add_argument('--max_src_tokens', type=int, default=512, help='Limit number of source tokens')
    main(parser.parse_args())
