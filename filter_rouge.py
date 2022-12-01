# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Filter the reference sentences by ROUGE overlap with the annotated sentences. This is the target for FilterBERT. """

import glob
import os
import time

import torch
from numpy import argmax
from rouge_score import rouge_scorer
from transformers import BertTokenizer

from data_schema import SchemaFactory


class Selector:

    def __init__(self, dataset, pretrained_dir, max_src_tokens=512):
        special_tokens = SchemaFactory.get_schema(dataset).get_special_tokens()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, additional_special_tokens=special_tokens)
        self.scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
        self.max_src_tokens = max_src_tokens

    def select_sentences(self, src_ids, tgt_ids):
        """ Selects source sentence with the highest ROUGE-2 recall, until the token limit is reached. """
        src_sents = [self.tokenizer.decode(sent_ids, skip_special_tokens=True) for sent_ids in src_ids]
        tgt_sents = [self.tokenizer.decode(sent_ids, skip_special_tokens=True) for sent_ids in tgt_ids]
        references = '\n'.join(tgt_sents)
        selected_src_sents = []
        cur_recall = 0

        # add source sentences until above the token limit or the ROUGE recall does not increase anymore
        while True:
            # compute recall values of each source sentence when combined with the already selected sentences
            combined_recall = []
            for i, candidate in enumerate(src_sents):
                # skip already selected sentences
                if i in selected_src_sents:
                    combined_recall.append(cur_recall)
                    continue

                cur_src_sents = sorted(selected_src_sents + [i])
                candidates = '\n'.join([src_sents[j] for j in cur_src_sents])
                recall = self.scorer.score(references, candidates)['rouge2'].recall
                combined_recall.append(recall)

            # compute recall increases normalized by sentence length (num tokens)
            recall_increases = [(recall - cur_recall) / len(src_ids[i]) for i, recall in enumerate(combined_recall)]

            # pick the source sentence with the highest recall increase
            best_idx = argmax(recall_increases)
            if recall_increases[best_idx] <= 0:
                # no further increase in recall: stop
                break
            cur_recall = combined_recall[best_idx]

            # check if the combined length exceeds the limit
            cur_selected_src_sents = sorted(selected_src_sents + [best_idx])
            combined_length = sum([len(src_ids[i]) for i in cur_selected_src_sents])
            if combined_length < self.max_src_tokens:
                selected_src_sents = cur_selected_src_sents
            elif not selected_src_sents:
                # if first selected sentence already exceeds the limit, add it nevertheless
                selected_src_sents = [best_idx]
                break
            else:
                # adding the next sentence would exceed the token limit: stop
                break
        return selected_src_sents


def main(args):
    selector = Selector(args.dataset, args.pretrained_dir, max_src_tokens=args.max_src_tokens)
    for split in ['train', 'valid', 'test']:
        data_files = sorted(glob.glob(os.path.join(args.datadir, f'{args.dataset}.{split}.pt')))
        for path in data_files:
            print(f'Processing {path}')
            data = torch.load(path)
            scores = []
            start_time = time.time()
            for i, example in enumerate(data, 1):
                src_ids = example['srcs']
                tgt_ids = example['tgts']
                selected_indices = selector.select_sentences(src_ids, tgt_ids)
                filter_target = [1 if i in selected_indices else 0 for i in range(len(src_ids))]
                scores.append({
                    'filter_target': filter_target,
                })
                if i % 10 == 0 and i > 0:
                    print(f'Processed {i} of {len(data)} examples in {time.time() - start_time:.1f}s.')
            path = path[:-3] + '.rouge.pt'
            torch.save(scores, path)
            print(f'Finished processing {path} after {time.time() - start_time:.1f}s.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Filter references with ROUGE.')
    parser.add_argument('--dataset', default='fomc', help='Dataset name')
    parser.add_argument('--datadir', default='data_fomc_pt', help='Path to data folder')
    parser.add_argument('--pretrained_dir', default='bert-base-uncased', help='Pretrained model name or dir')
    parser.add_argument('--max_src_tokens', type=int, default=512, help='Limit number of source tokens (0 = no limit)')
    main(parser.parse_args())
