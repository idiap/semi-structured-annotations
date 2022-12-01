# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Creates examples for the equivalence classes evaluation. """

import json
import os
import random

import numpy as np

from data_schema import SchemaFactory
from data_utils import SPLIT_SYMBOL

ACT_ONLY_EVALUATIONS = [
    'act',
    'act modals (positive)',
    'act modals (negative)',
    'act negation',
]

ACT_LABEL_EVALUATIONS = [
    'act labels',
    'act label modals (positive)',
    'act label modals (negative)',
    'act label negation',
]

CATEGORY_EVALUATIONS = [
    'act',
    'act labels',
    'attribution',
    'evidence',
    'motive',
    'temporal scope',
]


def find_positive_class(equivalence_classes, span):
    """ Returns the index of the positive class that contains `span`. """
    for i, equivalence_class in enumerate(equivalence_classes):
        if span in equivalence_class:
            return i
    return -1


def sample_negative_class(equivalence_classes, positive_class):
    """ Samples an equivalence class other than the one with the positive span. """
    possible_negative_classes = [i for i in range(len(equivalence_classes)) if i != positive_class]
    return random.choice(possible_negative_classes)


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    random.seed(args.seed)
    annotation_schema = SchemaFactory.get_schema(args.dataset)
    with open(args.equiv_classes_path, 'r') as f:
        equiv_classes_evaluations = json.load(f)
    for split in ['valid', 'test']:
        print(f'Creating examples for {split} split...')
        evaluation_examples = []
        with open(os.path.join(args.data_dir, f'{args.dataset}.filterbert.{split}.json'), 'r') as f:
            data = json.load(f)
        for equiv_evaluation in equiv_classes_evaluations:
            evaluation_name = equiv_evaluation['evaluation']
            positive_examples = []
            equivalence_classes = equiv_evaluation['equivalence_classes']
            start_marker = annotation_schema.mapping[equiv_evaluation['annotation']]['text_start']
            end_marker = annotation_schema.mapping[equiv_evaluation['annotation']]['text_end']
            for example in data:
                # only use a single StSt as prefix
                for stst in example['tgt'].split(SPLIT_SYMBOL):
                    cur_pos = 0
                    while start_marker in stst[cur_pos:]:
                        cur_tgt = stst[cur_pos:]
                        prefix = stst[:cur_pos] + cur_tgt[:cur_tgt.index(start_marker) + len(start_marker)].strip()
                        span_start = cur_tgt.index(start_marker) + len(start_marker)
                        span_end = cur_tgt.index(end_marker)
                        span = cur_tgt[span_start:span_end].strip()
                        cur_pos += span_end + len(end_marker)
                        if evaluation_name in ACT_ONLY_EVALUATIONS:
                            if '(' in span:
                                span = span[:span.index('(')].strip()
                            else:
                                continue
                        elif evaluation_name in ACT_LABEL_EVALUATIONS:
                            if '(' in span and ')' in span:
                                prefix = prefix + ' ' + span[:span.index('(')].strip()
                                span_start = span.index('(')  # include ( and ) for act label spans
                                span_end = span.index(')') + 1
                                span = span[span_start:span_end].strip()
                            else:
                                continue
                        if evaluation_name in ACT_LABEL_EVALUATIONS:
                            # remove ( and ) for act label to find positive equivalence class
                            positive_class = find_positive_class(equivalence_classes, span[1:-1])  # without ( and )
                        else:
                            positive_class = find_positive_class(equivalence_classes, span)
                        if positive_class > -1:
                            if args.replace_positive and len(equivalence_classes[positive_class]) == 1:
                                continue
                            elif args.replace_positive:
                                new_span = span
                                while span == new_span:
                                    new_span = random.choice(equivalence_classes[positive_class])
                                span = new_span
                            positive_examples.append({
                                'evaluation': evaluation_name,
                                'src_sents': example['src_sents'],
                                'prefix': prefix,
                                'positive_continuation': span,
                                'positive_class': positive_class,
                                'name': example['name'],
                                'tgt_i': example['tgt_i'],
                            })

            # if we're not at the target number of evaluation examples yet, sample multiple negatives per positive
            negatives_per_positive = (
                max(1, args.num_examples_per_evaluation // len(positive_examples)) if len(positive_examples) else 0
            )
            len_before = len(evaluation_examples)
            for example in positive_examples:
                sampled_spans = set()
                for neg_i in range(negatives_per_positive):
                    negative_class = -1
                    negative_span = ''
                    len_diff = np.abs(len(negative_span.split()) - len(example['positive_continuation'].split()))
                    num_tries = 0
                    while not negative_span or negative_span in sampled_spans or len_diff > args.max_len_diff > -1:
                        negative_class = sample_negative_class(equivalence_classes, example['positive_class'])
                        negative_span = random.choice(equivalence_classes[negative_class])
                        if example['evaluation'] in ACT_LABEL_EVALUATIONS:  # add ( and ) for act labels
                            negative_span = f'({negative_span})'
                        len_diff = np.abs(len(negative_span.split()) - len(example['positive_continuation'].split()))
                        num_tries += 1
                        if num_tries > 100:  # stop if no more negative examples are present
                            break
                    if num_tries > 100:
                        break
                    assert negative_span and negative_class > -1
                    sampled_spans.add(negative_span)
                    example['negative_continuation'] = negative_span
                    example['negative_class'] = negative_class
                    evaluation_examples.append(example.copy())
            print(f"Evaluation of {evaluation_name}: {len(positive_examples)} positives, "
                  f"{negatives_per_positive} negatives per positive, {len(evaluation_examples) - len_before} total.")
        num_category_evaluations = len([1 for e in evaluation_examples if e['evaluation'] in CATEGORY_EVALUATIONS])
        print(f"Category evaluation instances: {num_category_evaluations}")
        print(f"Total evaluation instances: {len(evaluation_examples)}")
        output_path = os.path.join(args.output_dir, f'equivalence_classes_evaluation.{args.dataset}.{split}.json')
        with open(output_path, 'w') as f:
            json.dump(evaluation_examples, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Perplexity evaluation for annotations.')
    parser.add_argument('--dataset', default='fomc', help='Dataset name')
    parser.add_argument('--data_dir', default='data_fomc_filtered_text', help='Path to data directory')
    parser.add_argument('--output_dir', default='data_equivalence_classes', help='Path to output directory')
    parser.add_argument('--equiv_classes_path', default='equivalence_classes.json', help='Path to equivalence classes')
    parser.add_argument('--replace_positive', action='store_true',
                        help='Replace positive continuation with a different example from the same equivalence class')
    parser.add_argument('--num_examples_per_evaluation', type=int, default=100,
                        help='Target number of examples per evaluation')
    parser.add_argument('--max_len_diff', type=int, default=2,
                        help='Maximum length difference for positive and negative span (-1: no restriction)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    main(parser.parse_args())
