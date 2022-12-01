# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Evaluations on saved model outputs. """

import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from bert_score import BERTScorer

from data_schema import SchemaFactory


class Evaluation(ABC):

    def __init__(self, args):
        self.input_dir = args.input_dir
        self.schema = SchemaFactory.get_schema(args.dataset)

    def read_file(self, filename):
        with open(os.path.join(self.input_dir, filename), 'r') as f:
            lines = list(map(str.strip, f.readlines()))
        return lines

    def get_special_tokens(self):
        """ Returns a list of special text tokens. """
        return set(self.schema.get_special_text_tokens())

    def remove_special_tokens(self, text):
        for st in self.get_special_tokens():
            text = text.replace(st, '')
        text = ' '.join(text.split())  # replace multiple white-spaces
        return text

    @abstractmethod
    def run(self):
        pass


class Rouge(Evaluation):
    """ Computes ROUGE-1/2/L, a measure of textual overlap between the generated and the reference StSts. """

    def __init__(self, args, remove_special_tokens=False):
        super().__init__(args)

        from rouge import RougeScorer, RougeAggregator
        self.sentence_separator = args.sentence_separator
        self.reference_separator = args.reference_separator
        self.do_remove_special_tokens = remove_special_tokens
        self.scorer = RougeScorer()
        self.aggregator = RougeAggregator()

    def run(self):
        references = self.read_file('references.txt')
        candidates = self.read_file('candidates.txt')
        assert len(references) == len(candidates), "Number of references and candidates differ"
        if self.do_remove_special_tokens:
            references = map(self.remove_special_tokens, references)
            candidates = map(self.remove_special_tokens, candidates)
        else:
            references = [r.replace('_', '') for r in references]  # remove _ in annotation markers, so they don't get
            candidates = [c.replace('_', '') for c in candidates]  # tokenized to separate tokens in rouge_score
        references = [r.split(self.reference_separator) for r in references]
        for candidate, reference_list in zip(candidates, references):
            scores = [
                self.scorer.compute_rouge_score(candidate, r, sentence_sep=self.sentence_separator)
                for r in reference_list
            ]
            best_score = max(scores, key=lambda x: x['rouge1'].fmeasure + x['rouge2'].fmeasure + x['rougeLsum'].fmeasure)
            self.aggregator.add_scores(best_score)
        r1, r2, rL = self.aggregator.get_rouge_scores()
        return {'rouge1': r1, 'rouge2': r2, 'rougeL': rL}


class DistinctNGrams(Evaluation):
    """ Computes the distinct n-grams generated. """

    def __init__(self, args):
        super().__init__(args)

        self.sentence_separator = args.sentence_separator
        self.ngrams = args.ngrams

    def run(self):
        candidates = self.read_file('candidates.txt')
        candidates = map(self.remove_special_tokens, candidates)
        distinct_ngrams = defaultdict(set)
        for candidate in candidates:
            sentences = candidate.split(self.sentence_separator)
            for sentence in sentences:
                words = sentence.split()
                for n in self.ngrams:
                    distinct_ngrams[n] = distinct_ngrams[n] | set((tuple(words[i:i+n]) for i in range(len(words)-n+1)))
        return {n: len(distinct_ngrams[n]) for n in self.ngrams}


class NovelNGrams(Evaluation):
    """ Computes the fraction of n-grams in the generated StSts that don't appear in the filtered refdoc. """

    def __init__(self, args):
        super().__init__(args)

        self.sentence_separator = args.sentence_separator
        self.ngrams = args.ngrams

    def run(self):
        sources = self.read_file('sources.txt')
        candidates = self.read_file('candidates.txt')
        sources = map(self.remove_special_tokens, sources)
        candidates = map(self.remove_special_tokens, candidates)

        def compute_ngram_overlap(src_words, tgt_words, n):
            """ Computes fraction of target n-grams appearing in source. """
            assert n > 0, "N for n-grams overlap needs to be positive"
            assert len(tgt_words) >= n, "Not enough words in target"
            src_ngrams = set((tuple(src_words[i:i + n]) for i in range(len(src_words) - n + 1)))
            tgt_ngrams = set((tuple(tgt_words[i:i + n]) for i in range(len(tgt_words) - n + 1)))
            ngram_overlap = src_ngrams & tgt_ngrams
            return len(ngram_overlap) / len(tgt_ngrams)

        novel_ngrams = defaultdict(list)
        for source, candidate in zip(sources, candidates):
            source_words = source.split()  # separated by periods, so matches across sentence boundaries are unlikely
            candidate_sentence_words = [sentence.split() for sentence in candidate.split(self.sentence_separator)]
            for n in self.ngrams:
                # remove candidate sentences that are shorter than n
                candidate_sents = [words for words in candidate_sentence_words if len(words) >= n]

                # compute overlaps individually per candidate sentence (avoid match over sentence boundaries),
                # then weight by its length share
                overlaps = [compute_ngram_overlap(source_words, candidate_words, n)
                            for candidate_words in candidate_sents]
                lengths = [len(candidate_words) for candidate_words in candidate_sentence_words]
                weighted_overlap = sum([o * l for o, l in zip(overlaps, lengths)]) / sum(lengths)
                novel_ngrams[n].append(1 - weighted_overlap)

        return {n: np.mean(novel_ngrams[n]) for n in self.ngrams}


class Length(Evaluation):
    """ Computes the length of generated StSts measured by number of StSts, tokens and words generated. """

    def __init__(self, args):
        super().__init__(args)

        self.sentence_separator = args.sentence_separator

    def run(self):
        candidates = self.read_file('candidates.txt')
        num_sents = []
        num_tokens = []
        num_words = []
        num_special_tokens = []
        tokens_per_sent = []
        words_per_sent = []
        special_tokens_per_sent = []
        for candidate in candidates:
            sents = list(filter(None, candidate.split(self.sentence_separator)))
            tokens = [t for s in sents for t in s.split()]
            words = list(filter(lambda w: w not in self.get_special_tokens(), tokens))
            special_tokens = list(filter(lambda t: t in self.get_special_tokens(), tokens))
            num_sents.append(len(sents))
            num_tokens.append(len(tokens))
            num_words.append(len(words))
            num_special_tokens.append(len(special_tokens))
            tokens_per_sent.append(len(tokens) / len(sents))
            words_per_sent.append(len(words) / len(sents))
            special_tokens_per_sent.append(len(special_tokens) / len(sents))
        return {
            'num_sents': {'mean': np.mean(num_sents), 'std': np.std(num_sents)},
            'num_tokens': {'mean': np.mean(num_tokens), 'std': np.std(num_tokens)},
            'num_words': {'mean': np.mean(num_words), 'std': np.std(num_words)},
            'num_special_tokens': {'mean': np.mean(num_special_tokens), 'std': np.std(num_special_tokens)},
            'tokens_per_sent': {'mean': np.mean(tokens_per_sent), 'std': np.std(tokens_per_sent)},
            'words_per_sent': {'mean': np.mean(words_per_sent), 'std': np.std(words_per_sent)},
            'special_tokens_per_sent': {'mean': np.mean(special_tokens_per_sent),
                                        'std': np.std(special_tokens_per_sent)},
        }


class Annotations(Evaluation):
    """ Computes the number and length of each type of generated annotations, and whether they are correctly closed. """

    def __init__(self, args):
        super().__init__(args)

        self.sentence_separator = args.sentence_separator

    def run(self):
        candidates = self.read_file('candidates.txt')
        annotation_counts = defaultdict(list)
        closed_correctly = []
        for candidate in candidates:
            counts = defaultdict(int)
            sents = list(filter(None, candidate.split(self.sentence_separator)))
            for sent in sents:
                tokens = sent.split()
                special_tokens = list(filter(lambda t: t in self.get_special_tokens(), tokens))
                for k, v in self.schema.mapping.items():
                    is_open = False
                    for st in special_tokens:
                        if st == v['text_start']:
                            counts['opened'] += 1
                            is_open = True
                        elif st == v['text_end'] and is_open:
                            counts[k] += 1
                            counts['closed'] += 1
                            is_open = False
            for k in self.schema.mapping.keys():
                annotation_counts[k].append(counts[k])
            closed_correctly.append(counts['closed'] / counts['opened'])
        results = {a: {'mean': np.mean(c), 'std': np.std(c)} for a, c in sorted(annotation_counts.items())}
        results.update({'closed_correctly': np.mean(closed_correctly)})
        return results


class BERTScore(Evaluation):
    """ Computes the semantic similarity between generated and reference StSts with BERTScore (Zhang et al., 2020). """

    def __init__(self, args):
        super().__init__(args)

        self.sentence_separator = args.sentence_separator
        self.reference_separator = args.reference_separator

    def run(self):
        references = self.read_file('references.txt')
        candidates = self.read_file('candidates.txt')
        references = map(self.remove_special_tokens, references)
        candidates = map(self.remove_special_tokens, candidates)

        references = [r.replace(self.sentence_separator, ' ') for r in references]
        references = [r.split(self.reference_separator) for r in references]
        candidates = [c.replace(self.sentence_separator, ' ') for c in candidates]
        scorer = BERTScorer(model_type='roberta-large', lang='en', rescale_with_baseline=True)
        (p, r, f1), hash = scorer.score(cands=candidates, refs=references, return_hash=True)
        return {'f1': np.mean(f1.tolist()), 'hash': hash}


class EvaluationFactory:

    _registry = {
        'rouge': Rouge,
        'rouge_words': Rouge,
        'distinct_ngrams': DistinctNGrams,
        'novel_ngrams': NovelNGrams,
        'length': Length,
        'annotations': Annotations,
        'bertscore': BERTScore,
    }

    @staticmethod
    def get_evaluations():
        return EvaluationFactory._registry.keys()

    @staticmethod
    def run_evaluation(name, args):
        kwargs = {'args': args} if name != 'rouge_words' else {'args': args, 'remove_special_tokens': True}
        evaluation = EvaluationFactory._registry[name](**kwargs)
        return evaluation.run()


def main(args):
    evaluations = EvaluationFactory.get_evaluations()
    results = {}
    results_path = os.path.join(args.input_dir, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    for e in evaluations:
        if args.overwrite or e not in results:
            print("Running evaluation", e)
            results[e] = EvaluationFactory.run_evaluation(e, args)
    with open(results_path, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run evaluations on saved model outputs.')
    parser.add_argument('--input_dir', required=True, help='Path to model outputs directory.')
    parser.add_argument('--dataset', default='fomc', help='Dataset name')
    parser.add_argument('--overwrite', action='store_true', help='Rerun evaluations even if present.')
    parser.add_argument('--sentence_separator', default='<sent>', help='Sentence separator')
    parser.add_argument('--reference_separator', default='<ref>', help='Reference separator')
    parser.add_argument('--ngrams', type=int, nargs='+', default=[1, 2, 3, 4], help='List of n for n-gram statistics.')
    main(parser.parse_args())
