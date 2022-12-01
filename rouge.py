# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Uses Google's ROUGE implementation to compute scores between candidate and reference summary. """

from rouge_score import rouge_scorer
from rouge_score.scoring import BootstrapAggregator


class RougeScorer:
    """ Computes ROUGE scores. """

    def __init__(self, use_stemmer=True):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=use_stemmer)

    def compute_rouge_score(self, candidate, reference, sentence_sep=None):
        """ Add ROUGE score for candidate-reference pair to aggregator. """
        if len(candidate) < 1 or len(reference) < 1:
            print('Empty candidate or reference')
            print(f'Candidate: {candidate}')
            print(f'Reference: {reference}')
            return
        if sentence_sep:
            candidate = candidate.replace(sentence_sep, '\n')
            reference = reference.replace(sentence_sep, '\n')
        return self.scorer.score(reference, candidate)


class RougeAggregator:
    """ Bootstrap aggregator for ROUGE scores. """

    def __init__(self):
        self.aggregator = BootstrapAggregator()

    def add_scores(self, scores):
        """ Add ROUGE scores from single comparison. """
        self.aggregator.add_scores(scores)

    def get_rouge_scores(self):
        """ Aggregate the scores added by `add_scores`. Returns ROUGE-1/2/L. """
        results = self.aggregator.aggregate()
        return results['rouge1'].mid.fmeasure, results['rouge2'].mid.fmeasure, results['rougeLsum'].mid.fmeasure
