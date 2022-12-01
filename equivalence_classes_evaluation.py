# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Equivalence classes evaluation for human annotations. """

import functools
import glob
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
import tqdm
from pytorch_lightning import seed_everything
from pytorch_lightning.core.saving import load_hparams_from_yaml

from data_utils import BartData, BertData
from dataloader import SummarizationDataModule
from main import MODELS


class EquivalenceClassesEvaluation:

    def __init__(self, data_encoder, data_module, model):
        self.data_encoder = data_encoder
        self.data_module = data_module
        self.model = model
        self.scores = defaultdict(list)

    def _evaluate_continuation(self, example, continuation_key):
        # compute prefix length (in tokens)
        example['tgt'] = example['prefix']
        encoded = self.data_encoder.encode(example)
        prefix_length = len(encoded['tgt']) - 1  # without EOS token

        # encode prefix with continuation, run model, get continuation log-probs
        example['tgt'] = example['prefix'] + ' ' + example[continuation_key]
        encoded = self.data_encoder.encode(example)
        batch = self.data_module.collate([encoded])
        assert batch.batch_size == 1
        with torch.no_grad():
            output = self.model(batch)
        logits_start = prefix_length - 1  # without decoder start token
        probs = F.softmax(output.logits[0, logits_start:], dim=-1)
        token_probs = torch.gather(probs, -1, batch.tgt[:, prefix_length:].T).flatten().tolist()
        token_probs = token_probs[:-1]  # remove EOS token
        return functools.reduce(lambda x, y: x * y, token_probs)

    def evaluate(self, example):
        positive_prob = self._evaluate_continuation(example, 'positive_continuation')
        negative_prob = self._evaluate_continuation(example, 'negative_continuation')
        self.scores[example['evaluation']].append(1 if positive_prob > negative_prob else 0)

        # add result for combination of positive/negative equivalence class
        key = f"{example['evaluation']}-pos_{example['positive_class']}-neg_{example['negative_class']}"
        self.scores[key].append(1 if positive_prob > negative_prob else 0)

    def get_results(self):
        return {k: (sum(v), len(v)) for k, v in self.scores.items()}


def restore_training_args(args):
    """ Restores arguments from the checkpoint hparams file. """
    hparams = load_hparams_from_yaml(os.path.join(args.model_dir, 'version_0', 'hparams.yaml'))
    for key, value in hparams.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def load_model(args):
    """ Loads the model. """
    checkpoints = glob.glob(os.path.join(args.model_dir, '*.ckpt'))
    assert len(checkpoints) == 1
    model_class = MODELS[args.model]
    model = model_class(args).load_from_checkpoint(args=args, checkpoint_path=checkpoints[0])
    model.eval()
    return model


def main(args):
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    if os.path.exists(args.output_path) and not args.overwrite:
        return

    seed_everything(args.seed)
    restore_training_args(args)
    model = load_model(args)
    if args.model == 'bart':
        data_encoder = BartData(model.tokenizer)
    else:
        data_encoder = BertData(args.dataset, max_src_tokens=args.max_pos)
    data_module = SummarizationDataModule(args)
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    evaluation = EquivalenceClassesEvaluation(data_encoder, data_module, model)
    for example in tqdm.tqdm(data):
        if args.evaluation and example['evaluation'] != args.evaluation:
            continue
        evaluation.evaluate(example)

    # save perplexities to model dir
    with open(args.output_path, 'w') as f:
        json.dump(evaluation.get_results(), f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Equivalence classes evaluation for annotations.')
    parser.add_argument('--data_path', required=True, help='Path to evaluation examples.')
    parser.add_argument('--model_dir', required=True, help='Path to directory with model checkpoint')
    parser.add_argument('--output_path', required=True, help='Path to output file')
    parser.add_argument('--evaluation', default=None, help='Select a specific evaluation')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite results of previous evaluation')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    main(parser.parse_args())
