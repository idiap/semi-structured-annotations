# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" FilterBERT: BERT-based model that selects source document sentences to keep. """

from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, GPUStatsMonitor
from pytorch_lightning.core import LightningModule
from rouge_score.rouge_scorer import RougeScorer
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

from filter_data import FilteringBatch
from filter_data import FilteringDataModule


class FilterBert(LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        model_name_or_path = self.hparams.pretrained_dir or 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.model = BertForSequenceClassification.from_pretrained(model_name_or_path)
        self.scorer = RougeScorer(['rouge2'], use_stemmer=True)
        self.automatic_optimization = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--pretrained_dir', default=None, help='Path to pretrained model')

        # optimization args
        parser.add_argument('--monitor', default='val_overlap_f1', choices=['val_overlap_f1', 'val_selection_f1'],
                            help='Monitor variable to select best model')
        parser.add_argument('--max_lr', type=float, default=1e-3, help='Maximum learning rate.')
        parser.add_argument('--warmup', type=float, default=0.1, help='Fraction of training spent in warmup')
        parser.add_argument('--lr_anneal', default='linear', choices=['linear', 'cos'],
                            help='Annealing of learning rate')
        parser.add_argument('--div_warmup', type=float, default=100,
                            help='Divide max learning rate by this for initial learning rate')
        parser.add_argument('--div_final', type=float, default=100,
                            help='Divide max learning rate by this for final learning rate')

        return parser

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters())
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=self.hparams.max_lr,
            total_steps=self.hparams.max_steps,
            pct_start=self.hparams.warmup,
            anneal_strategy=self.hparams.lr_anneal,
            cycle_momentum=False,
            div_factor=self.hparams.div_warmup,
            final_div_factor=self.hparams.div_final,
            last_epoch=-1,
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    def training_step(self, batch: FilteringBatch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        num_sentences = batch.srcs.size(0)
        loss = 0
        for i in range(0, num_sentences, self.hparams.batch_size):
            classifier_output = self(
                input_ids=batch.srcs[i:i+self.hparams.batch_size],
                attention_mask=batch.mask_srcs[i:i+self.hparams.batch_size],
                labels=batch.filter_target[i:i+self.hparams.batch_size],
            )
            self.manual_backward(classifier_output.loss)  # clear the computation graph to save memory
            true_batch_size = len(batch.srcs[i:i+self.hparams.batch_size])  # also correct for last batch
            loss += classifier_output.loss.item() * true_batch_size  # CrossEntropyLoss uses 'mean' reduction
        loss /= num_sentences
        self.log('train_loss', loss)
        opt.step()

    @staticmethod
    def evaluate_selection(predictions, targets):
        """ Computes precision, recall and F1 scores. """
        assert predictions.dim() == 1, 'Unexpected dimensionality of predictions'
        assert targets.dim() == 1, 'Unexpected dimensionality of targets'
        assert len(predictions) == len(targets), "Predictions and targets length do not match"
        accuracy = torch.sum(predictions == targets).item() / len(targets)
        true_positives = torch.sum(predictions + targets == 2).item()
        positive_predictions = torch.sum(predictions).item()
        positive_targets = torch.sum(targets).item()
        precision = true_positives / positive_predictions if positive_predictions > 0 else 0
        recall = true_positives / positive_targets if positive_targets > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return accuracy, precision, recall, f1

    @staticmethod
    def pick_top_logits(logits, num_tokens, token_limit=512):
        """ Picks the indices with the highest logits until `token_limit` is reached. """
        assert logits.dim() == 2, 'Unexpected dimensionality of logits'
        assert logits.size(0) == len(num_tokens), 'Lengths of logits and `num_tokens` does not match'
        logits_pick_sentence = logits[:, 1].tolist()  # only look at logits to pick a sentence
        sorted_indices = reversed(np.argsort(logits_pick_sentence))
        indices_picked = torch.zeros(logits.size(0), dtype=torch.int64, device=logits.device)
        tokens_picked = 0
        for next_idx in sorted_indices:
            tokens_picked += num_tokens[next_idx]
            if tokens_picked > token_limit:
                break
            indices_picked[next_idx] = 1
        return indices_picked

    def validation_step(self, batch: FilteringBatch, batch_idx):
        num_sentences = batch.srcs.size(0)
        accumulated_loss = 0
        logits = []
        for i in range(0, num_sentences, self.hparams.batch_size):
            classifier_output = self(
                input_ids=batch.srcs[i:i+self.hparams.batch_size],
                attention_mask=batch.mask_srcs[i:i+self.hparams.batch_size],
                labels=batch.filter_target[i:i+self.hparams.batch_size],
            )
            accumulated_loss += classifier_output.loss.item()
            logits.append(classifier_output.logits.detach())
        self.log('val_loss', accumulated_loss / num_sentences)

        # compute metrics on selection of sentences
        logits = torch.cat(logits)
        predictions = FilterBert.pick_top_logits(logits, batch.mask_srcs.sum(dim=1).tolist())
        acc_sel, prec_sel, rec_sel, f1_sel = FilterBert.evaluate_selection(predictions, batch.filter_target)

        # compute metrics on overlap of reference and selected text
        reference_sent_indices = [i for i, picked in enumerate(batch.filter_target) if picked]
        candidate_sent_indices = [i for i, picked in enumerate(predictions) if picked]
        reference_sentences = [self.tokenizer.decode(batch.srcs[i], skip_special_tokens=True)
                               for i in reference_sent_indices]
        candidate_sentences = [self.tokenizer.decode(batch.srcs[i], skip_special_tokens=True)
                               for i in candidate_sent_indices]
        results = self.scorer.score('\n'.join(reference_sentences), '\n'.join(candidate_sentences))
        prec_overlap = results['rouge2'].precision
        rec_overlap = results['rouge2'].recall
        f1_overlap = results['rouge2'].fmeasure
        return prec_overlap, rec_overlap, f1_overlap, acc_sel, prec_sel, rec_sel, f1_sel

    def validation_epoch_end(self, val_outputs):
        """ Aggregates scores. """
        self.log('val_overlap_precision', np.mean([x[0] for x in val_outputs]))
        self.log('val_overlap_recall', np.mean([x[1] for x in val_outputs]))
        self.log('val_overlap_f1', np.mean([x[2] for x in val_outputs]))
        self.log('val_selection_accuracy', np.mean([x[3] for x in val_outputs]))
        self.log('val_selection_precision', np.mean([x[4] for x in val_outputs]))
        self.log('val_selection_recall', np.mean([x[5] for x in val_outputs]))
        self.log('val_selection_f1', np.mean([x[6] for x in val_outputs]))


def main(args):
    seed_everything(args.seed)
    data_module = FilteringDataModule(args)
    model = FilterBert(args)
    model_checkpoint = ModelCheckpoint(
        dirpath=args.model_dir,
        filename=args.model + '-{epoch}-{' + args.monitor + ':.2f}',
        monitor=args.monitor,
        save_top_k=1,
        mode='max',
    )
    early_stopping = EarlyStopping(args.monitor, mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [model_checkpoint, early_stopping, lr_monitor]
    if isinstance(args.gpus, int):
        gpu_monitor = GPUStatsMonitor(intra_step_time=True, inter_step_time=True)
        callbacks.append(gpu_monitor)
    logger = loggers.TensorBoardLogger(
        save_dir=args.model_dir,
        name='',
    )
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extractive summarization.')

    # select model and add model args
    parser.add_argument('--model', default='bert', choices=['bert'], help='Model name')
    temp_args, _ = parser.parse_known_args()
    parser = FilterBert.add_model_specific_args(parser)

    # data args
    parser.add_argument('--data_dir', default='data_filterbert', help='Path to data directory')
    parser.add_argument('--num_workers', type=int, default=0, help='Num workers for data loading')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')

    # trainer args
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--model_dir', default='models', help='Path to model directory')
    parser.add_argument('--seed', default=1, help='Random seed')
    main(parser.parse_args())
