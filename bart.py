# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" BART seq2seq model. """

from argparse import ArgumentParser

from pytorch_lightning.core import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from transformers import BartTokenizer, BartForConditionalGeneration, AddedToken

from data_schema import SchemaFactory
from dataloader import BartBatch
from rouge import RougeScorer, RougeAggregator


class BartSummarizer(LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.annotation_schema = SchemaFactory.get_schema(args.dataset)
        special_tokens = self.annotation_schema.get_special_text_tokens()
        special_tokens = [AddedToken(t) for t in special_tokens]
        self.tokenizer = BartTokenizer.from_pretrained(
            args.model_name_or_path, additional_special_tokens=special_tokens,
        )
        self.model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
        self.model.resize_token_embeddings(len(self.tokenizer))  # extend embedding matrices for special tokens

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dataset', default='fomc', help='Dataset name')
        parser.add_argument('--model_name_or_path', default='facebook/bart-large', help='Path to pretrained model')

        # optimization args
        parser.add_argument('--max_lr_enc', type=float, default=1e-5, help='Maximum learning rate for encoder params')
        parser.add_argument('--max_lr_dec', type=float, default=1e-5, help='Maximum learning rate for decoder params')
        parser.add_argument('--max_lr_lm_head', type=float, default=1e-6, help='Maximum learning rate for LM head')
        parser.add_argument('--warmup', type=float, default=0.1, help='Fraction of training spent in warmup')
        parser.add_argument('--lr_anneal', default='linear', choices=['linear', 'cos'],
                            help='Annealing of learning rate')
        parser.add_argument('--div_warmup', type=float, default=100,
                            help='Divide max learning rate by this for initial learning rate')
        parser.add_argument('--div_final', type=float, default=100,
                            help='Divide max learning rate by this for final learning rate')

        # generation args
        parser.add_argument('--max_length', type=int, default=500, help='Max generation length')
        parser.add_argument('--min_length', type=int, default=100, help='Min generation length')
        parser.add_argument('--length_penalty', type=float, default=0.95, help='Alpha for length penalty')
        parser.add_argument('--val_log_summaries', type=int, default=5, help='Number of logged summaries in validation')

        return parser

    def forward(self, batch: BartBatch):
        labels = batch.tgt.clone().detach()
        labels = labels[:, 1:].contiguous()  # remove decoder start token (EOS token for BART)
        labels[labels == self.tokenizer.pad_token_id] = -100  # set padded tokens to -100 to ignore in LM loss
        decoder_outputs = self.model(
            input_ids=batch.src,
            attention_mask=batch.mask_src,
            decoder_input_ids=batch.tgt[:, :-1],
            decoder_attention_mask=batch.mask_tgt[:, :-1],
            labels=labels,
            return_dict=True,
        )
        return decoder_outputs

    def configure_optimizers(self):
        # remove shared embedding matrix (used in LM head and can't have the same params in different param groups)
        encoder_params = (p for n, p in self.model.model.encoder.named_parameters() if n != 'embed_tokens.weight')
        decoder_params = (p for n, p in self.model.model.decoder.named_parameters() if n != 'embed_tokens.weight')
        optimizer = Adam([
            {'params': encoder_params},
            {'params': decoder_params},
            {'params': self.model.lm_head.parameters()},
        ])
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=[self.hparams.max_lr_enc, self.hparams.max_lr_dec, self.hparams.max_lr_lm_head],
            total_steps=self.hparams.max_steps,
            pct_start=self.hparams.warmup,
            anneal_strategy=self.hparams.lr_anneal,
            cycle_momentum=False,
            div_factor=self.hparams.div_warmup,
            final_div_factor=self.hparams.div_final,
            last_epoch=-1,
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    def training_step(self, batch, batch_idx):
        seq2seq_lm_output = self(batch)
        loss = seq2seq_lm_output.loss
        self.log('train_loss', loss)
        return loss

    def decode(self, output_ids):
        """ Decode BPE tokens into text. """
        text = self.tokenizer.decode(output_ids, spaces_between_special_tokens=False)
        stst_start = self.annotation_schema.mapping['standardized sentence']['text_start']
        stst_end = self.annotation_schema.mapping['standardized sentence']['text_end']
        text = text.replace(f'{stst_end} {stst_start}', f'{stst_end}<sent>{stst_start}')
        text = text.replace(self.tokenizer.bos_token, '')
        text = text.replace(self.tokenizer.eos_token, '')
        text = ' '.join(text.split())
        return text

    def compute_rouge(self, beam_output, targets):
        scorer = RougeScorer()
        scores = []
        for output, target in zip(beam_output, targets):
            candidate = self.decode(output)
            reference = self.decode(target)
            scores.append(scorer.compute_rouge_score(candidate, reference, sentence_sep='<sent>'))
        return scores

    def _shared_eval(self, batch, batch_idx, prefix):
        # decode with beam search
        beam_output = self.model.generate(
            input_ids=batch.src,
            attention_mask=batch.mask_src,
            max_length=self.hparams.max_length,
            min_length=self.hparams.min_length,
            no_repeat_ngram_size=3,
            length_penalty=self.hparams.length_penalty,
            num_beams=5,
        )

        # if validating, log first N summaries
        if prefix == 'val' and hasattr(self.hparams, 'val_log_summaries') and batch_idx < self.hparams.val_log_summaries:
            self.logger.experiment.add_text(f'val_summary_{batch_idx}', self.decode(beam_output[0]), self.global_step)

        # if testing, write results out
        if prefix == 'test':
            self.write_output(beam_output, batch.tgt)

        # in validation, compute loss
        if prefix == 'val':
            labels = batch.tgt.clone().detach()
            labels = labels[:, 1:].contiguous()  # remove decoder start token (EOS token for BART)
            labels[labels == self.tokenizer.pad_token_id] = -100  # set padded tokens to -100 to ignore in LM loss
            decoder_outputs = self.model(
                input_ids=batch.src,
                attention_mask=batch.mask_src,
                decoder_input_ids=batch.tgt[:, :-1],
                decoder_attention_mask=batch.mask_tgt[:, :-1],
                labels=labels,
                return_dict=True,
            )
            val_loss = decoder_outputs.loss.item()
            self.log('val_loss', val_loss)

        # compute ROUGE scores
        return self.compute_rouge(beam_output, batch.tgt)

    def write_output(self, beam_output, targets):
        import os
        for output, target in zip(beam_output, targets):
            candidate = self.decode(output)
            reference = self.decode(target)
            with open(os.path.join(self.output_dir, 'candidates.txt'), 'a') as f:
                f.write(candidate + '\n')
            with open(os.path.join(self.output_dir, 'references.txt'), 'a') as f:
                f.write(reference + '\n')

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, 'test')

    def validation_epoch_end(self, val_outputs):
        """ Aggregates ROUGE scores. """
        # flatten list of lists
        scores = []
        for val_output in val_outputs:
            scores.extend(val_output)

        # aggregate ROUGE scores
        aggregator = RougeAggregator()
        for score in scores:
            aggregator.add_scores(score)
        rouge1, rouge2, rougeL = tuple(map(lambda x: x * 100, aggregator.get_rouge_scores()))
        self.log('val_rouge1', rouge1)
        self.log('val_rouge2', rouge2)
        self.log('val_rougeL', rougeL)
        self.log('val_rouge', (rouge1 + rouge2 + rougeL) / 3)
