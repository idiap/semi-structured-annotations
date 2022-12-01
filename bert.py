# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Seq2seq model with BERT encoder, Transformer decoder. """

from argparse import ArgumentParser
from copy import deepcopy

from pytorch_lightning.core import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from transformers import BertModel, BertLMHeadModel, BertTokenizer, BertConfig

from data_schema import SchemaFactory
from dataloader import SummarizationBatch
from rouge import RougeScorer, RougeAggregator


class TransformerDecoder(BertLMHeadModel):

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        inputs = super(TransformerDecoder, self).prepare_inputs_for_generation(
            input_ids=input_ids,
            past=past,
            attention_mask=attention_mask,
            **model_kwargs,
        )
        inputs['encoder_hidden_states'] = model_kwargs['encoder_hidden_states']
        return inputs


class BertSummarizer(LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        model_name_or_path = self.hparams.pretrained_dir or 'bert-base-uncased'
        self.annotation_schema = SchemaFactory.get_schema(args.dataset)
        special_tokens = self.annotation_schema.get_special_tokens()
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path, additional_special_tokens=special_tokens)
        self.encoder = BertModel.from_pretrained(model_name_or_path)
        decoder_config = BertConfig.from_pretrained(model_name_or_path)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        decoder_config.num_hidden_layers = self.hparams.decoder_layers
        decoder_config.hidden_size = self.hparams.decoder_hidden_dim
        decoder_config.num_attention_heads = self.hparams.decoder_heads
        decoder_config.intermediate_size = self.hparams.decoder_intermediate_size
        decoder_config.hidden_dropout_prob = self.hparams.decoder_hidden_dropout
        decoder_config.attention_probs_dropout_prob = self.hparams.decoder_attention_dropout
        decoder_config.max_position_embeddings = self.hparams.max_pos
        self.decoder = TransformerDecoder(decoder_config)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dataset', default='fomc', help='Dataset name')
        parser.add_argument('--pretrained_dir', default=None, help='Path to pretrained model')

        # decoder args
        parser.add_argument('--decoder_layers', type=int, default=12)
        parser.add_argument('--decoder_hidden_dim', type=int, default=768)
        parser.add_argument('--decoder_heads', type=int, default=12)
        parser.add_argument('--decoder_intermediate_size', type=int, default=3072)
        parser.add_argument('--decoder_hidden_dropout', type=float, default=0.1)
        parser.add_argument('--decoder_attention_dropout', type=float, default=0.1)
        parser.add_argument('--max_pos', type=int, default=512)

        # optimization args
        parser.add_argument('--max_lr_pre', type=float, default=1e-5, help='Maximum learning rate for pretrained parameters.')
        parser.add_argument('--max_lr_rand', type=float, default=1e-4, help='Maximum learning rate for randomly initialized parameters.')
        parser.add_argument('--warmup', type=float, default=0.1, help='Fraction of training spent in warmup')
        parser.add_argument('--lr_anneal', default='linear', choices=['linear', 'cos'], help='Annealing of learning rate')
        parser.add_argument('--div_warmup', type=float, default=100, help='Divide max learning rate by this for initial learning rate')
        parser.add_argument('--div_final', type=float, default=100, help='Divide max learning rate by this for final learning rate')

        # generation args
        parser.add_argument('--max_length', type=int, default=500, help='Max generation length')
        parser.add_argument('--min_length', type=int, default=100, help='Min generation length')
        parser.add_argument('--length_penalty', type=float, default=0.95, help='Alpha for length penalty')
        parser.add_argument('--val_log_summaries', type=int, default=5, help='Number of logged summaries in validation')

        return parser

    def run_encoder(self, batch: SummarizationBatch):
        encoder_outputs = self.encoder(
            input_ids=batch.src,
            attention_mask=batch.mask_src,
            token_type_ids=batch.segs,
            return_dict=True,
        )
        return encoder_outputs.last_hidden_state

    def forward(self, batch: SummarizationBatch):
        encoder_hidden_states = self.run_encoder(batch)
        labels = deepcopy(batch.tgt)
        labels[labels == 0] = -100  # set padded tokens to -100 to ignore in LM loss
        decoder_outputs = self.decoder(
            input_ids=batch.tgt,
            attention_mask=batch.mask_tgt,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=batch.mask_src,
            labels=labels,
            return_dict=True,
        )
        return decoder_outputs

    def configure_optimizers(self):
        optimizer = Adam([
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()},
        ])
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=[self.hparams.max_lr_pre, self.hparams.max_lr_rand],
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
        causal_lm_output = self(batch)
        loss = causal_lm_output.loss
        self.log('train_loss', loss)
        return loss

    def decode(self, output_ids):
        """ Decode BPE tokens into text. """
        text = self.tokenizer.decode(output_ids, clean_up_tokenization_spaces=False)
        stst_start = self.annotation_schema.mapping['standardized sentence']['subtoken_start']
        stst_end = self.annotation_schema.mapping['standardized sentence']['subtoken_end']
        text = text.replace(f'{stst_end} [unused2] {stst_start}', f'{stst_end}<sent>{stst_start}')
        text = text.replace('[unused0]', '')
        text = text.replace('[unused1]', '')
        text = text.replace('[unused2]', '')
        text = text.replace('[PAD]', '')
        text = text.replace('[CLS]', '')
        text = text.replace('[SEP]', '')

        # replace annotation BPE tokens with text version
        for annotation in self.annotation_schema.mapping.values():
            text = text.replace(annotation['subtoken_start'], annotation['text_start'])
            text = text.replace(annotation['subtoken_end'], annotation['text_end'])

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
        # encode source document
        encoder_hidden_states = self.run_encoder(batch)
        # `encoder_hidden_states` has shape: [batch_size, src_length, hidden_dim]

        # decode with beam search
        beam_output = self.decoder.generate(
            max_length=self.hparams.max_length,
            min_length=self.hparams.min_length,
            early_stopping=True,
            no_repeat_ngram_size=3,
            bos_token_id=1,
            pad_token_id=0,
            eos_token_id=2,
            length_penalty=self.hparams.length_penalty,
            num_beams=5,
            use_cache=True,
            encoder_hidden_states=encoder_hidden_states,
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
            labels[labels == 0] = -100  # set padded tokens to -100 to ignore in LM loss
            decoder_outputs = self.decoder(
                input_ids=batch.tgt,
                attention_mask=batch.mask_tgt,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=batch.mask_src,
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
