# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Seq2seq model with randomly initialized Transformer. """

from argparse import ArgumentParser

from transformers import BertModel, BertConfig

from bert import BertSummarizer


class TransformerSummarizer(BertSummarizer):
    """ Randomly initialized Transformer. Uses BertModel architecture and tokenizer. """

    def __init__(self, args):
        super().__init__(args)
        model_name_or_path = self.hparams.pretrained_dir or 'bert-base-uncased'

        # encoder
        encoder_config = BertConfig.from_pretrained(model_name_or_path)
        encoder_config.num_hidden_layers = self.hparams.encoder_layers
        encoder_config.hidden_size = self.hparams.encoder_hidden_dim
        encoder_config.num_attention_heads = self.hparams.encoder_heads
        encoder_config.intermediate_size = self.hparams.encoder_intermediate_size
        encoder_config.hidden_dropout_prob = self.hparams.encoder_hidden_dropout
        encoder_config.attention_probs_dropout_prob = self.hparams.encoder_attention_dropout
        encoder_config.max_position_embeddings = self.hparams.max_pos
        self.encoder = BertModel(encoder_config, add_pooling_layer=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = BertSummarizer.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--encoder_layers', type=int, default=12)
        parser.add_argument('--encoder_hidden_dim', type=int, default=768)
        parser.add_argument('--encoder_heads', type=int, default=12)
        parser.add_argument('--encoder_intermediate_size', type=int, default=3072)
        parser.add_argument('--encoder_hidden_dropout', type=float, default=0.1)
        parser.add_argument('--encoder_attention_dropout', type=float, default=0.1)
        return parser
