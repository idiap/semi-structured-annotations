# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Data utils for preprocessing and encoding. """

from transformers import BertTokenizer

from data_schema import SchemaFactory

SPLIT_SYMBOL = '<sent>'


class BertData:

    def __init__(self, dataset, tgt_format='presumm', max_src_tokens=0):
        self.tgt_format = tgt_format
        self.max_src_tokens = max_src_tokens
        self.start_token = '[unused0]'
        self.end_token = '[unused1]'
        self.sent_split_token = '[unused2]'
        self.annotation_schema = SchemaFactory.get_schema(dataset)
        special_tokens = self.annotation_schema.get_special_tokens()
        special_tokens.extend([self.start_token, self.end_token, self.sent_split_token])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', additional_special_tokens=special_tokens)

    def encode(self, example, sent_sep='<q>'):
        tgt = example['tgt']
        # bring text format back to token format
        for annotation in self.annotation_schema.mapping.values():
            tgt = tgt.replace(annotation['text_start'], annotation['subtoken_start'])
            tgt = tgt.replace(annotation['text_end'], annotation['subtoken_end'])
        tgt_sents = tgt.split(sent_sep)
        src_ids, seg_ids, tgt_ids = self.preprocess(example['src_sents'], tgt_sents)
        return {
            'src': src_ids,
            'src_segs': seg_ids,
            'tgt': tgt_ids,
            'name': example['name'],
        }

    def preprocess(self, src_sents, tgt_sents):
        # bring src into Bert format: [CLS] s1 [SEP] [CLS] s2 [SEP]
        src_subtokens = []
        for sent in src_sents:
            sent_subtokens = self.tokenizer.tokenize(sent)
            sent_subtokens = [self.tokenizer.cls_token] + sent_subtokens + [self.tokenizer.sep_token]
            if 0 < self.max_src_tokens < len(src_subtokens) + len(sent_subtokens):
                if len(src_subtokens) == 0:
                    # truncate the first sentence in case it is already too long
                    src_subtokens = sent_subtokens[:self.max_src_tokens - 1] + [self.tokenizer.sep_token]
                break
            src_subtokens.extend(sent_subtokens)
        src_subtoken_ids = self.tokenizer.convert_tokens_to_ids(src_subtokens)

        # compute segment ids (alternating 0 and 1 for each sentence in src)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_ids) if t == self.tokenizer.sep_token_id]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        if self.tgt_format == 'bert':
            # bring tgt into Bert format: [CLS] s1 [SEP] [CLS] s2 [SEP] [unused1]
            tgt_txt = ' {} {} '.format(self.tokenizer.sep_token, self.tokenizer.cls_token).join(tgt_sents)
            tgt_subtokens = self.tokenizer.tokenize(tgt_txt)
            tgt_subtokens = [self.tokenizer.cls_token] + tgt_subtokens + [self.tokenizer.sep_token] + [self.end_token]
        elif self.tgt_format == 'presumm':
            tgt_txt = ' {} '.format(self.sent_split_token).join(tgt_sents)
            tgt_subtokens = self.tokenizer.tokenize(tgt_txt)
            tgt_subtokens = [self.start_token] + tgt_subtokens + [self.end_token]
        else:
            raise ValueError('Unknown tgt format: %s' % self.tgt_format)
        tgt_subtoken_ids = self.tokenizer.convert_tokens_to_ids(tgt_subtokens)

        return src_subtoken_ids, segments_ids, tgt_subtoken_ids


class BartData:

    def __init__(self, tokenizer, separate_sentences=False, max_src_tokens=1024, max_tgt_tokens=512):
        self.tokenizer = tokenizer
        self.separate_sentences = separate_sentences
        self.max_src_tokens = max_src_tokens
        self.max_tgt_tokens = max_tgt_tokens

    def encode(self, example):
        if self.separate_sentences:
            src = [ids for sent in example['src_sents'] for ids in self.tokenizer.encode(sent)]
            if len(src) > self.max_src_tokens:
                src = src[:self.max_src_tokens - 1] + [self.tokenizer.eos_token_id]
        else:
            src = self.tokenizer.encode(
                ' '.join(example['src_sents']), max_length=self.max_src_tokens, truncation=True
            )
        tgt = self.tokenizer.encode(example['tgt'], max_length=self.max_tgt_tokens - 1, truncation=True)
        tgt = [self.tokenizer.eos_token_id] + tgt  # prepend EOS token (BART format)
        return {
            'src': src,
            'tgt': tgt,
            'name': example['name'],
            'tgt_i': example['tgt_i'],
        }
