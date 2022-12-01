# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Preprocess source and target for BERT format. """

from transformers import BertTokenizer

from data_utils import SPLIT_SYMBOL


class ReferenceFormat:

    def __init__(self, annotation_schema, max_src_tokens=512, max_tgt_tokens=512):
        self.max_src_tokens = max_src_tokens
        self.max_tgt_tokens = max_tgt_tokens
        self.bos_token = '[unused0]'
        self.eos_token = '[unused1]'
        self.split_token = '[unused2]'
        special_tokens = annotation_schema.get_special_tokens()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', additional_special_tokens=special_tokens)

    def preprocess_source(self, src_sents):
        """ format each source sentence into: [CLS] sent [SEP] """
        src_sents_tokens = []
        for sent in src_sents:
            sent_tokens = self.tokenizer.tokenize(sent)
            if 0 < self.max_src_tokens < len(sent_tokens) + 2:
                sent_tokens = sent_tokens[:self.max_src_tokens - 2]  # truncate sentence if too long
            sent_tokens = [self.tokenizer.cls_token] + sent_tokens + [self.tokenizer.sep_token]
            src_sents_tokens.append(sent_tokens)

        # convert tokens to ids
        src_sents_ids = [self.tokenizer.convert_tokens_to_ids(sent_tokens) for sent_tokens in src_sents_tokens]
        return src_sents_ids

    def preprocess_target(self, tgt_paragraphs):
        """
        format each target paragraph into: [unused0] s1 [unused2] s2 [unused2] s3 [unused1]
        [unused0] = beginning of output, [unused1] = end of output, [unused2] = sentence separator
        """
        tgt_sents_tokens = []
        tgt_segment_ids = []
        for paragraph in tgt_paragraphs:
            paragraph_tokens = [self.bos_token]
            paragraph_segments = [0]
            sents = paragraph.split(SPLIT_SYMBOL)
            for i, sent in enumerate(sents):
                sent_tokens = self.tokenizer.tokenize(sent)
                # discard subsequent sentences when they exceed the length limit
                if 0 < self.max_tgt_tokens < len(paragraph_tokens) + len(sent_tokens) + 2:
                    if len(paragraph_tokens) == 1:
                        # truncate the first sentence as it is already too long
                        paragraph_tokens.extend(sent_tokens[:self.max_tgt_tokens - 2])
                        paragraph_segments.extend([0] * (self.max_tgt_tokens - 2))
                    break
                if len(paragraph_tokens) > 1:
                    # not first sentence, insert split symbol
                    paragraph_tokens.append(self.split_token)
                    paragraph_segments.append(paragraph_segments[-1])
                paragraph_tokens.extend(sent_tokens)
                sent_segment_id = i % 2
                paragraph_segments.extend([sent_segment_id] * len(sent_tokens))
            paragraph_tokens.append(self.eos_token)
            paragraph_segments.append(paragraph_segments[-1])
            tgt_sents_tokens.append(paragraph_tokens)
            tgt_segment_ids.append(paragraph_segments)

        # convert tokens to ids
        tgt_sents_ids = [self.tokenizer.convert_tokens_to_ids(sent_tokens) for sent_tokens in tgt_sents_tokens]
        return tgt_sents_ids, tgt_segment_ids
