# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Helper class for annotation schema. """

from abc import ABC


class AnnotationSchema(ABC):

    def __init__(self):
        self.mapping = {
            'standardized sentence': {
                'text_start': '[STD_SENTENCE_START]',
                'text_end': '[STD_SENTENCE_END]',
                'subtoken_start': '[unused10]',
                'subtoken_end': '[unused11]',
            },
            'attribution': {
                'text_start': '[ATTRIBUTION_START]',
                'text_end': '[ATTRIBUTION_END]',
                'subtoken_start': '[unused12]',
                'subtoken_end': '[unused13]',
            },
            'actor': {
                'text_start': '[ACTOR_START]',
                'text_end': '[ACTOR_END]',
                'subtoken_start': '[unused14]',
                'subtoken_end': '[unused15]',
            },
            'motive': {
                'text_start': '[MOTIVE_START]',
                'text_end': '[MOTIVE_END]',
                'subtoken_start': '[unused16]',
                'subtoken_end': '[unused17]',
            },
            'scope': {
                'text_start': '[SCOPE_START]',
                'text_end': '[SCOPE_END]',
                'subtoken_start': '[unused18]',
                'subtoken_end': '[unused19]',
            },
            'evidence': {
                'text_start': '[EVIDENCE_START]',
                'text_end': '[EVIDENCE_END]',
                'subtoken_start': '[unused20]',
                'subtoken_end': '[unused21]',
            },
            'reference': {
                'text_start': '[REFERENCE_START]',
                'text_end': '[REFERENCE_END]',
                'subtoken_start': '[unused22]',
                'subtoken_end': '[unused23]',
            },
            'act': {
                'text_start': '[ACT_START]',
                'text_end': '[ACT_END]',
                'subtoken_start': '[unused24]',
                'subtoken_end': '[unused25]',
            },
        }
        self.false_id = None
        self.duplicate_id = None

    def from_entity_id(self, entity_id):
        for v in self.mapping.values():
            if v['entity_id'] == entity_id:
                return v
        raise KeyError('Unknown entity id: %s' % entity_id)

    def from_label_id(self, label_id):
        for v in self.mapping.values():
            if v['label_id'] == label_id:
                return v
        raise KeyError('Unknown label id: %s' % label_id)

    def get_special_tokens(self):
        """ Returns a list of special tokens. """
        subtokens_start = [v['subtoken_start'] for v in self.mapping.values()]
        subtokens_end = [v['subtoken_end'] for v in self.mapping.values()]
        return subtokens_start + subtokens_end

    def get_special_text_tokens(self):
        """ Returns a list of special text tokens. """
        text_starts = [v['text_start'] for v in self.mapping.values()]
        text_ends = [v['text_end'] for v in self.mapping.values()]
        return text_starts + text_ends

    def is_false_annotation(self, annotation):
        """ Returns true if the annotation for this article is not relevant (entity is false). """
        for entity in annotation['entities']:
            eid = entity['classId']
            if eid == self.false_id:
                assert len(annotation['entities']) == 1 or all(e['classId'] == self.false_id for e in annotation['entities'])
                return True


class FOMCSchema(AnnotationSchema):

    def __init__(self):
        super().__init__()
        self.mapping['standardized sentence']['entity_id'] = 'e_24'
        self.mapping['standardized sentence']['label_id'] = 'f_36'
        self.mapping['attribution']['entity_id'] = 'e_25'
        self.mapping['attribution']['label_id'] = 'f_40'
        self.mapping['actor']['entity_id'] = 'e_26'
        self.mapping['actor']['label_id'] = 'f_42'
        self.mapping['motive']['entity_id'] = 'e_29'
        self.mapping['motive']['label_id'] = 'f_47'
        self.mapping['scope']['entity_id'] = 'e_31'
        self.mapping['scope']['label_id'] = 'f_46'
        self.mapping['evidence']['entity_id'] = 'e_33'
        self.mapping['evidence']['label_id'] = 'f_44'
        self.mapping['reference']['entity_id'] = 'e_35'
        self.mapping['reference']['label_id'] = 'f_37'
        self.mapping['act']['entity_id'] = 'e_38'
        self.mapping['act']['label_id'] = 'f_43'
        self.false_id = 'e_41'
        self.duplicate_id = 'e_48'


class SchemaFactory:

    registry = {
        'fomc': FOMCSchema,
    }

    @staticmethod
    def get_schema(name) -> AnnotationSchema:
        if name not in SchemaFactory.registry:
            raise ValueError(f'Unknown annotation schema: {name}')
        return SchemaFactory.registry[name]()
