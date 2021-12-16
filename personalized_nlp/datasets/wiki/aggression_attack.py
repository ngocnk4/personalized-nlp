import pandas as pd
import pickle
import torch
import os

from personalized_nlp.datasets.wiki.base import WikiDataModule
from personalized_nlp.settings import STORAGE_DIR, AGGRESSION_URL
from personalized_nlp.utils.biases import get_annotator_biases


class AggressionAttackDataModule(WikiDataModule):
    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.annotation_column = ['aggression', 'attack']
        self.word_stats_annotation_column = 'aggression'

        self.embeddings_path = STORAGE_DIR / \
            f'wiki_data/embeddings/rev_id_to_emb_{self.embeddings_type}_aggression.p'


    @property
    def class_dims(self):
        return [2, 2]

    def prepare_data(self) -> None:
        self.data = pd.read_csv(
            self.data_dir / (self.annotation_column[0] + '_annotated_comments.tsv'), sep='\t')
        self.data = self._remap_column_names(self.data)
        self.data['text'] = self.data['text'].str.replace(
            'NEWLINE_TOKEN', '  ')

        self.annotators = pd.read_csv(
            self.data_dir / (self.annotation_column[0] + '_worker_demographics.tsv'), sep='\t')
        self.annotators = self._remap_column_names(self.annotators)

        aggression_annotations = pd.read_csv(
            self.data_dir / (self.annotation_column[0] + '_annotations.tsv'), sep='\t')
        attack_annotations = pd.read_csv(
            self.data_dir / (self.annotation_column[1] + '_annotations.tsv'), sep='\t')        
        
        self.annotations = aggression_annotations.merge(attack_annotations)
        
        self.annotations = self._remap_column_names(self.annotations)

        self._assign_splits()
        
        personal_df = self.annotations_with_data.loc[self.annotations_with_data.split == 'past']
        self.compute_annotator_biases(personal_df)