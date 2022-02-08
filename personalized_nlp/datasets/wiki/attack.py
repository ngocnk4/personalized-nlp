import os

from typing import List

import pandas as pd

from personalized_nlp.datasets.wiki.base import WikiDataModule
from personalized_nlp.settings import STORAGE_DIR, ATTACK_URL
from personalized_nlp.utils.data_splitting import split_texts, split_texts_by_original


class AttackDataModule(WikiDataModule):
    def __init__(
            self,
            split_sizes: List[float] = [0.55, 0.15, 0.15, 0.15],
            normalize=False,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = STORAGE_DIR / 'wiki_data'
        self.split_sizes = split_sizes
        self.annotation_column = ['attack']
        self.text_column = 'comment'

        self.word_stats_annotation_column = 'attack'
        self.embeddings_path = STORAGE_DIR / \
                               f'wiki_data/embeddings/text_id_to_emb_{self.embeddings_type}.p'

        self.train_split_names = ['present', 'past']
        self.val_split_names = ['future1']
        self.test_split_names = ['future2']

        self.normalize = normalize

        os.makedirs(self.data_dir / 'embeddings', exist_ok=True)

    @property
    def class_dims(self):
        return [2]

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def prepare_data(self) -> None:
        self.data = pd.read_csv(
            self.data_dir / 'attack_annotated_comments.tsv', sep='\t')
        self.data['text'] = self.data['comment']
        self.data['text_id'] = self.data['rev_id']

        self.annotations = pd.read_csv(
            self.data_dir / 'attack_annotations.tsv', sep='\t').dropna()
        self.annotations['annotator_id'] = self.annotations['worker_id']
        self.annotations['text_id'] = self.annotations['rev_id']

        if self.normalize:
            self.normalize_labels()

        self._assign_splits()
        print(self.data.split.value_counts())

        personal_df = self.annotations_with_data.loc[self.annotations_with_data.split == 'past']
        self.compute_annotator_biases(personal_df)

    def normalize_labels(self):
        annotation_column = self.annotation_column
        df = self.annotations

        mins = df.loc[:, annotation_column].values.min(axis=0)
        df.loc[:, annotation_column] = (df.loc[:, annotation_column] - mins)

        maxes = df.loc[:, annotation_column].values.max(axis=0)
        df.loc[:, annotation_column] = df.loc[:, annotation_column] / maxes

    def _assign_splits(self):
        sizes = [0.55, 0.15, 0.15, 0.15]
        # self.data = split_texts(self.data, sizes)
        self.data = split_texts_by_original(self.data, sizes)

