from personalized_nlp.datasets.wiki.base import WikiDataModule
from personalized_nlp.settings import STORAGE_DIR, TOXICITY_URL
from typing import List
from pathlib import Path


class ToxicityDataModule(WikiDataModule):
    @property
    def class_dims(self) -> List[int]:
        return [2]

    @property
    def annotation_columns(self) -> List[str]:
        return ["toxicity"]

    @property
    def embeddings_path(self) -> Path:
        return (
            self.data_dir / f"embeddings/rev_id_to_emb_{self.embeddings_type}_toxic.p"
        )
