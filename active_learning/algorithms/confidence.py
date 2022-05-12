import pandas as pd
import numpy as np


def confidence_selector(
    texts: pd.DataFrame,
    amount: int,
    annotated: pd.DataFrame,
    not_annotated: pd.DataFrame,
    confidences: np.ndarray,
):
    if confidences is not None:
        confidences = confidences.max(axis=1)
        sorted_index = np.argsort(confidences)

        return not_annotated.iloc[sorted_index[:amount]]

    return not_annotated.sample(n=amount)
