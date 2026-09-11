import numpy as np
import pandas as pd

from cistempoforge import STRUCTURE_COLUMNS, fit_transform_structure


def test_structure_scaler_is_fit_on_train_only():
    values = np.arange(24, dtype=float).reshape(3, 8)
    values[1, 0] = np.nan
    frame = pd.DataFrame(values, columns=STRUCTURE_COLUMNS)
    frame["split"] = ["train", "train", "test"]
    normalized, masks, scaler = fit_transform_structure(frame)
    assert normalized.shape == masks.shape == (3, 8)
    assert np.isfinite(normalized).all()
    assert scaler["n_train"] == 2
    assert scaler["means"] != values[2].tolist()
