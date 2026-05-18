from pathlib import Path
import tarfile

import numpy as np
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, df, feature_cols, signals_path,
                 idx_col="norm_idx", label_col="chagas_label"):
        self.df = df.reset_index(drop=True)
        self.feature_cols = list(feature_cols)
        self.signals_path = str(signals_path)
        self.idx_col = idx_col
        self.label_col = label_col
        self._signals = None

    def _path(self):
        p = Path(self.signals_path)
        if p.suffix == ".tar":
            out = p.with_suffix("")
            if not out.exists():
                with tarfile.open(p, "r") as t:
                    t.extractall(p.parent)
            return out
        return p

    def _open(self):
        if self._signals is None:
            self._signals = np.load(str(self._path()), mmap_mode="r")
        return self._signals

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        signals = self._open()
        signal = np.array(signals[int(row[self.idx_col])]).T.astype(np.float32)
        tab = row[self.feature_cols].to_numpy(dtype=np.float32)
        tab = np.nan_to_num(tab, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            torch.from_numpy(np.ascontiguousarray(signal)),
            torch.from_numpy(tab),
            torch.tensor([float(row[self.label_col])], dtype=torch.float32),
        )
