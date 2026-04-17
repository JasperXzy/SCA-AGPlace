import logging

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import pytorch_lightning as pl
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError("PyTorch Lightning is required for SCADataModule.") from exc

from datasets.datasets_ws_kitti360 import (
    KITTI360BaseDataset,
    KITTI360TripletsDataset,
    kitti360_collate_fn,
)
from datasets.datasets_ws_nuscenes import (
    NuScenesBaseDataset,
    NuScenesTripletsDataset,
    nuscenes_collate_fn,
)


class _ValidationSignalDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return torch.tensor(0)


class SCADataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.triplets_ds = None
        self.test_ds = None
        self.collate_fn = None

    def setup(self, stage=None):
        if self.triplets_ds is not None and self.test_ds is not None:
            return

        if self.cfg.dataset == "kitti360":
            self.triplets_ds = KITTI360TripletsDataset(
                self.cfg,
                self.cfg.datasets_folder,
                self.cfg.dataset_name,
                "train",
                self.cfg.negs_num_per_query,
            )
            self.test_ds = KITTI360BaseDataset(
                self.cfg, self.cfg.datasets_folder, self.cfg.dataset_name, "test"
            )
            self.collate_fn = kitti360_collate_fn
        elif self.cfg.dataset == "nuscenes":
            self.triplets_ds = NuScenesTripletsDataset(
                self.cfg,
                self.cfg.datasets_folder,
                self.cfg.dataset_name,
                "train",
                self.cfg.negs_num_per_query,
            )
            self.test_ds = NuScenesBaseDataset(
                self.cfg, self.cfg.datasets_folder, self.cfg.dataset_name, "test"
            )
            self.collate_fn = nuscenes_collate_fn
        else:
            raise ValueError(f"Unsupported dataset: {self.cfg.dataset}")

        self.triplets_ds.triplets_global_indexes = torch.zeros(
            (max(1, self.cfg.cache_refresh_rate), self.cfg.negs_num_per_query + 2),
            dtype=torch.long,
        )
        logging.info("Train query set: %s", self.triplets_ds)
        logging.info("Test set: %s", self.test_ds)

    def train_dataloader(self):
        if self.triplets_ds is None:
            raise RuntimeError("SCADataModule.setup() must run before train_dataloader().")
        self.triplets_ds.is_inference = False
        return DataLoader(
            self.triplets_ds,
            batch_size=self.cfg.train_batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=str(self.cfg.device).startswith("cuda"),
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(_ValidationSignalDataset(), batch_size=1)
