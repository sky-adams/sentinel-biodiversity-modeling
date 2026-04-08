from pathlib import Path
from dataclasses import dataclass

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


@dataclass
class SplitConfig:
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15
    nan_threshold: float = 0.10


class SentinelBIITileDataset(Dataset):
    def __init__(self, tif_path, patch_size=64, stride=64, nan_threshold=0.10):
        self.tif_path = Path(tif_path)
        self.patch_size = patch_size
        self.stride = stride
        self.nan_threshold = nan_threshold

        with rasterio.open(self.tif_path) as src:
            self.width = src.width
            self.height = src.height
            self.count = src.count
            self.crs = src.crs
            self.transform = src.transform
            self.nodata = src.nodata

        self.windows = []
        self.skip_stats = {"total": 0, "kept": 0, "skipped": 0}
        self._build_windows()

    def _read_window_array(self, src, row, col):
        window = rasterio.windows.Window(col, row, self.patch_size, self.patch_size)
        arr = src.read(window=window).astype(np.float32)
        if self.nodata is not None:
            arr = np.where(arr == self.nodata, np.nan, arr)
        return arr

    def _build_windows(self):
        with rasterio.open(self.tif_path) as src:
            for row in range(0, self.height - self.patch_size + 1, self.stride):
                for col in range(0, self.width - self.patch_size + 1, self.stride):
                    self.skip_stats["total"] += 1
                    arr = self._read_window_array(src, row, col)
                    nan_frac = 1.0 - np.isfinite(arr).mean()
                    if nan_frac > self.nan_threshold:
                        self.skip_stats["skipped"] += 1
                        continue
                    self.windows.append((row, col))
                    self.skip_stats["kept"] += 1

    def __len__(self):
        return len(self.windows)

    def _read_window(self, row, col):
        with rasterio.open(self.tif_path) as src:
            return self._read_window_array(src, row, col)

    def __getitem__(self, idx):
        row, col = self.windows[idx]
        arr = self._read_window(row, col)

        # GeoTIFF bands:
        # B2 = blue
        # B3 = green
        # B4 = red
        # B8 = near-infrared
        # bii_label = the target band
        x = arr[:4]
        y_band = arr[4]

        if np.isnan(x).any():
            band_means = np.nanmean(x, axis=(1, 2), keepdims=True)
            band_means = np.where(np.isfinite(band_means), band_means, 0.0)
            x = np.where(np.isnan(x), band_means, x)

        y_mask = np.isfinite(y_band)
        if y_mask.sum() == 0:
            return None
        y = np.nanmean(y_band).astype(np.float32)
        if not np.isfinite(y):
            return None

        x = torch.from_numpy(x.copy())
        y = torch.tensor([y], dtype=torch.float32)
        return x, y