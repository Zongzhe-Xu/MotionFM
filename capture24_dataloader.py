import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import pyarrow.parquet as pq

class Capture24Dataset(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 context_len: int = 1000,
                 downsampling_ratio: float = 1.0,
                 transform=None):
        """
        Args:
            data_dir: Path to folder with parquet files
            context_len: Number of data points per sequence
            downsampling_ratio: Downsample factor (e.g., 2.0 halves the sampling rate)
            transform: Optional transform to apply to each sample
        """
        self.context_len = context_len
        self.downsampling_ratio = downsampling_ratio
        self.transform = transform

        # Get all parquet files
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        if not self.files:
            raise ValueError(f"No parquet files found in {data_dir}")

        # Precompute file lengths
        self.file_lengths = []
        for f in self.files:
            table = pq.read_table(f, columns=['x'])  # Just to get length
            self.file_lengths.append(len(table))

        # Map global index to (file_index, offset)
        self.sample_index = []
        stride = int(self.context_len * self.downsampling_ratio)
        for file_idx, length in enumerate(self.file_lengths):
            for start in range(0, length - stride, stride):
                self.sample_index.append((file_idx, start))

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        file_idx, start_idx = self.sample_index[idx]
        file_path = self.files[file_idx]

        # Read data window
        df = pd.read_parquet(file_path)
        end_idx = int(start_idx + self.context_len * self.downsampling_ratio)
        df_window = df.iloc[start_idx:end_idx:int(self.downsampling_ratio)].reset_index(drop=True)

        if len(df_window) < self.context_len:
            # Pad with zeros if too short
            pad_len = self.context_len - len(df_window)
            for col in ['x', 'y', 'z']:
                df_window[col] = df_window[col].astype(float)
            df_window = pd.concat([df_window, pd.DataFrame(np.zeros((pad_len, 3)), columns=['x', 'y', 'z'])])

        accel = df_window[['x', 'y', 'z']].to_numpy().T.astype(np.float32)  # Shape (3, context_len)
        timestamps = pd.to_datetime(df_window['time']).astype(np.int64).to_numpy()

        sample = {
            'accel_data': torch.tensor(accel),  # (3, context_len)
            'timestamps': torch.tensor(timestamps),  # (context_len,)
            'annotation': df_window['annotation'].iloc[0] if 'annotation' in df_window else ''
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
    

from torch.utils.data import DataLoader

class Capture24DataLoader:
    def __init__(self, 
                 data_dir: str, 
                 context_len: int = 1000,
                 downsampling_ratio: float = 1.0,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 shuffle: bool = True):
        
        self.dataset = Capture24Dataset(
            data_dir=data_dir,
            context_len=context_len,
            downsampling_ratio=downsampling_ratio
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            drop_last=True
        )

    def _collate_fn(self, batch):
        accel_data = torch.stack([item['accel_data'] for item in batch])
        timestamps = torch.stack([item['timestamps'] for item in batch])
        annotations = [item['annotation'] for item in batch]

        return {
            'accel_data': accel_data,  # (batch_size, 3, context_len)
            'timestamps': timestamps,  # (batch_size, context_len)
            'annotations': annotations
        }

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)