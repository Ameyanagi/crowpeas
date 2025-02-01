import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import random

class VariableLengthDataset(torch.utils.data.Dataset):
    def __init__(self, spectra_list, parameters_list):
        self.spectra_list = spectra_list
        self.parameters_list = parameters_list

    def __len__(self):
        return len(self.spectra_list)

    def __getitem__(self, idx):
        # Return a single (spectra, parameters) pair of variable length
        return self.spectra_list[idx], self.parameters_list[idx]


class CrowPeasDataModule(pl.LightningDataModule):
    """Data module for CrowPeas"""

    def __init__(
        self,
        spectra,
        parameters,
        random_seed: int = 42,
        sequence_length: int = 50,
        t_span_min: float = 0.5,
        t_span_max: float = 10,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ):
        super().__init__()
        self.spectra = spectra
        self.parameters = parameters
        self.random_seed = random_seed
        self.sequence_length = sequence_length
        self.t_span_min = t_span_min
        self.t_span_max = t_span_max
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        # Attributes for datasets
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """No downloading needed here."""
        pass

    def setup(self, stage: str = None):
        """
        If self.spectra and self.parameters are lists of sequences (variable length),
        we directly create a VariableLengthDataset.
        """
        # Assume self.spectra and self.parameters are lists of arrays with shapes:
        # spectra[i]: [seq_length_i, 401]
        # parameters[i]: [seq_length_i, 4]

        dataset = VariableLengthDataset(self.spectra, self.parameters)

        # Compute sizes for splits
        train_size = int(len(dataset) * self.train_ratio)
        val_size = int(len(dataset) * self.val_ratio)
        test_size = len(dataset) - train_size - val_size

        # Split the dataset
        self.data_train, self.data_val, self.data_test = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.random_seed),
        )

    def _collate_fn(self, batch):
        # batch is a list of (spectra_array, params_array) pairs
        spectra_list, parameters_list = zip(*batch)

        # Convert each to a torch tensor
        spectra_list = [torch.tensor(s, dtype=torch.float32) for s in spectra_list]
        parameters_list = [torch.tensor(p, dtype=torch.float32) for p in parameters_list]

        # Create t_span for each sequence
        t_spans = [torch.linspace(0, 1, s.shape[0]) for s in spectra_list]

        # Return lists as-is for variable-length processing
        return spectra_list, parameters_list, t_spans


    def train_dataloader(self, batch_size: int = None):
        if batch_size is None or batch_size > len(self.data_train):
            batch_size = len(self.data_train)
        return DataLoader(self.data_train, batch_size=batch_size, collate_fn=self._collate_fn)

    def val_dataloader(self, batch_size: int = None):
        if batch_size is None or batch_size > len(self.data_val):
            batch_size = len(self.data_val)
        return DataLoader(self.data_val, batch_size=batch_size, collate_fn=self._collate_fn)

    def test_dataloader(self, batch_size: int = None):
        if batch_size is None or batch_size > len(self.data_test):
            batch_size = len(self.data_test)
        return DataLoader(self.data_test, batch_size=batch_size, collate_fn=self._collate_fn)

    def predict_dataloader(self, batch_size: int = None):
        if hasattr(self, 'data_predict'):
            if batch_size is None or batch_size > len(self.data_predict):
                batch_size = len(self.data_predict)
            return DataLoader(self.data_predict, batch_size=batch_size, collate_fn=self._collate_fn)
        else:
            raise AttributeError("data_predict not defined.")
