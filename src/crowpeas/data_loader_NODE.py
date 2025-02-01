import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import random

class PaddedDataset(Dataset):
    """
    Dataset class for handling padded spectra, parameters, and masks.
    """
    def __init__(self, spectra: np.ndarray, parameters: np.ndarray, masks: np.ndarray):
        """
        Args:
            spectra (np.ndarray): Padded spectra data with shape [num_examples, max_len, k_len]
            parameters (np.ndarray): Padded parameters data with shape [num_examples, max_len, param_dim]
            masks (np.ndarray): Boolean masks indicating valid entries with shape [num_examples, max_len]
        """
        self.spectra = torch.tensor(spectra, dtype=torch.float32)
        self.parameters = torch.tensor(parameters, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.bool)

    def __len__(self):
        return self.spectra.shape[0]

    def __getitem__(self, idx):
        return {
            'spectra': self.spectra[idx],          # Shape: [max_len, k_len]
            'parameters': self.parameters[idx],    # Shape: [max_len, param_dim]
            'mask': self.masks[idx]                 # Shape: [max_len]
        }

class CrowPeasDataModuleNODE(pl.LightningDataModule):
    """Data module for CrowPeas with support for padded data and masks."""

    def __init__(
        self,
        spectra: np.ndarray,
        parameters: np.ndarray,
        masks: np.ndarray,
        random_seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """
        Args:
            spectra (np.ndarray): Padded spectra data with shape [num_examples, max_len, k_len]
            parameters (np.ndarray): Padded parameters data with shape [num_examples, max_len, param_dim]
            masks (np.ndarray): Boolean masks indicating valid entries with shape [num_examples, max_len]
            random_seed (int): Seed for random operations
            train_ratio (float): Proportion of data for training
            val_ratio (float): Proportion of data for validation
            batch_size (int): Batch size for DataLoaders
            num_workers (int): Number of worker processes for DataLoaders
        """
        super().__init__()
        self.spectra = spectra
        self.parameters = parameters
        self.masks = masks
        self.random_seed = random_seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Placeholders for datasets
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """No downloading or preprocessing needed."""
        pass

    def setup(self, stage: str = None):
        """
        Splits the dataset into training, validation, and testing sets.
        """
        # Initialize the dataset
        dataset = PaddedDataset(self.spectra, self.parameters, self.masks)

        # Calculate split sizes
        train_size = int(len(dataset) * self.train_ratio)
        val_size = int(len(dataset) * self.val_ratio)
        test_size = len(dataset) - train_size - val_size

        # Perform the split
        self.data_train, self.data_val, self.data_test = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.random_seed),
        )

    def _collate_fn(self, batch):
        """
        Custom collate function to stack spectra, parameters, and masks into batched tensors.

        Args:
            batch (list): List of dictionaries with keys 'spectra', 'parameters', and 'mask'

        Returns:
            tuple: (spectra_batch, parameters_batch, masks_batch)
        """
        spectra = torch.stack([item['spectra'] for item in batch])          # Shape: [batch_size, max_len, k_len]
        parameters = torch.stack([item['parameters'] for item in batch])    # Shape: [batch_size, max_len, param_dim]
        masks = torch.stack([item['mask'] for item in batch])              # Shape: [batch_size, max_len]
        return spectra, parameters, masks

    def train_dataloader(self):
        """
        Returns the training DataLoader.

        Returns:
            DataLoader: DataLoader for training data
        """
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader: DataLoader for validation data
        """
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )

    def test_dataloader(self):
        """
        Returns the testing DataLoader.

        Returns:
            DataLoader: DataLoader for testing data
        """
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )

    def predict_dataloader(self, batch_size: int = None):
        """
        Returns the prediction DataLoader if 'data_predict' is defined.

        Args:
            batch_size (int, optional): Batch size for prediction. Defaults to None.

        Raises:
            AttributeError: If 'data_predict' is not defined.

        Returns:
            DataLoader: DataLoader for prediction data
        """
        if hasattr(self, 'data_predict'):
            return DataLoader(
                self.data_predict,
                batch_size=batch_size or self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self._collate_fn,
                pin_memory=True
            )
        else:
            raise AttributeError("data_predict not defined.")
