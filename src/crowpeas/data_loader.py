import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class CrowPeasDataModule(pl.LightningDataModule):
    """Data module for CrowPeas"""

    def __init__(
        self,
        spectra: np.ndarray,
        parameters: np.ndarray,
        random_seed: int = 42,
    ):
        """Initialize the data module

        Args:
            spectra (np.ndarray): k-weighted spectra data
            parameters (np.ndarray): Normalized parameters data
            random_seed (int, optional): Random seed. Defaults to 42.
        """
        super().__init__()
        self.spectra = spectra
        self.parameters = parameters
        self.random_seed = random_seed

    def prepare_data(self):
        pass

    def setup(self, stage: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            dataset = TensorDataset(
                torch.Tensor(self.spectra), torch.Tensor(self.parameters)
            )

            train_data_num = int(len(dataset) * train_ratio)
            val_data_num = int(len(dataset) * val_ratio)
            test_data_num = len(dataset) - train_data_num - val_data_num
            self.data_train, self.data_val, self.data_test = random_split(
                dataset,
                [train_data_num, val_data_num, test_data_num],
                generator=torch.Generator().manual_seed(self.random_seed),
            )

        if stage == "test":
            self.data_test = TensorDataset(
                torch.Tensor(self.spectra), torch.Tensor(self.parameters)
            )

        if stage == "predict":
            self.data_predict = TensorDataset(
                torch.Tensor(self.spectra), torch.Tensor(self.parameters)
            )

    def train_dataloader(self, batch_size: int | None = None):
        if (batch_size is None) or (batch_size > len(self.data_train)):
            batch_size = len(self.data_train)
        return DataLoader(self.data_train, batch_size=batch_size)

    def val_dataloader(self, batch_size: int | None = None):
        if (batch_size is None) or (batch_size > len(self.data_val)):
            batch_size = len(self.data_val)
        return DataLoader(self.data_val, batch_size=batch_size)

    def test_dataloader(self, batch_size: int | None = None):
        if (batch_size is None) or (batch_size > len(self.data_test)):
            batch_size = len(self.data_test)
        return DataLoader(self.data_test, batch_size=batch_size)

    def predict_dataloader(self, batch_size: int | None = None):
        if (batch_size is None) or (batch_size > len(self.data_predict)):
            batch_size = len(self.data_predict)
        return DataLoader(self.data_predict, batch_size=batch_size)
