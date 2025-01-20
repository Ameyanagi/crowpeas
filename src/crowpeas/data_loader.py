import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class NoisyDataset(Dataset):
    """Dataset that dynamically adds noise to spectra"""
    
    def __init__(
        self, 
        spectra: torch.Tensor, 
        parameters: torch.Tensor,
        max_noise_std: float = 0.01,
        min_noise_std: float = 0.0,
        noise_type: str = "gaussian"
    ):
        self.spectra = spectra
        self.parameters = parameters
        self.max_noise_std = max_noise_std
        self.min_noise_std = min_noise_std
        self.noise_type = noise_type

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectrum = self.spectra[idx]
        
        # Generate random noise level for this sample
        noise_std = torch.rand(1) * (self.max_noise_std - self.min_noise_std) + self.min_noise_std
        
        # Add noise
        if self.noise_type == "gaussian":
            noise = torch.randn_like(spectrum) * noise_std
        else:  # uniform noise
            noise = (torch.rand_like(spectrum) * 2 - 1) * noise_std
            
        noisy_spectrum = spectrum + noise
        return noisy_spectrum, self.parameters[idx]


class CrowPeasDataModule(pl.LightningDataModule):
    def __init__(
        self,
        spectra: np.ndarray,
        parameters: np.ndarray,
        random_seed: int = 42,
        max_noise_std: float = 0.01,
        min_noise_std: float = 0.0,
        noise_type: str = "gaussian",
        num_workers: int | None = None
    ):
        super().__init__()
        self.spectra = torch.Tensor(spectra)
        self.parameters = torch.Tensor(parameters)
        self.random_seed = random_seed
        self.max_noise_std = max_noise_std
        self.min_noise_std = min_noise_std
        self.noise_type = noise_type
        
        # Auto-detect optimal workers if not specified
        if num_workers is None:
            ValueError("num_workers must be an integer or None")
        else:
            self.num_workers = num_workers
            

    def setup(self, stage: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
        if stage == "fit":
            # Create noisy dataset for training
            dataset = NoisyDataset(
                self.spectra,
                self.parameters,
                max_noise_std=self.max_noise_std,
                min_noise_std=self.min_noise_std,
                noise_type=self.noise_type
            )

            # Split into train/val/test
            train_size = int(len(dataset) * train_ratio)
            val_size = int(len(dataset) * val_ratio)
            test_size = len(dataset) - train_size - val_size

            self.data_train, self.data_val, self.data_test = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.random_seed),
            )

        if stage == "test":
            # Use clean data for testing
            self.data_test = NoisyDataset(
                self.spectra,
                self.parameters,
                max_noise_std=0.0
            )

        if stage == "predict":
            # Use clean data for prediction
            self.data_predict = NoisyDataset(
                self.spectra,
                self.parameters,
                max_noise_std=0.0
            )

    def train_dataloader(self, batch_size: int | None = None):
        if (batch_size is None) or (batch_size > len(self.data_train)):
            batch_size = len(self.data_train)
        return DataLoader(
            self.data_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True  # Helps with GPU training
        )

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



# import lightning.pytorch as pl
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, TensorDataset, random_split


# class CrowPeasDataModule(pl.LightningDataModule):
#     """Data module for CrowPeas"""

#     def __init__(
#         self,
#         spectra: np.ndarray,
#         parameters: np.ndarray,
#         random_seed: int = 42,
#     ):
#         """Initialize the data module

#         Args:
#             spectra (np.ndarray): k-weighted spectra data
#             parameters (np.ndarray): Normalized parameters data
#             random_seed (int, optional): Random seed. Defaults to 42.
#         """
#         super().__init__()
#         self.spectra = spectra
#         self.parameters = parameters
#         self.random_seed = random_seed

#     def prepare_data(self):
#         pass

#     def setup(self, stage: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
#         # Assign train/val datasets for use in dataloaders
#         if stage == "fit":
#             dataset = TensorDataset(
#                 torch.Tensor(self.spectra), torch.Tensor(self.parameters)
#             )

#             train_data_num = int(len(dataset) * train_ratio)
#             val_data_num = int(len(dataset) * val_ratio)
#             test_data_num = len(dataset) - train_data_num - val_data_num
#             self.data_train, self.data_val, self.data_test = random_split(
#                 dataset,
#                 [train_data_num, val_data_num, test_data_num],
#                 generator=torch.Generator().manual_seed(self.random_seed),
#             )

#         if stage == "test":
#             self.data_test = TensorDataset(
#                 torch.Tensor(self.spectra), torch.Tensor(self.parameters)
#             )

#         if stage == "predict":
#             self.data_predict = TensorDataset(
#                 torch.Tensor(self.spectra), torch.Tensor(self.parameters)
#             )

#     def train_dataloader(self, batch_size: int | None = None):
#         if (batch_size is None) or (batch_size > len(self.data_train)):
#             batch_size = len(self.data_train)
#         return DataLoader(self.data_train, batch_size=batch_size)

#     def val_dataloader(self, batch_size: int | None = None):
#         if (batch_size is None) or (batch_size > len(self.data_val)):
#             batch_size = len(self.data_val)
#         return DataLoader(self.data_val, batch_size=batch_size)

#     def test_dataloader(self, batch_size: int | None = None):
#         if (batch_size is None) or (batch_size > len(self.data_test)):
#             batch_size = len(self.data_test)
#         return DataLoader(self.data_test, batch_size=batch_size)

#     def predict_dataloader(self, batch_size: int | None = None):
#         if (batch_size is None) or (batch_size > len(self.data_predict)):
#             batch_size = len(self.data_predict)
#         return DataLoader(self.data_predict, batch_size=batch_size)
