import random

import numpy as np
from larch.fitting import guess, param, param_group
from larch.io import read_ascii
from larch.xafs import (
    autobk,
    feffit,
    feffit_dataset,
    feffit_report,
    feffit_transform,
    feffpath,
    path2chi,
)
import pandas as pd
from pathlib import Path


class SyntheticSpectrum:

    training_mode: bool
    num_examples: int
    spectra: np.ndarray
    parameters: np.ndarray
    k: np.ndarray | None
    k_weight: int
    spectrum_noise: bool
    k_range: tuple | list
    k_grid: np.ndarray

    def __init__(
        self,
        feff_path_file,
        path_parameters=None,
        param_ranges=None,
        training_mode=False,
        num_examples=1,
        spectrum_noise=False,
        noise_range=(0, 0.01),
        k_range=(2.5, 12.5),
        k_weight=2,
        seed=None,
        generate=True,
    ):
        self.training_mode = training_mode
        self.num_examples = num_examples

        path1 = feffpath(feff_path_file)
        self.k = path1.k

        # This should be k_weighted_chi and add k_weight
        self.kweighted_chi = None
        self.k_weight = k_weight
        self.spectrum_noise = spectrum_noise
        self.noise_range = noise_range
        self.k_range = k_range
        self.k_grid = np.linspace(k_range[0], k_range[1], num=100)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if generate:
            if self.training_mode:
                self.generate_training_examples(feff_path_file, param_ranges)
            else:
                self.generate_spectrum(feff_path_file, path_parameters, param_ranges)

    def generate_spectrum(
        self, feff_path_file, path_parameters=None, param_ranges=None
    ):
        path1 = feffpath(feff_path_file)

        if param_ranges:
            path_parameters = [
                random.uniform(param_ranges[key][0], param_ranges[key][1])
                for key in ["s02", "degen", "deltar", "sigma2", "e0"]
            ]

        if path_parameters:
            path1.s02 = path_parameters[0]
            path1.degen = path_parameters[1]
            path1.A = path1.s02 * path1.degen
            path1.deltar = path_parameters[2]
            path1.sigma2 = path_parameters[3]
            path1.e0 = path_parameters[4]

        path2chi(path1)

        if self.k is None:
            self.k = path1.k

        self.kweighted_chi = path1.chi * path1.k**self.k_weight

        if self.spectrum_noise:
            noise_level = random.uniform(self.noise_range[0], self.noise_range[1])
            noise = np.random.normal(0, noise_level, len(self.kweighted_chi))
            self.kweighted_chi += noise

        return path1.s02 * path1.degen, path1.deltar, path1.sigma2, path1.e0
        # return path1.s02 * path1.degen, path1.s02, path1.degen, path1.deltar, path1.sigma2, path1.e0

    def generate_training_examples(self, feff_path_file, param_ranges):

        tmp_spectra = []
        tmp_parameters = []
        for _ in range(self.num_examples):
            spectrum = self.generate_spectrum(feff_path_file, param_ranges=param_ranges)
            # interpolated_spectrum = self.interpolate_spectrum(
            #     self.k, self.k2weighted, self.k_grid
            # )
            tmp_spectra.append(self.kweighted_chi)
            tmp_parameters.append(spectrum)

        self.spectra = np.array(tmp_spectra)
        self.parameters = np.array(tmp_parameters)

    def interpolate_spectrum(self, original_k, original_spectrum, target_k):
        """
        Interpolate the original spectrum onto the target k-grid.

        Parameters:
        original_k (np.ndarray): The original k-grid.
        original_spectrum (np.ndarray): The original spectrum values.
        target_k (np.ndarray): The target k-grid.

        Returns:
        np.ndarray: Interpolated spectrum values on the target k-grid.
        """
        interpolated_spectrum = np.interp(target_k, original_k, original_spectrum)
        return interpolated_spectrum

    def plot(self):
        if not self.training_mode:
            plt.plot(self.k, self.kweighted_chi, label=r"$\sigma^2 = 0$")
            plt.xlabel(r" $ k \rm\, (\AA^{-1})$")
            plt.ylabel(r"$ k^2\chi(k)$")
            plt.legend()
            plt.show()
        else:
            print("Training mode is active. No plot available.")

    def get_training_data(self):
        if self.training_mode is False:
            raise ValueError("Training mode is not active. No training data available.")

        if self.spectra is None or self.parameters is None:
            raise ValueError("No training data available.")
            w
        return self.spectra, self.parameters

    def save_training_data(self, prefix="./synthetic"):

        if self.training_mode is False:
            raise ValueError("Training mode is not active. No data to save.")

        filename_spectra = f"{prefix}_spectra.feather"
        filename_parameters = f"{prefix}_parameters.feather"

        spectra_df = pd.DataFrame(self.spectra)
        parameters_df = pd.DataFrame(
            self.parameters, columns=["A", "deltar", "sigma2", "e0"]
        )
        spectra_df.to_feather(filename_spectra)
        parameters_df.to_feather(filename_parameters)

        print(f"Spectra data saved to {filename_spectra}")
        print(f"Parameters data saved to {filename_parameters}")

    def exists_training_data(self, prefix="./synthetic"):
        filename_spectra = f"{prefix}_spectra.feather"
        filename_parameters = f"{prefix}_parameters.feather"

        if Path(filename_spectra).is_file() and Path(filename_parameters).is_file():
            return True

        return False

    def load_training_data(self, prefix="./synthetic"):

        filename_spectra = f"{prefix}_spectra.feather"
        filename_parameters = f"{prefix}_parameters.feather"

        self.spectra = pd.read_feather(filename_spectra).values
        self.parameters = pd.read_feather(filename_parameters).values
        print(f"Spectra data loaded from {filename_spectra}")
        print(f"Parameters data loaded from {filename_parameters}")

    def free_training_data(self):
        del self.spectra
        del self.parameters
        print("Training data has been freed.")
