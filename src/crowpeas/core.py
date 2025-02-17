"""Main module."""

from functools import reduce
from numpy.linalg import cond, norm
from scipy.ndimage import label
import toml
import os
import math
from typing import Sequence, Literal
import json
import pandas as pd
import numdifftools as nd
from uncertainties import covariance_matrix
from lightning.pytorch.callbacks import EarlyStopping

from .data_generator import SyntheticSpectrum
from .data_generatorS import SyntheticSpectrumS
from .data_loader import CrowPeasDataModule
from .data_loader_NODE import CrowPeasDataModuleNODE
import lightning as pl
from .model.BNN import BNN
from .model.MLP import MLP
from .model.CNN import CNN
from .model.NODE import NODE
from .model.HetMLPNM import hetMLP
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
from torchinfo import torchinfo
from laplace import Laplace
from scipy.stats import f, multivariate_normal
import itertools
from scipy.integrate import nquad
import uncertainties

from .utils import (
    normalize_data,
    denormalize_data,
    normalize_spectra,
    denormalize_spectra,
    interpolate_spectrum,
    predict_with_uncertainty,
    predict_with_uncertainty_hetMLP,
    normalize_data_S,
    normalize_spectra_S
)
from larch.xafs import xftf, xftr, feffpath, path2chi, ftwindow, estimate_noise
from larch.io import read_ascii
from larch.fitting import param, guess, param_group
from larch import Group 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

import plotext as plt_t

from scipy.linalg import inv

from .utils import create_random_series

# from laplace import Laplace, marglik_training
# from copy import deepcopy


class CrowPeas:
    # general parameters
    title: str = ""
    config_filename: str
    config_dir: str
    config: dict
    exp_config_filename: str
    exp_config_dir: str
    exp_config: dict
    norm_params_spectra: dict
    norm_params_parameters: dict
    output_dir: str

    k: np.ndarray

    # parameters related to general neural network
    neural_network: dict
    model_name: str
    model_dir: str
    checkpoint_dir: str
    checkpoint_name: str
    model: pl.LightningModule

    # parameters related to neural network architecture
    nn_type: Literal["MLP", "BNN", "hetMLP", "CNN", "NODE"]
    nn_activation: str
    nn_output_activation: str
    nn_output_dim: int
    nn_hidden_dims: Sequence[int]
    nn_weight_decay: float
    nn_dropout_rates: Sequence[float]
    nn_filter_sizes: Sequence[int]
    nn_kernel_sizes: Sequence[int]
    nn_k_grid: np.ndarray

    # parameters related to training mode
    seed: int | None = None
    feff_path_file: str
    training_set_dir: str
    training_data_prefix: str
    param_ranges: dict
    training_mode: bool
    num_examples: int
    spectrum_noise: bool
    input_type: str
    noise_range: Sequence[float]
    k_range: Sequence[float]
    r_range: Sequence[float]
    k_weight: int
    train_size_ratio: float
    val_size_ratio: float
    test_size_ratio: float
    training_noise: bool
    training_noise_range: Sequence[float]

    # parameters related to neural network hyperparameters
    epochs: int
    batch_size: int
    learning_rate: float

    # synthetic spectra for training
    synthetic_spectra: SyntheticSpectrum | SyntheticSpectrumS
    data_loader: CrowPeasDataModule
    history: dict = {"train/loss": [], "val/loss": []}

    # only used for validating the test results
    x_test: torch.Tensor
    y_test: torch.Tensor
    denormalized_x_test: np.ndarray
    denormalized_y_test: np.ndarray

    test_pred: torch.Tensor
    denormalized_test_pred: np.ndarray

    def __init__(self) -> None:
        pass

    def load_config(self, config_file: str):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found")

        self.config_dir = os.path.dirname(config_file)

        if config_file.endswith("toml"):
            self.config_filename = config_file
            with open(config_file, "r") as f:
                self.config = toml.load(f)
        elif config_file.endswith("json"):
            self.config_filename = config_file
            with open(config_file, "r") as f:
                self.config = json.load(f)
        else:
            raise ValueError(
                "File format not supported. Only TOML and JSON are supported."
            )
        with open(config_file, "r") as f:
            self.config = toml.load(f)

        return self

    def load_and_validate_config(self, validate_save_config=True):
        required_sections = ["general", "training", "neural_network"]
        required_for_general = ["title", "mode", "output_dir"]
        required_for_training = [
            "feffpath",
            "param_ranges",
            "training_set_dir",
            "num_examples",
            "k_weight",
            "spectrum_noise",
            "noise_range",
            "input_type",
            "k_range",
        ]
        required_for_training_param_ranges = ["s02", "degen", "deltar", "sigma2", "e0"]
        required_for_nn = [
            "model_name",
            "model_dir",
            "checkpoint_dir",
            "checkpoint_name",
        ]
        required_for_nn_hyperparameters = ["epochs", "batch_size", "learning_rate"]
        required_for_nn_architecture = [
            "type",
            "activation",
            "output_activation",
            "output_dim",
            "hidden_dims",
            "dropout_rates",
            "weight_decay",
            "filter_sizes",
            "kernel_sizes",
        ]
        required_for_experiment = [
            "dataset_names",
            "dataset_dir",
            "k_range",
            "r_range",
            "k_weight",
        ]
        required_for_artemis = [
            "result",
            "unc",

        ]

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Section {section} is missing in config file")

        for param in required_for_general:
            if param not in self.config["general"]:
                raise ValueError(
                    f"Parameter {param} is missing in general section of config file"
                )

        if self.config["general"]["mode"].lower().startswith("t"):
            for param in required_for_training:
                if param not in self.config["training"]:
                    raise ValueError(
                        f"Parameter {param} is missing in training section of config file"
                    )

            if "param_ranges" not in self.config["training"]:
                raise ValueError(
                    "Parameter param_ranges is missing in training section of config file"
                )

            for param in required_for_training_param_ranges:
                if param not in self.config["training"]["param_ranges"]:
                    raise ValueError(
                        f"Parameter {param} is missing in param_ranges section of config file"
                    )

            if "hyperparameters" not in self.config["neural_network"]:
                raise ValueError(
                    "Parameter hyperparameters is missing in neural_network section of config file"
                )

            for param in required_for_nn_hyperparameters:
                if param not in self.config["neural_network"]["hyperparameters"]:
                    raise ValueError(
                        f"Parameter {param} is missing in hyperparameters section of config file"
                    )

        if "neural_network" not in self.config:
            raise ValueError("Section neural_network is missing in config file")

        for param in required_for_nn:
            if param not in self.config["neural_network"]:
                raise ValueError(
                    f"Parameter {param} is missing in neural_network section of config file"
                )

        if "architecture" not in self.config["neural_network"]:
            raise ValueError(
                "Parameter architecture is missing in neural_network section of config file"
            )

        for param in required_for_nn_architecture:
            if param not in self.config["neural_network"]["architecture"]:
                raise ValueError(
                    f"Parameter {param} is missing in architecture section of config file"
                )

        if "experiment" in self.config:
            print("Experiment section found")
            for param in required_for_experiment:
                if param not in self.config["experiment"]:
                    raise ValueError(
                        f"Parameter {param} is missing in experiment section of config file"
                    )
            self.exp_krange = self.config["experiment"]["k_range"]
            self.exp_rrange = self.config["experiment"]["r_range"]
        
        if "artemis" in self.config:
            print("Artemis section found")
            for param in required_for_artemis:
                if param not in self.config["artemis"]:
                    raise ValueError(
                        f"Parameter {param} is missing in artemis section of config file"
                    )
        # Convert numeric strings to floats in norm_params_spectra
        norm_params_spectra = self.config.get('general', {}).get('norm_params_spectra', {})
        self.config['general']['norm_params_spectra'] = {
            key: float(value) for key, value in norm_params_spectra.items()
        }

        # Repeat the process for norm_params_parameters
        norm_params_parameters = self.config.get('general', {}).get('norm_params_parameters')
        if isinstance(norm_params_parameters, list):
            self.config['general']['norm_params_parameters'] = np.array(norm_params_parameters)
        elif isinstance(norm_params_parameters, dict):
            self.config['general']['norm_params_parameters'] = norm_params_parameters
    
        # Ensure experiment[dataset_dir] is a list of strings
        dataset_dirs = self.config.get('experiment', {}).get('dataset_dir', [])
        if isinstance(dataset_dirs, list):
            self.config['experiment']['dataset_dir'] = [str(dir) for dir in dataset_dirs]
        
        # Ensure experiment[dataset_names] is a list of strings
        dataset_names = self.config.get('experiment', {}).get('dataset_names', [])
        if isinstance(dataset_names, list):
            self.config['experiment']['dataset_names'] = [str(name) for name in dataset_names]
        

        # Convert artemis[result] and artemis[unc] to lists of lists of floats
        artemis_results = self.config.get('artemis', {}).get('result', [])
        if isinstance(artemis_results, list):
            self.config['artemis']['result'] = [
                [float(item) for item in sublist] if isinstance(sublist, list) else [float(sublist)]
                for sublist in artemis_results
            ]

        artemis_unc = self.config.get('artemis', {}).get('unc', [])
        if isinstance(artemis_unc, list):
            self.config['artemis']['unc'] = [
                [float(item) for item in sublist] if isinstance(sublist, list) else [float(sublist)]
                for sublist in artemis_unc
            ]

        self.title = self.config["general"]["title"]
        self.output_dir = self.config["general"]["output_dir"]
        self.norm_params_spectra = self.config["general"].get(
            "norm_params_spectra", None
        )
        self.norm_params_parameters = self.config["general"].get(
            "norm_params_parameters", None
        )

        if self.config["general"]["mode"].lower().startswith("t"):
            self.training_mode = True
        else:
            self.training_mode = False

        self.model_name = self.config["neural_network"]["model_name"]

        if os.path.isabs(self.config["neural_network"]["model_dir"]):
            self.model_dir = self.config["neural_network"]["model_dir"]
        else:
            self.model_dir = os.path.join(
                self.config_dir, self.config["neural_network"]["model_dir"]
            )

        if os.path.isabs(self.config["neural_network"]["checkpoint_dir"]):
            self.checkpoint_dir = self.config["neural_network"]["checkpoint_dir"]
        else:
            self.checkpoint_dir = os.path.join(
                self.config_dir, self.config["neural_network"]["checkpoint_dir"]
            )

        self.checkpoint_name = self.config["neural_network"]["checkpoint_name"]
        #print("Checkpoint name: ", self.checkpoint_name)

        self.nn_type = self.config["neural_network"]["architecture"]["type"]
        self.nn_activation = self.config["neural_network"]["architecture"]["activation"]
        self.nn_output_activation = self.config["neural_network"]["architecture"][
            "output_activation"
        ]
        self.nn_output_dim = self.config["neural_network"]["architecture"]["output_dim"]
        self.nn_hidden_dims = self.config["neural_network"]["architecture"][
            "hidden_dims"
        ]
        self.nn_dropout_rates = self.config["neural_network"]["architecture"].get(
            "dropout_rates", None
        )
        self.nn_weight_decay = self.config["neural_network"]["architecture"].get(
            "weight_decay", None
        )
        self.nn_filter_sizes = self.config["neural_network"]["architecture"][
            "filter_sizes"
        ]
        self.nn_kernel_sizes = self.config["neural_network"]["architecture"][
            "kernel_sizes"
        ]
        self.nn_k_grid = self.config["neural_network"].get("k_grid", None)

        if self.training_mode:
            if os.path.isabs(self.config["training"]["feffpath"]):
                self.feff_path_file = self.config["training"]["feffpath"]
            else:
                self.feff_path_file = os.path.join(
                    self.config_dir, self.config["training"]["feffpath"]
                )

            if os.path.isabs(self.config["training"]["training_set_dir"]):
                self.training_set_dir = self.config["training"]["training_set_dir"]
            else:
                self.training_set_dir = os.path.join(
                    self.config_dir, self.config["training"]["training_set_dir"]
                )
            self.training_data_prefix = self.config["training"].get(
                "training_data_prefix", None
            )
            self.param_ranges = self.config["training"]["param_ranges"]
            self.num_examples = self.config["training"]["num_examples"]
            self.spectrum_noise = self.config["training"]["spectrum_noise"]
            self.noise_range = self.config["training"]["noise_range"]
            self.input_type = self.config["training"].get("input_type", "r")
            self.k_range = self.config["training"]["k_range"]
            self.k_weight = self.config["training"]["k_weight"]
            self.r_range = self.config["training"]["r_range"]
            self.seed = self.config["training"].get("seed", None)
            self.train_size_ratio = self.config["training"].get("train_size_ratio", 0.8)
            self.val_size_ratio = self.config["training"].get("val_size_ratio", 0.1)
            self.test_size_ratio = self.config["training"].get("test_size_ratio", 0.1)
            self.training_noise = self.config["training"].get("training_noise", False)
            self.training_noise_range = self.config["training"].get(
                "training_noise_range", None
            )



        self.epochs = self.config["neural_network"]["hyperparameters"]["epochs"]
        self.batch_size = self.config["neural_network"]["hyperparameters"]["batch_size"]
        self.learning_rate = self.config["neural_network"]["hyperparameters"][
            "learning_rate"
        ]

        if validate_save_config:
            config = self.config
            self.save_config()

            not_to_check = [
                "seed",
                "feffpath",
                "training_set_dir",
                "model_dir",
                "checkpoint_dir",
            ]

            for key, value in self.config.items():
                if key in not_to_check:
                    continue

                if key not in config:
                    print(f"Key {key} not in config")

                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subkey in not_to_check or subvalue is None:
                            continue
                        if subkey not in config[key]:
                            print(f"Subkey {subkey} not in config")
                        elif value[subkey] != config[key][subkey]:
                            print(f"Subkey {subkey} is different")

                elif value != config[key]:
                    print(f"Key {key} is different")

        else:
            self.save_config()

        
        return self

    def init_obj_params(self):
        self.exp_obj_params = []


    def save_config(self, path: str | None = None):

        # check is the output directory exists, if not create it
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        

        # Process norm_params_parameters
        norm_params_parameters = self.config["general"].get('norm_params_parameters')
        if isinstance(norm_params_parameters, dict):
            for key, value in norm_params_parameters.items():
                if isinstance(value, np.ndarray):
                    norm_params_parameters[key] = value.tolist()
        elif isinstance(norm_params_parameters, np.ndarray):
            self.config["general"]['norm_params_parameters'] = norm_params_parameters.tolist()

        if "experiment" in self.config and "artemis" not in self.config:
            self.config = {
            "general": {
                "title": self.title,
                "mode": "training" if self.training_mode else "inference",
                "output_dir": self.output_dir,
                "seed": self.seed,
                "norm_params_spectra": self.norm_params_spectra,
                "norm_params_parameters": self.norm_params_parameters,
            },
            "training": (
                {
                    "feffpath": os.path.relpath(self.feff_path_file, self.config_dir),
                    "training_set_dir": os.path.relpath(
                        self.training_set_dir, self.config_dir
                    ),
                    "training_data_prefix": self.training_data_prefix,
                    "num_examples": self.num_examples,
                    "spectrum_noise": self.spectrum_noise,
                    "noise_range": self.noise_range,
                    "input_type": self.input_type,
                    "k_range": self.k_range,
                    "k_weight": self.k_weight,
                    "r_range": self.r_range,
                    "train_size_ratio": self.train_size_ratio,
                    "val_size_ratio": self.val_size_ratio,
                    "test_size_ratio": self.test_size_ratio,
                    "training_noise": self.training_noise,
                    "training_noise_range": self.training_noise_range,
                    "param_ranges": self.param_ranges,
                }
            ),
            "neural_network": {
                "model_name": self.model_name,
                "model_dir": os.path.relpath(self.model_dir, self.config_dir),
                "checkpoint_dir": os.path.relpath(self.checkpoint_dir, self.config_dir),
                "checkpoint_name": self.checkpoint_name,
                "hyperparameters": {
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                },
                "architecture": {
                    "type": self.nn_type,
                    "activation": self.nn_activation,
                    "output_activation": self.nn_output_activation,
                    "output_dim": self.nn_output_dim,
                    "hidden_dims": self.nn_hidden_dims,
                    "dropout_rates": self.nn_dropout_rates,
                    "weight_decay": self.nn_weight_decay,
                    "filter_sizes": self.nn_filter_sizes,
                    "kernel_sizes": self.nn_kernel_sizes,
                },
                "k_grid": self.nn_k_grid,
            },
            "experiment": {
                "dataset_names": self.config["experiment"]["dataset_names"],
                "dataset_dir": self.config["experiment"]["dataset_dir"],
                "k_range": self.config["experiment"]["k_range"],
                "r_range": self.config["experiment"]["r_range"],
                "k_weight": self.config["experiment"]["k_weight"],
            },
            
            
                }
        if "experiment" in self.config and "artemis" in self.config:
            self.config = {
            "general": {
                "title": self.title,
                "mode": "training" if self.training_mode else "inference",
                "output_dir": self.output_dir,
                "seed": self.seed,
                "norm_params_spectra": self.norm_params_spectra,
                "norm_params_parameters": self.norm_params_parameters,
            },
            "training": (
                {
                    "feffpath": os.path.relpath(self.feff_path_file, self.config_dir),
                    "training_set_dir": os.path.relpath(
                        self.training_set_dir, self.config_dir
                    ),
                    "training_data_prefix": self.training_data_prefix,
                    "num_examples": self.num_examples,
                    "spectrum_noise": self.spectrum_noise,
                    "noise_range": self.noise_range,
                    "input_type": self.input_type,
                    "k_range": self.k_range,
                    "k_weight": self.k_weight,
                    "r_range": self.r_range,
                    "train_size_ratio": self.train_size_ratio,
                    "val_size_ratio": self.val_size_ratio,
                    "test_size_ratio": self.test_size_ratio,
                    "training_noise": self.training_noise,
                    "training_noise_range": self.training_noise_range,
                    "param_ranges": self.param_ranges,
                }
            ),
            "neural_network": {
                "model_name": self.model_name,
                "model_dir": os.path.relpath(self.model_dir, self.config_dir),
                "checkpoint_dir": os.path.relpath(self.checkpoint_dir, self.config_dir),
                "checkpoint_name": self.checkpoint_name,
                "hyperparameters": {
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                },
                "architecture": {
                    "type": self.nn_type,
                    "activation": self.nn_activation,
                    "output_activation": self.nn_output_activation,
                    "output_dim": self.nn_output_dim,
                    "hidden_dims": self.nn_hidden_dims,
                    "dropout_rates": self.nn_dropout_rates,
                    "weight_decay": self.nn_weight_decay,
                    "filter_sizes": self.nn_filter_sizes,
                    "kernel_sizes": self.nn_kernel_sizes,
                },
                "k_grid": self.nn_k_grid,
            },
            "experiment": {
                "dataset_names": self.config["experiment"]["dataset_names"],
                "dataset_dir": self.config["experiment"]["dataset_dir"],
                "k_range": self.config["experiment"]["k_range"],
                "r_range": self.config["experiment"]["r_range"],
                "k_weight": self.config["experiment"]["k_weight"],
            },
            "artemis": {
                "result": self.config["artemis"]["result"],
                "unc": self.config["artemis"]["unc"],

            },
            
            
                }
        else:

            self.config = {
                "general": {
                    "title": self.title,
                    "mode": "training" if self.training_mode else "inference",
                    "output_dir": self.output_dir,
                    "seed": self.seed,
                    "norm_params_spectra": self.norm_params_spectra,
                    "norm_params_parameters": self.norm_params_parameters,
                },
                "training": (
                    {
                        "feffpath": os.path.relpath(self.feff_path_file, self.config_dir),
                        "training_set_dir": os.path.relpath(
                            self.training_set_dir, self.config_dir
                        ),
                        "training_data_prefix": self.training_data_prefix,
                        "num_examples": self.num_examples,
                        "spectrum_noise": self.spectrum_noise,
                        "noise_range": self.noise_range,
                        "input_type": self.input_type,
                        "k_range": self.k_range,
                        "k_weight": self.k_weight,
                        "r_range": self.r_range,
                        "train_size_ratio": self.train_size_ratio,
                        "val_size_ratio": self.val_size_ratio,
                        "test_size_ratio": self.test_size_ratio,
                        "training_noise": self.training_noise,
                        "training_noise_range": self.training_noise_range,
                        "param_ranges": self.param_ranges,
                    }
                ),
                "neural_network": {
                    "model_name": self.model_name,
                    "model_dir": os.path.relpath(self.model_dir, self.config_dir),
                    "checkpoint_dir": os.path.relpath(self.checkpoint_dir, self.config_dir),
                    "checkpoint_name": self.checkpoint_name,
                    "hyperparameters": {
                        "epochs": self.epochs,
                        "batch_size": self.batch_size,
                        "learning_rate": self.learning_rate,
                    },
                    "architecture": {
                        "type": self.nn_type,
                        "activation": self.nn_activation,
                        "output_activation": self.nn_output_activation,
                        "output_dim": self.nn_output_dim,
                        "hidden_dims": self.nn_hidden_dims,
                        "dropout_rates": self.nn_dropout_rates,
                        "weight_decay": self.nn_weight_decay,
                        "filter_sizes": self.nn_filter_sizes,
                        "kernel_sizes": self.nn_kernel_sizes,
                    },
                    "k_grid": self.nn_k_grid,
                },
                
            }

        print(path)

        if path is not None:
            if path.endswith("toml"):
                self.config_filename = path
                self.config_dir = os.path.dirname(path)

                with open(path, "w") as f:
                    toml.dump(self.config, f)
            elif path.endswith("json"):
                self.config_filename = path
                self.config_dir = os.path.dirname(path)

                with open(path, "w") as f:
                    json.dump(self.config, f)
            else:
                raise ValueError(
                    "File format not supported. Only TOML and JSON are supported."
                )
        else:
            if self.config_filename.endswith("toml"):
                print("Saving config file")
                print(self.config_dir)
                print(self.output_dir)
                self.config_filename = os.path.basename(self.config_filename) # if given a long path
                print(self.config_filename)
                save_path = os.path.join(self.config_dir, self.output_dir)
                with open(save_path + "/" + self.config_filename, "w") as f:
                    toml.dump(self.config, f)
            elif self.config_filename.endswith("json"): #TODO add json support
                with open(self.config_filename, "w") as f:
                    json.dump(self.config, f)
            else:
                raise ValueError(
                    "File format not supported. Only TOML and JSON are supported."
                )

        return self

    def init_synthetic_spectra(self, generate=True): # for training
        self.synthetic_spectra = SyntheticSpectrum(
            feff_path_file=self.feff_path_file,
            param_ranges=self.param_ranges,
            training_mode=self.training_mode,
            num_examples=self.num_examples,
            k_weight=self.k_weight,
            k_range=self.k_range,
            spectrum_noise=self.spectrum_noise,
            noise_range=self.noise_range,
            seed=self.seed,
            generate=generate,
        )

        if hasattr(self, "synthetic_spectra") and (
            self.synthetic_spectra.k is not None
        ):
            self.nn_k_grid = self.synthetic_spectra.k

        return self

    def init_synthetic_spectra_S(self, generate=True): # for testing with sequences
        self.synthetic_spectra = SyntheticSpectrumS(
            feff_path_file=self.feff_path_file,
            param_ranges=self.param_ranges,
            training_mode=self.training_mode,
            num_examples=self.num_examples,
            k_weight=self.k_weight,
            k_range=self.k_range,
            spectrum_noise=self.spectrum_noise,
            noise_range=self.noise_range,
            seed=self.seed,
            generate=generate,
        )

        if hasattr(self, "synthetic_spectra") and (
            self.synthetic_spectra.k is not None
        ):
            self.nn_k_grid = self.synthetic_spectra.k

        return self

    def load_synthetic_spectra(self):

        training_data_prefix = os.path.join(
            self.training_set_dir, self.training_data_prefix
        )

        if not hasattr(self, "synthetic_spectra") or self.synthetic_spectra is None:
            self.init_synthetic_spectra(generate=False)

        if not self.synthetic_spectra.exists_training_data(prefix=training_data_prefix):
            print("Generating synthetic spectra")
            self.init_synthetic_spectra(generate=True)
            self.save_training_data()

            return self

        #print("Loading training data")
        print(f"Loading training data from {training_data_prefix}")

        self.synthetic_spectra.load_training_data(
            prefix=training_data_prefix,
        )

        return self

    def prepare_dataloader(self, setup="fit"):
        if self.synthetic_spectra is None:
            self.init_synthetic_spectra()

        if self.seed is None:
            seed = 42
        else:
            seed = self.seed
       # check network type
        if self.config["neural_network"]["architecture"]["type"].lower().startswith("node"):

            mask_bool = self.synthetic_spectra.masks.astype(bool)

            normalized_spectra, norm_spectra_params = normalize_spectra_S(
                self.synthetic_spectra.spectra, mask_bool
            )
            normalized_parameters, norm_parameters_params = normalize_data_S(
                self.synthetic_spectra.parameters, mask_bool
            )
        else:

            # TODO: read normal_params
            normalized_spectra, norm_spectra_params = normalize_spectra(
                self.synthetic_spectra.spectra
            )
            normalized_parameters, norm_parameters_params = normalize_data(
                self.synthetic_spectra.parameters
            )

        self.norm_params_spectra = norm_spectra_params
        self.norm_params_parameters = norm_parameters_params



        if self.config["neural_network"]["architecture"]["type"].lower().startswith("node"):
            self.data_loader = CrowPeasDataModuleNODE(
                spectra=normalized_spectra,
                parameters=normalized_parameters,
                masks=mask_bool,
                random_seed=seed,
                train_ratio=self.config.get("train_ratio", 0.8),
                val_ratio=self.config.get("val_ratio", 0.1),
                batch_size=self.batch_size,
                num_workers=self.config.get("num_workers", 4),
            )
            self.data_loader.setup(stage=setup)

        else:
            self.data_loader = CrowPeasDataModule(
            spectra=normalized_spectra,
            parameters=normalized_parameters,
            random_seed=seed,
            min_noise_std=self.training_noise_range[0], # added
            max_noise_std=self.training_noise_range[-1], # added
            noise_type="gaussian" , # added
            num_workers=16 # added
            )

        self.data_loader.setup(setup)
        print("DataLoader has been prepared.")

        return self

    def get_training_data(self):
        if not hasattr(self, "synthetic_spectra") or self.synthetic_spectra is None:
            self.init_synthetic_spectra()

        return self.synthetic_spectra.get_training_data()

    def save_training_data(self):
        if not hasattr(self, "synthetic_spectra") or self.synthetic_spectra is None:
            self.init_synthetic_spectra()

        if self.training_data_prefix is None:
            self.training_data_prefix = self.title + "_training_data"

        save_path = os.path.join(self.training_set_dir, self.training_data_prefix)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.synthetic_spectra.save_training_data(save_path)
        print(f"Training data saved to {save_path}")

        return self

    def load_model(self):
        if self.nn_type.lower().startswith("bnn"):
            self.model = BNN.load_from_checkpoint(
                os.path.join(self.checkpoint_dir, self.checkpoint_name)
            )
        elif self.nn_type.lower().startswith("mlp"):
            self.model = MLP.load_from_checkpoint(
                os.path.join(self.checkpoint_dir, self.checkpoint_name), strict=False
            )
        elif self.nn_type.lower().startswith("node"):
            self.model = NODE.load_from_checkpoint(
                os.path.join(self.checkpoint_dir, self.checkpoint_name)
            )
        elif self.nn_type.lower().startswith("cnn"):
            self.model = CNN.load_from_checkpoint(
                os.path.join(self.checkpoint_dir, self.checkpoint_name)
            )            
        elif self.nn_type.lower().startswith("het"):
            self.model = hetMLP.load_from_checkpoint(
                os.path.join(self.checkpoint_dir, self.checkpoint_name)
            )

        return self

    def init_model(self):
        if self.nn_type.lower().startswith("bnn"):
            self.model = BNN(
                hidden_layers=self.nn_hidden_dims,
                output_size=self.nn_output_dim,
                input_form=self.input_type,
                k_min=self.k_range[0],
                k_max=self.k_range[1],
                r_min=self.r_range[0],
                r_max=self.r_range[1],
                rmax_out=6,
                window="kaiser",
                dx=1,
                activation=self.nn_activation,
            )
        elif self.nn_type.lower().startswith("mlp"):
            self.model = MLP(
                hidden_layers=self.nn_hidden_dims,
                dropout_rates=self.nn_dropout_rates,
                weight_decay=self.nn_weight_decay,
                output_size=self.nn_output_dim,
                input_form=self.input_type,
                k_min=self.k_range[0],
                k_max=self.k_range[1],
                r_min=self.r_range[0],
                r_max=self.r_range[1],
                rmax_out=6,
                window="kaiser",
                dx=1,
                activation=self.nn_activation,
            )
        elif self.nn_type.lower().startswith("node"):
            self.model = NODE(
                hidden_layers=self.nn_hidden_dims,
                output_size=self.nn_output_dim,
                input_form=self.input_type,
                k_min=self.k_range[0],
                k_max=self.k_range[1],
                r_min=self.r_range[0],
                r_max=self.r_range[1],
                rmax_out=6,
                window="kaiser",
                dx=1,
                activation=self.nn_activation,
            )
        elif self.nn_type.lower().startswith("cnn"):
            self.model = CNN(
                hidden_layers=self.nn_hidden_dims,
                num_filters=self.nn_filter_sizes,
                kernel_sizes=self.nn_kernel_sizes,
                output_size=self.nn_output_dim,
                k_min=self.k_range[0],
                k_max=self.k_range[1],
                r_min=self.r_range[0],
                r_max=self.r_range[1],
                rmax_out=6,
                window="kaiser",
                dx=1,
                input_form=self.input_type,
                activation=self.nn_activation,
            )            
        elif self.nn_type.lower().startswith("het"):
            self.model = hetMLP(
                hidden_layers=self.nn_hidden_dims,
                output_size=self.nn_output_dim,
                k_min=self.k_range[0],
                k_max=self.k_range[1],
                r_min=self.r_range[0],
                r_max=self.r_range[1],
                rmax_out=6,
                window="kaiser",
                dx=1,
                input_form=self.input_type,
                activation=self.nn_activation,
            )

    def create_checkpoint_callback(
        self,
        monitor: str = "val/loss_best",
        save_top_k: int = 1,
        mode: str = "min",
    ):
        return ModelCheckpoint(
            monitor=monitor,
            dirpath=self.checkpoint_dir,
            filename=f"{self.title}-{{epoch:02d}}",
            save_top_k=save_top_k,
            mode=mode,
        )

    def train(self, save_checkpoint=True):
        if not self.training_mode:
            raise ValueError(
                "Training mode is not enabled. Please set genera.mode to training"
            )

        if (
            not hasattr(self, "synthetic_spectra") or self.synthetic_spectra is None
        ):  # noqa
            self.init_synthetic_spectra()

        if not hasattr(self, "data_loader") or self.data_loader is None:
            self.prepare_dataloader()

        if not hasattr(self, "model") or self.model is None:
            self.init_model()

        self.model = torch.compile(self.model)
        checkpoint_callback = self.create_checkpoint_callback()
        
        # Create early stopping callback
        early_stop_callback = EarlyStopping(
            monitor='val/loss',
            min_delta=0.00,
            patience=1000,
            verbose=True,
            mode='min'
        )

        history_logger = HistoryLogger(self.history)
        callbacks = [checkpoint_callback, early_stop_callback, history_logger]  # Add early stopping to callbacks

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            callbacks=callbacks
        )
        
        trainer.fit(
            self.model,
            self.data_loader.train_dataloader(batch_size=self.batch_size),
            self.data_loader.val_dataloader(batch_size=self.batch_size),
        )

        best_checkpoint = checkpoint_callback.best_model_path
        self.checkpoint_dir = os.path.dirname(best_checkpoint)
        self.checkpoint_name = os.path.basename(best_checkpoint)

        if save_checkpoint:
            self.save_config()

        return self

    def predict_and_denormalize(self, spectrum: torch.Tensor):
        if not hasattr(self, "model") or self.model is None:
            self.load_model()

        if not hasattr(self, "norm_params_spectra") or self.norm_params_spectra is None:
            raise ValueError("Normalization parameters are not available")

        if (
            not hasattr(self, "norm_params_parameters")
            or self.norm_params_parameters is None
        ):
            raise ValueError("Normalization parameters are not available")

        self.model.eval()

        with torch.no_grad():
            self.test_pred = self.model(spectrum.to(self.model.device))

        self.denormalized_test_pred = self.denormalize_data(self.test_pred)

        return self

    def predict_and_denormalize_BNN(self, spectrum: torch.Tensor):
        if not hasattr(self, "model") or self.model is None:
            self.load_model()

        if not hasattr(self, "norm_params_spectra") or self.norm_params_spectra is None:
            raise ValueError("Normalization parameters are not available")

        if (
            not hasattr(self, "norm_params_parameters")
            or self.norm_params_parameters is None
        ):
            raise ValueError("Normalization parameters are not available")

        self.model.eval()

        self.denormalized_test_pred = predict_with_uncertainty(self.model, spectrum, self.norm_params_parameters, n_samples=100)

        return self

    def predict_and_denormalize_hetMLP(self, spectrum: torch.Tensor):
        if not hasattr(self, "model") or self.model is None:
            self.load_model()

        if not hasattr(self, "norm_params_spectra") or self.norm_params_spectra is None:
            raise ValueError("Normalization parameters are not available")

        if (
            not hasattr(self, "norm_params_parameters")
            or self.norm_params_parameters is None
        ):
            raise ValueError("Normalization parameters are not available")

        self.model.eval()

        self.denormalized_test_pred = predict_with_uncertainty_hetMLP(self.model, spectrum, self.norm_params_parameters)

        return self

    def denormalize_data(self, data: torch.Tensor | np.ndarray):

        if not hasattr(self, "norm_params_spectra") or self.norm_params_spectra is None:
            raise ValueError("Normalization parameters are not available")

        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()

        return denormalize_data(data, self.norm_params_parameters)

    def denormalize_spectra(self, spectra: torch.Tensor | np.ndarray):

        if not hasattr(self, "norm_params_spectra") or self.norm_params_spectra is None:
            raise ValueError("Normalization parameters are not available")

        if isinstance(spectra, torch.Tensor):
            spectra = spectra.cpu().detach().numpy()

        return denormalize_spectra(spectra, self.norm_params_spectra)

    def validate_model(self):

        if not hasattr(self, "synthetic_spectra") or self.synthetic_spectra is None:
            self.load_synthetic_spectra()

        if not hasattr(self, "data_loader") or self.data_loader is None:
            self.prepare_dataloader()

        if not hasattr(self, "model") or self.model is None:
            self.load_model()
        network_type = self.config["neural_network"]["architecture"]["type"]
        if network_type.lower().startswith("node"):
            self.x_test, self.y_test, _ = next(iter(self.data_loader.test_dataloader()))
        else:
            self.x_test, self.y_test = next(iter(self.data_loader.test_dataloader()))

        self.model.eval()

        self.node_test_pred_list = []

        with torch.no_grad():
            if network_type.lower().startswith("node"):
                self.test_pred = self.model(self.x_test.to(self.model.device))
            else:
                self.test_pred = self.model(self.x_test.to(self.model.device))

        if self.nn_type.lower().startswith("het"):
            self.test_pred_mu, self.test_pred_sigma = self.test_pred # test_pred is a tuple for hetMLP
            self.denormalized_x_test = self.denormalize_spectra(self.x_test)
            self.denormalized_y_test = self.denormalize_data(self.y_test)
            self.denormalized_test_pred = self.denormalize_data(self.test_pred_mu)
        if self.nn_type.lower().startswith("node"):
            print("node")
        else:
            self.denormalized_x_test = self.denormalize_spectra(self.x_test)
            self.denormalized_y_test = self.denormalize_data(self.y_test)
            self.denormalized_test_pred = self.denormalize_data(self.test_pred)

        return self

    def plot_parity(self, save_path="/parity.png"): 

        save_path = os.path.join(self.config_dir, self.output_dir) + save_path 

        parameter_name_dict = {0: "A", 1: "$\Delta R (Å)$", 2: "$\sigma^2 (Å^{2})$", 3: "$E_0$ (eV)"}

        if not hasattr(self, "denormalized_y_test") or not hasattr(
            self, "denormalized_test_pred"
        ):
            self.validate_model()

        num_parameters = self.denormalized_y_test.shape[1]

        # Plot predicted vs. true values for each parameter in a single row
        fig, axs = plt.subplots(2, 2, figsize=(5, 5))
        axs = axs.flatten()

        for i in range(num_parameters):
            axs[i].scatter(
                self.denormalized_y_test[:, i],
                self.denormalized_test_pred[:, i],
                alpha=0.5,
            )
            axs[i].plot(
                [
                    self.denormalized_y_test[:, i].min(),
                    self.denormalized_y_test[:, i].max(),
                ],
                [
                    self.denormalized_y_test[:, i].min(),
                    self.denormalized_y_test[:, i].max(),
                ],
                "r--",
            )
            axs[i].set_xlabel("True Values", fontsize=12)
            axs[i].set_ylabel("Predicted Values", fontsize=12)
            axs[i].set_title(f"{parameter_name_dict[i]}", fontsize=12)
            # make font sizes 12
            axs[i].tick_params(axis='both', which='major', labelsize=12)
            axs[i].tick_params(axis='both', which='minor', labelsize=12)


        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path)

        return fig


    def plot_parity2(self, save_path="/parity.png"): # for sequence training data


        # Construct full save path
        save_path = os.path.join(self.config_dir, self.output_dir) + save_path

        parameter_name_dict = {0: "A", 1: "deltar", 2: "sigma2", 3: "e0"}

        # Extract test data and predictions
        y_test_data = self.y_test  # Shape: (64, 20, 4)
        print(f"y_test_data shape: {y_test_data.shape}")
        y_pred_data = self.test_pred  # Shape: (64, 20, 4)
        print(f"y_pred_data shape: {y_pred_data.shape}")

        # Convert tensors to CPU numpy arrays if necessary
        def to_cpu_numpy(data):
            if isinstance(data, torch.Tensor):
                return data.cpu().detach().numpy()
            return data

        def process_data(data):
            # Convert to CPU numpy array
            data_array = to_cpu_numpy(data)
            # Flatten to [1280, 4]
            if len(data_array.shape) == 3:  # Ensure it's [samples, sequences, parameters]
                data_array = data_array.reshape(-1, data_array.shape[-1])  # Flatten to [samples * sequences, parameters]
            return data_array

        y_test_data = process_data(y_test_data)  # Shape: (1280, 4)
        y_pred_data = process_data(y_pred_data)  # Shape: (1280, 4)

        # Check for mismatched shapes
        if y_test_data.shape != y_pred_data.shape:
            print(f"Mismatched shapes: y_test_data {y_test_data.shape}, y_pred_data {y_pred_data.shape}")
            return None

        num_parameters = y_test_data.shape[1]
        print(f"Plotting parity plot for {num_parameters} parameters.")
        print(f"data size: {y_test_data.shape}, pred size: {y_pred_data.shape}")

        # Plot predicted vs. true values for each parameter
        fig, axs = plt.subplots(1, num_parameters, figsize=(5 * num_parameters, 5))
        if num_parameters == 1:
            axs = [axs]

        for i in range(num_parameters):
            x_data = y_test_data[:, i]
            y_data = y_pred_data[:, i]

            axs[i].scatter(x_data, y_data, alpha=0.5)
            min_val = min(x_data.min(), y_data.min())
            max_val = max(x_data.max(), y_data.max())
            axs[i].plot([min_val, max_val], [min_val, max_val], "r--")
            axs[i].set_xlabel("True Values")
            axs[i].set_ylabel("Predicted Values")
            param_name = parameter_name_dict.get(i, f"Param {i}")
            axs[i].set_title(f"Parameter {param_name}: Predicted vs. True")

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path)
            print(f"Parity plot saved to {save_path}")

        return fig



    def plot_test_spectra(self, index, save_path="/spectra.png"):

        save_path = os.path.join(self.config_dir, self.output_dir) + save_path

        if not hasattr(self, "denormalized_y_test") or not hasattr(
            self, "denormalized_test_pred"
        ):
            self.validate_model()

        A_pred, deltar_pred, sigma2_pred, e0_pred = self.denormalized_test_pred[index]
        A_true, deltar_true, sigma2_true, e0_true = self.denormalized_y_test[index]

        path_predicted = feffpath(self.feff_path_file)
        path_predicted.s02 = 1
        path_predicted.degen = A_pred
        path_predicted.deltar = deltar_pred
        path_predicted.sigma2 = sigma2_pred
        path_predicted.e0 = e0_pred
        path2chi(path_predicted)

        kweight = self.k_weight
        kmin = self.k_range[0]
        kmax = self.k_range[1]
        rmin = self.r_range[0]
        rmax = self.r_range[1]

        xftf(path_predicted, kweight=kweight, kmin=kmin, kmax=kmax)
        xftr(path_predicted, rmin=rmin, rmax=rmax)

        k_grid = np.arange(kmin, kmax, 0.05)

        interpolated_predicted = np.interp(
            k_grid, path_predicted.q, path_predicted.chiq_re
        )

        path_true = feffpath(self.feff_path_file)
        path_true.s02 = 1
        path_true.degen = A_true
        path_true.deltar = deltar_true
        path_true.sigma2 = sigma2_true
        path_true.e0 = e0_true
        path2chi(path_true)

        xftf(path_true, kweight=kweight, kmin=kmin, kmax=kmax)
        xftr(path_true, rmin=rmin, rmax=rmax)

        interpolated_true = np.interp(k_grid, path_true.q, path_true.chiq_re)

        interpolated_chi = np.interp(
            k_grid, self.nn_k_grid, self.denormalized_x_test[index]
        )

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        ax.plot(
            k_grid,
            interpolated_chi,
            linestyle="--",
            color="black",
            linewidth=2,
            label="$\chi$(k) True",
        )

        ax.plot(
            k_grid,
            interpolated_true,
            linestyle="--",
            color="blue",
            linewidth=2,
            label="$\chi$(q) True",
        )
        ax.plot(
            k_grid,
            interpolated_predicted,
            linestyle="-",
            color="red",
            linewidth=2,
            label="$\chi$(q) Predicted",
        )

        ax.set_xlabel(r"$ k \ \rm (\AA^{-1})$", fontsize=14)

        k_weight_str = "" if kweight == 1 else r"^{" + f"{kweight}" + "}"
        ax.set_ylabel(r"$ k" + k_weight_str + r"\chi(q)$", fontsize=14)
        ax.set_title(f"Spectrum {index}", fontsize=14)

        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        ax.legend(fontsize=12)

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path)

        return fig

    def load_exp_data(self, dataset_dir: str):
        data = read_ascii(dataset_dir)
        return data

    def process_exp_data(self, data):
        kmin = self.exp_krange[0]
        kmax = self.exp_krange[1]
        rmin = self.exp_rrange[0]
        rmax = self.exp_rrange[1]
        k_grid = self.config["neural_network"]["k_grid"] # this is actually a list of strings #TODO: make sure this is not an issue elsewhere in the code
        k_grid = np.array(k_grid, dtype=np.float32)
        k_weight = self.config["experiment"]["k_weight"]
        print("k_weight", k_weight)

        xftf(data, kweight=k_weight, kmin=kmin, kmax=kmax)
        xftr(data, rmin=rmin, rmax=rmax)

        if k_weight == 0:
            interpolated_chi_k = interpolate_spectrum(data.k, data.chi, k_grid)

        elif k_weight == 1:
            data.chi1 = data.chi * data.k
            interpolated_chi_k = interpolate_spectrum(data.k, data.chi1, k_grid)

        elif k_weight == 2:
            data.chi2 = data.chi * data.k ** 2
            interpolated_chi_k = interpolate_spectrum(data.k, data.chi2, k_grid)

        else:
            raise ValueError(f"Invalid k-weight, {k_weight}, type {type(k_weight)}")
        
        interpolated_chi_k = torch.tensor(interpolated_chi_k).unsqueeze(0)
        xftf(data, kweight=k_weight, kmin=kmin, kmax=kmax)
        xftr(data, rmin=rmin, rmax=rmax)
        interpolated_chi_q_re = interpolate_spectrum(data.q, data.chiq_re, k_grid)
        interpolated_chi_q_im = interpolate_spectrum(data.q, data.chiq_im, k_grid)

        #print("giving it the group")
        estimate_noise(data)
        noise_k = data.epsilon_k
        #print("noise_k", noise_k)
        noise_r = data.epsilon_r
        #print("noise_r", noise_r)


        return interpolated_chi_k, interpolated_chi_q_re, interpolated_chi_q_im, noise_r

    def run_predictions_S(self):
        dataset_dirs = self.config['experiment']['dataset_dir']
        artemis_results = self.config['artemis']['result']
        artemis_unc = self.config['artemis']['unc']
        dataset_names = self.config['experiment']['dataset_names']
        real_values = len(dataset_dirs)
        self.sequenced_exp_spectra = np.zeros((1, 20, 401))

        for i in range(len(dataset_dirs)):
            dataset_dir = dataset_dirs[i]
            interpolated_chi_k, interpolated_chi_q, noise_r = self.process_exp_data(self.load_exp_data(dataset_dir)) # this is the exp data in k and q
            interpolated_chi_k = torch.tensor(interpolated_chi_k).unsqueeze(0)

            # Normalize the data
            max_abs_val = self.norm_params_spectra["max_abs_val"]
            normalized_chi_k = interpolated_chi_k / max_abs_val
            
            self.sequenced_exp_spectra[0, i] = normalized_chi_k
        
        #print(self.sequenced_exp_spectra)
        #print(self.sequenced_exp_spectra.shape)

        # Perform prediction
        self.model.eval()
        with torch.no_grad():
            spectra_tensor = torch.tensor(self.sequenced_exp_spectra).to(self.model.device)
            self.exp_pred = self.model(spectra_tensor)
        
        #print(self.exp_pred)
        #print(self.exp_pred.shape)

        # get the real values which are the first real_values elements
        real_values = self.exp_pred[0][:real_values]
        # denormalize
        real_values = self.denormalize_data(real_values)
        self.denormalized_test_pred = real_values
    
        #print(real_values)

        self.predictions = []

        for i in range(len(dataset_dirs)):
            dataset_dir = dataset_dirs[i]
            artemis_result = artemis_results[i]
            artemis_unc_entry = artemis_unc[i]
            dataset_name = dataset_names[i]
            interpolated_chi_k, interpolated_chi_q, noise_r = self.process_exp_data(self.load_exp_data(dataset_dir)) # this is the exp data in k and q
            interpolated_chi_k = torch.tensor(interpolated_chi_k).unsqueeze(0)

            predicted_a, predicted_deltar, predicted_sigma2, predicted_e0 = self.denormalized_test_pred[i]
            a_unc, deltar_unc, sigma2_unc, e0_unc = [0,0,0,0]

            self.predictions.append({
                'predicted_a': predicted_a,
                'predicted_deltar': predicted_deltar,
                'predicted_sigma2': predicted_sigma2,
                'predicted_e0': predicted_e0,
                'uncertainty_a': a_unc,
                'uncertainty_deltar': deltar_unc,
                'uncertainty_sigma2': sigma2_unc,
                'uncertainty_e0': e0_unc,
                'interpolated_chi_k': interpolated_chi_k,
                'interpolated_chi_q': interpolated_chi_q,
                'artemis_result': artemis_result,
                'artemis_unc': artemis_unc_entry,
                'dataset_name': dataset_name
                })
        #print(self.predictions)
        

        return self.predictions

    def run_predictions(self):
        network_type = self.config["neural_network"]["architecture"]["type"]
        dataset_dirs = self.config['experiment']['dataset_dir']
        artemis_results = self.config['artemis']['result']
        artemis_unc = self.config['artemis']['unc']
        dataset_names = self.config['experiment']['dataset_names']

        self.predictions = []

        for i in range(len(dataset_dirs)):
            dataset_dir = dataset_dirs[i]
            artemis_result = artemis_results[i]
            artemis_unc_entry = artemis_unc[i]
            dataset_name = dataset_names[i]

            # Load and preprocess data for the current dataset_dir
            interpolated_chi_k, interpolated_chi_q_re, interpolated_chi_q_im, noise_r = self.process_exp_data(self.load_exp_data(dataset_dir)) # this is the exp data in k and q
            interpolated_chi_k = torch.tensor(interpolated_chi_k).unsqueeze(0)

            # Normalize the data
            max_abs_val = self.norm_params_spectra["max_abs_val"]
            normalized_chi_k = interpolated_chi_k / max_abs_val

            # Perform prediction
            if network_type == "BNN":
                self.predict_and_denormalize_BNN(normalized_chi_k[0])
                preds, uncs = self.denormalized_test_pred
                predicted_a, predicted_deltar, predicted_sigma2, predicted_e0 = preds
                a_unc, deltar_unc, sigma2_unc, e0_unc = uncs

            if network_type == "MLP":
                self.predict_and_denormalize(normalized_chi_k[0])
                predicted_a, predicted_deltar, predicted_sigma2, predicted_e0 = self.denormalized_test_pred[0]
                a_unc, deltar_unc, sigma2_unc, e0_unc = [0,0,0,0]

            if network_type == "NODE":
                self.predict_and_denormalize(normalized_chi_k[0])
                predicted_a, predicted_deltar, predicted_sigma2, predicted_e0 = self.denormalized_test_pred[0]
                a_unc, deltar_unc, sigma2_unc, e0_unc = [0,0,0,0]

            if network_type == "CNN":
                self.predict_and_denormalize(normalized_chi_k[0])
                predicted_a, predicted_deltar, predicted_sigma2, predicted_e0 = self.denormalized_test_pred[0]
                a_unc, deltar_unc, sigma2_unc, e0_unc = [0,0,0,0]

            if network_type == "hetMLP":
                self.predict_and_denormalize_hetMLP(normalized_chi_k[0])
                preds, uncs = self.denormalized_test_pred
                predicted_a, predicted_deltar, predicted_sigma2, predicted_e0 = preds[0]
                a_unc, deltar_unc, sigma2_unc, e0_unc = uncs[0]

            self.predictions.append({
                'predicted_a': predicted_a,
                'predicted_deltar': predicted_deltar,
                'predicted_sigma2': predicted_sigma2,
                'predicted_e0': predicted_e0,
                'uncertainty_a': a_unc,
                'uncertainty_deltar': deltar_unc,
                'uncertainty_sigma2': sigma2_unc,
                'uncertainty_e0': e0_unc,
                'interpolated_chi_k': interpolated_chi_k,
                'interpolated_chi_q_re': interpolated_chi_q_re,
                'interpolated_chi_q_im': interpolated_chi_q_im,
                'normalized_chi_k': normalized_chi_k[0],
                'artemis_result': artemis_result,
                'artemis_unc': artemis_unc_entry,
                'dataset_name': dataset_name,
                'noise_r': noise_r

            })

        return self.predictions

    def build_synth_spectra(self, predicted_params: list | np.ndarray, space_type: str = "re"):

        kmin = self.k_range[0]
        kmax = self.k_range[1]
        rmin = self.r_range[0]
        rmax = self.r_range[1]
        k_grid = self.config["neural_network"]["k_grid"] # this is actually a list of strings #TODO: make sure this is not an issue elsewhere in the code
        k_grid = np.array(k_grid, dtype=np.float32)
        k_weight = self.config["experiment"]["k_weight"]

        predicted_a, predicted_deltar, predicted_sigma2, predicted_e0 = predicted_params


        
        path_predicted = feffpath(self.feff_path_file)
        path_predicted.s02 = 1
        path_predicted.degen = predicted_a
        path_predicted.deltar = predicted_deltar
        path_predicted.sigma2 = predicted_sigma2
        path_predicted.e0 = predicted_e0
        path2chi(path_predicted)
        xftf(path_predicted, kweight=k_weight, kmin=kmin, kmax=kmax)
        xftr(path_predicted, rmin=rmin, rmax=rmax)

        interpolated_predicted_re = interpolate_spectrum(path_predicted.q, path_predicted.chiq_re, k_grid)
        interpolated_predicted_im = interpolate_spectrum(path_predicted.q, path_predicted.chiq_im, k_grid)
        if space_type == "re":
            interpolated_predicted = interpolated_predicted_re
        elif space_type == "im":
            interpolated_predicted = interpolated_predicted_im
        else:
            interpolated_predicted = [interpolated_predicted_re, interpolated_predicted_im]
        return interpolated_predicted

    def get_MSE_error(self, interpolated_artemis, interpolated_exp):
        
        kmin = self.k_range[0]
        kmax = self.k_range[1]
        k_grid = self.config["neural_network"]["k_grid"]
        k_grid = np.array(k_grid, dtype=np.float32)


        # get exp in range
        interpolated_exp_in_range = [i[1] for i in zip(k_grid, interpolated_exp) if kmin <= i[0] <= kmax]
        interpolated_exp_in_range = np.array(interpolated_exp_in_range)

        # MSE error between predicted and artemis with nano
        interpolated_artemis_in_range = [i[1] for i in zip(k_grid, interpolated_artemis) if kmin <= i[0] <= kmax]
        interpolated_artemis_in_range = np.array(interpolated_artemis_in_range)

        e2 = np.mean((interpolated_artemis_in_range - interpolated_exp_in_range)**2) / np.var(interpolated_exp_in_range)

        return e2



    def save_predictions_to_toml(self, filename):

        filename = os.path.join(self.config_dir, self.output_dir) + "/" + filename

        def tensor_to_list(tensor):
            # Convert the tensor to a NumPy array and then to a list
            #tensor_np = tensor.detach().cpu().numpy()  # Ensure tensor is on CPU and detached from the computation graph
            tensor_list = tensor.tolist()
            return tensor_list

        with open(filename, 'w') as file:
            for i, prediction in enumerate(self.predictions):
                file.write(f"[prediction_{i}]\n")
                file.write(f"dataset_name = \"{prediction['dataset_name']}\"\n")
                file.write(f"predicted_a = {prediction['predicted_a']}\n")
                file.write(f"predicted_deltar = {prediction['predicted_deltar']}\n")
                file.write(f"predicted_sigma2 = {prediction['predicted_sigma2']}\n")
                file.write(f"predicted_e0 = {prediction['predicted_e0']}\n")
                file.write(f"uncertainty_a = {prediction['uncertainty_a']}\n")
                file.write(f"uncertainty_deltar = {prediction['uncertainty_deltar']}\n")
                file.write(f"uncertainty_sigma2 = {prediction['uncertainty_sigma2']}\n")
                file.write(f"uncertainty_e0 = {prediction['uncertainty_e0']}\n")
                file.write(f"interpolated_chi_k = {tensor_to_list(prediction['interpolated_chi_k'])}\n")
                file.write(f"interpolated_chi_q = {tensor_to_list(prediction['interpolated_chi_q_re'])}\n")
                file.write(f"artemis_result = \"{prediction['artemis_result']}\"\n")
                file.write(f"artemis_unc = \"{prediction['artemis_unc']}\"\n")
                file.write("\n")




    def plot_results(self, sequence = False):
        dataset_names = self.config["experiment"]["dataset_names"]
        if not sequence:
            predictions = self.run_predictions()
        else:
            predictions = self.run_predictions_S() # for sequence training data
        num_predictions = len(predictions)
        kmin = self.k_range[0]
        kmax = self.k_range[1]

        # ============================
        # Plot Parameters
        # ============================
        num_param_plots = 4
        fig_params, axs_params = plt.subplots(1, num_param_plots, figsize=(20, 4))

        # find min and max artemis values
        all_artemis = [pred['artemis_result'] for pred in predictions]
        all_artemis = np.array(all_artemis)
        min_artemis = np.min(all_artemis, axis=0)
        max_artemis = np.max(all_artemis, axis=0)

        network_type = self.config["neural_network"]["architecture"]["type"]
        if network_type == "MLP" or network_type == "NODE":
            print("Calculating MLP uncertainties")
            self.get_MLP_uncertainty()
            for idx,pred in enumerate(predictions):
                pred['uncertainty_a'], pred['uncertainty_deltar'], pred['uncertainty_sigma2'], pred['uncertainty_e0'] = self.mlp_uncertainties[idx]
            # dumb lazy way to add uncertainties to the predictions should make this more clear at some point.
            for idx,pred in enumerate(predictions):
                self.predictions[idx]['uncertainty_a'], self.predictions[idx]['uncertainty_deltar'], self.predictions[idx]['uncertainty_sigma2'], self.predictions[idx]['uncertainty_e0'] = self.mlp_uncertainties[idx]
       

        colors = cm.rainbow(np.linspace(0, 1, num_predictions))

        for idx, (pred, color) in enumerate(zip(predictions, colors)):
            # Plot Delta A
            axs_params[0].errorbar(
                pred['predicted_a'], pred['artemis_result'][0],
                xerr=pred['uncertainty_a'], yerr=pred['artemis_unc'][0],
                fmt='o', label=dataset_names[idx], color=color
            )
            axs_params[0].plot(
                [min_artemis[0]-1, max_artemis[0]+1],
                [min_artemis[0]-1, max_artemis[0]+1],
                'r--'
            )
            axs_params[0].set_title('A')
            axs_params[0].set_xlabel('NN')
            axs_params[0].set_ylabel('Artemis')
            axs_params[0].tick_params(axis='both', which='major', labelsize=8)

            # Plot Delta R
            axs_params[1].errorbar(
                pred['predicted_deltar'], pred['artemis_result'][1],
                xerr=pred['uncertainty_deltar'], yerr=pred['artemis_unc'][1],
                fmt='o', label=dataset_names[idx], color=color
            )
            axs_params[1].plot(
                [min_artemis[1]-0.1, max_artemis[1]+0.1],
                [min_artemis[1]-0.1, max_artemis[1]+0.1],
                'r--'
            )
            axs_params[1].set_title('Delta R')
            axs_params[1].set_xlabel('NN')
            axs_params[1].set_ylabel('Artemis')
            axs_params[1].tick_params(axis='both', which='major', labelsize=8)

            # Plot Sigma2
            axs_params[2].errorbar(
                pred['predicted_sigma2'], pred['artemis_result'][2],
                xerr=pred['uncertainty_sigma2'], yerr=pred['artemis_unc'][2],
                fmt='o', label=dataset_names[idx], color=color
            )
            axs_params[2].plot(
                [min_artemis[2]-0.01, max_artemis[2]+0.01],
                [min_artemis[2]-0.01, max_artemis[2]+0.01],
                'r--'
            )
            axs_params[2].set_title('$\sigma^{2}$')
            axs_params[2].set_xlabel('NN')
            axs_params[2].set_ylabel('Artemis')
            axs_params[2].tick_params(axis='both', which='major', labelsize=8)

            # Plot E0
            axs_params[3].errorbar(
                pred['predicted_e0'], pred['artemis_result'][3],
                xerr=pred['uncertainty_e0'], yerr=pred['artemis_unc'][3],
                fmt='o', label=dataset_names[idx], color=color
            )
            axs_params[3].plot(
                [-10, 10], [-10, 10], 'r--'
            )
            axs_params[3].set_title('$\Delta$E0')
            axs_params[3].set_xlabel('NN')
            axs_params[3].set_ylabel('Artemis')
            axs_params[3].tick_params(axis='both', which='major', labelsize=8)

        axs_params[0].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.config_dir, self.output_dir) +'/parameters.png')
        print("Parameters plot saved")
        plt.close(fig_params)  # Close the figure to free memory

        # ============================
        # Plot Q-space
        # ============================
        # Determine grid size based on number of predictions
        cols_q = num_predictions  # Number of columns in the grid
        rows_q = math.ceil(num_predictions / cols_q)

        fig_q, axs_q = plt.subplots(rows_q, cols_q, figsize=(5 * cols_q, 4 * rows_q))
        axs_q = axs_q.flatten() if num_predictions > 1 else [axs_q]

        for idx, (pred, color) in enumerate(zip(predictions, colors)):
            ax = axs_q[idx]
            k_grid = self.config["neural_network"]["k_grid"]
            k_grid = np.array(k_grid, dtype=np.float32) # TODO fix this

            interpolated_chi_q = pred['interpolated_chi_q_re']
            interpolated_artemis = self.build_synth_spectra(pred['artemis_result'], space_type="re")
            predicted_parameter_array = [pred['predicted_a'], pred['predicted_deltar'], pred['predicted_sigma2'], pred['predicted_e0']]
            interpolated_nn = self.build_synth_spectra(predicted_parameter_array, space_type="re")

            mse_error_artemis = self.get_MSE_error(interpolated_artemis, interpolated_chi_q)
            mse_error_nn = self.get_MSE_error(interpolated_nn, interpolated_chi_q)


            network_type = self.config["neural_network"]["architecture"]["type"]

            ax.plot(k_grid, interpolated_chi_q, color=color, label='Exp')
            ax.plot(
                k_grid, interpolated_artemis,
                label=f'Artemis @ {dataset_names[idx]} MSE = {mse_error_artemis:.4f}', color="black"
            )
            ax.plot(
                k_grid, interpolated_nn,
                label=f'{network_type} @ {dataset_names[idx]} MSE = {mse_error_nn:.4f}', color="gray"
            )
            ax.set_xlim(kmin, kmax)
            ax.set_title(f'Q-space Prediction {idx + 1}')
            ax.legend()
            ax.set_xlabel('k')
            ax.set_ylabel(r'$Re[\chi(q)] \ (\mathrm{\AA}^{-2})$')
            ax.tick_params(axis='both', which='major', labelsize=8)

        # Remove any unused subplots
        for idx in range(num_predictions, rows_q * cols_q):
            fig_q.delaxes(axs_q[idx])

        plt.tight_layout()
        plt.savefig(os.path.join(self.config_dir, self.output_dir) +'/qspace.png')
        plt.close(fig_q)

    def plot_training_history(self, save_path="/training_history.png"):
            
            save_path = os.path.join(self.config_dir, self.output_dir) + save_path

            """Plot training and validation loss curves using log scale"""
            plt.figure(figsize=(10, 6))
            plt.plot(np.log(self.history["train/loss"]), label="Training Loss")
            plt.plot(np.log(self.history["val/loss"]), label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Log Loss")
            plt.title("Training History (Log Scale)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            
            return plt.gcf()

 

    # def monte_carlo_uncertainty(self, EXAFS_experiment, predicted_params, num_samples=1000, perturb_scale=0.05):
    #     sampled_params = np.random.normal(loc=predicted_params, 
    #                                     scale = [perturb_scale * abs(p) for p in predicted_params], 
    #                                     size=(num_samples, len(predicted_params)))
    #     mse_list = []
    #     for params in sampled_params:
    #         synth_EXAFS = self.build_synth_spectra(params)
    #         mse = self.get_MSE_error(EXAFS_experiment, synth_EXAFS)
    #         mse_list.append(mse)
    #     return np.mean(mse_list), np.std(mse_list)



    def visualize_uncertainty_estimation2(self, prediction, param1_name, param2_name, scale_factor=0.1):
        """
        Visualizes the effect of parameter perturbations on the EXAFS spectrum and the objective function landscape.

        Args:
            prediction (dict): A dictionary containing the predicted parameters and the interpolated EXAFS spectrum.
                            It should have keys like 'predicted_a', 'predicted_deltar', etc., and 'interpolated_chi_q'.
            param1_name (str): The name of the first parameter to visualize (e.g., 'predicted_a').
            param2_name (str): The name of the second parameter to visualize (e.g., 'predicted_deltar').
            scale_factor (float): The scaling factor for parameter perturbations.
        """

        # 1. Parameter Perturbation and Resulting Spectra
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)

        # Original spectrum
        original_params = [prediction['predicted_a'], prediction['predicted_deltar'], 
                        prediction['predicted_sigma2'], prediction['predicted_e0']]
        original_spectrum = self.build_synth_spectra(original_params, space_type='re')
        k_values = self.nn_k_grid
        k_values = np.array(k_values, dtype=np.float32)

        plt.plot(k_values, original_spectrum, label='Original', linewidth=2)

        # Perturb parameters and plot spectra
        param1_idx = ['predicted_a', 'predicted_deltar', 'predicted_sigma2', 'predicted_e0'].index(param1_name)
        param2_idx = ['predicted_a', 'predicted_deltar', 'predicted_sigma2', 'predicted_e0'].index(param2_name)
        
        param1_val = prediction[param1_name]
        param2_val = prediction[param2_name]

        epsilon1 = scale_factor * (abs(param1_val) if abs(param1_val) > 1e-8 else 1e-6)
        epsilon2 = scale_factor * (abs(param2_val) if abs(param2_val) > 1e-8 else 1e-6)

        for sign in [-1, 1]:
            print("og ", original_params)
            perturbed_params = original_params.copy()
            
            perturbed_params[param1_idx] += sign * epsilon1
            print("perturbed ", perturbed_params)
            perturbed_spectrum = self.build_synth_spectra(perturbed_params)
            plt.plot(k_values, perturbed_spectrum, linestyle='--', label=f'{param1_name} {"+" if sign == 1 else "-"}{epsilon1:.2e}')

            perturbed_params = original_params.copy()
            perturbed_params[param2_idx] += sign * epsilon2
            perturbed_spectrum = self.build_synth_spectra(perturbed_params)
            plt.plot(k_values, perturbed_spectrum, linestyle=':', label=f'{param2_name} {"+" if sign == 1 else "-"}{epsilon2:.2e}')

        plt.xlabel('k (Wave Vector)')
        plt.ylabel('χ(k) (EXAFS Signal)')
        plt.title('EXAFS Spectra with Parameter Perturbations')
        plt.legend()

        # 2. Objective Function Landscape (Contour Plot)
        plt.subplot(1, 2, 2)

        param1_range = np.linspace(param1_val - 5 * epsilon1, param1_val + 5 * epsilon1, 30)
        param2_range = np.linspace(param2_val - 5 * epsilon2, param2_val + 5 * epsilon2, 30)
        
        param1_grid, param2_grid = np.meshgrid(param1_range, param2_range)
        mse_grid = np.zeros_like(param1_grid)

        for i in range(param1_grid.shape[0]):
            for j in range(param1_grid.shape[1]):
                perturbed_params = original_params.copy()
                perturbed_params[param1_idx] = param1_grid[i, j]
                perturbed_params[param2_idx] = param2_grid[i, j]
                mse_grid[i, j] = self.objective(prediction['interpolated_chi_q_re'], perturbed_params)

        contour = plt.contourf(param1_grid, param2_grid, mse_grid, levels=50, cmap='viridis')
        plt.colorbar(contour, label='MSE')
        plt.plot(param1_val, param2_val, 'ro', markersize=8, label='Predicted Values')

        plt.xlabel(param1_name)
        plt.ylabel(param2_name)
        plt.title('Objective Function Landscape')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.config_dir, self.output_dir) + f'/uncertainty_{param1_name}_{param2_name}.png')
        plt.close()

    def visualize_4d_objective_landscape(self, prediction, id, scale_factor=0.2):
        """
        Visualizes the 4D objective function landscape by creating 2D contour plots 
        for all pairs of parameters, holding the other two parameters constant at their predicted values.

        Args:
            prediction (dict): A dictionary containing the predicted parameters and the interpolated EXAFS spectrum.
            scale_factor (float): The scaling factor for parameter perturbations.
        """

        param_names = ['predicted_a', 'predicted_deltar', 'predicted_sigma2', 'predicted_e0']
        original_params = [prediction[name] for name in param_names]

        # artemis_a, artemis_r, artemis_s, artemis_e = prediction['artemis_result']
        # artemis_dict = {
        #     'predicted_a': artemis_a,
        #     'predicted_deltar': artemis_r,
        #     'predicted_sigma2': artemis_s,
        #     'predicted_e0': artemis_e
        # }

        # Generate all unique pairs of parameters
        param_pairs = list(itertools.combinations(param_names, 2))
        #artemis_pairs = list(itertools.combinations(artemis_dict.keys(), 2))

        plt.figure(figsize=(18, 12))

        plot_num = 1
        for pair in param_pairs:
            param1_name, param2_name = pair
            param1_idx = param_names.index(param1_name)
            param2_idx = param_names.index(param2_name)

            param1_val = prediction[param1_name]
            param2_val = prediction[param2_name]

            #artemis_param1 = artemis_dict[param1_name]
            #artemis_param2 = artemis_dict[param2_name]


            epsilon1 = scale_factor * (abs(param1_val) if abs(param1_val) > 1e-8 else 1e-6)
            epsilon2 = scale_factor * (abs(param2_val) if abs(param2_val) > 1e-8 else 1e-6)

            param1_range = np.linspace(param1_val - 5 * epsilon1, param1_val + 5 * epsilon1, 30)
            param2_range = np.linspace(param2_val - 5 * epsilon2, param2_val + 5 * epsilon2, 30)

            param1_grid, param2_grid = np.meshgrid(param1_range, param2_range)
            mse_grid = np.zeros_like(param1_grid)

            for i in range(param1_grid.shape[0]):
                for j in range(param1_grid.shape[1]):
                    perturbed_params = original_params.copy()
                    perturbed_params[param1_idx] = param1_grid[i, j]
                    perturbed_params[param2_idx] = param2_grid[i, j]
                    mse_grid[i, j] = self.objective(prediction['interpolated_chi_q_re'], perturbed_params)

            plt.subplot(2, 3, plot_num)  # 2 rows, 3 columns of plots
            contour = plt.contourf(param1_grid, param2_grid, mse_grid, levels=50, cmap='viridis')
            plt.colorbar(contour, label='MSE')
            plt.plot(param1_val, param2_val, 'ro', markersize=8, label='Predicted Values')
            #plt.plot(artemis_param1, artemis_param2, 'bo', markersize=8, label='Artemis Values')


            plt.xlabel(param1_name)
            plt.ylabel(param2_name)
            plt.title(f'Objective Landscape: {param1_name} vs {param2_name}')
            plt.legend()

            plot_num += 1

        plt.tight_layout()
        plt.savefig(os.path.join(self.config_dir, self.output_dir) + f'/uncertainty_{id}.png')
        plt.close()



    def visualize_uncertainty_of_A(self, prediction, uncertainties, id, scale_factor=0.2):
        """
        Visualizes the uncertainty of parameter 'A' along with a 2D slice of the objective landscape.
        """
        param_names = ['predicted_a', 'predicted_deltar', 'predicted_sigma2', 'predicted_e0']
        original_params = [prediction[name] for name in param_names]
        a_idx = param_names.index('predicted_a')
        uncertainty_a = uncertainties[a_idx]

        # Choose two parameters for the 2D slice (e.g., deltar and sigma2)
        param1_name, param2_name = 'predicted_deltar', 'predicted_sigma2' 
        param1_idx = param_names.index(param1_name)
        param2_idx = param_names.index(param2_name)
        param1_val = prediction[param1_name]
        param2_val = prediction[param2_name]

        # Ensure that e0 is fixed at its predicted value for the 2D slice
        e0_idx = param_names.index('predicted_e0')
        e0_val = prediction['predicted_e0']

        epsilon1 = scale_factor * (abs(param1_val) if abs(param1_val) > 1e-8 else 1e-6)
        epsilon2 = scale_factor * (abs(param2_val) if abs(param2_val) > 1e-8 else 1e-6)

        param1_range = np.linspace(param1_val - 5 * epsilon1, param1_val + 5 * epsilon1, 30)
        param2_range = np.linspace(param2_val - 5 * epsilon2, param2_val + 5 * epsilon2, 30)

        param1_grid, param2_grid = np.meshgrid(param1_range, param2_range)
        mse_grid = np.zeros_like(param1_grid)

        for i in range(param1_grid.shape[0]):
            for j in range(param1_grid.shape[1]):
                perturbed_params = original_params.copy()
                perturbed_params[param1_idx] = param1_grid[i, j]
                perturbed_params[param2_idx] = param2_grid[i, j]
                perturbed_params[a_idx] = original_params[a_idx]  # Keep 'A' fixed
                perturbed_params[e0_idx] = e0_val  # Keep 'E0' fixed
                mse_grid[i, j] = self.objective(prediction['interpolated_chi_q_re'], perturbed_params)

        # Create the plots
        fig, axes = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={'width_ratios': [3, 1]})

        # 2D Slice
        contour = axes[0].contourf(param1_grid, param2_grid, mse_grid, levels=50, cmap='viridis')
        plt.colorbar(contour, ax=axes[0], label='MSE')
        axes[0].plot(param1_val, param2_val, 'ro', markersize=8, label='Predicted Values')
        axes[0].set_xlabel("$\Delta$R")
        axes[0].set_ylabel('$\sigma^{2}$')
        axes[0].set_title('Objective Landscape around fixed A and $\Delta E_{0}$')
        # make fonts larger
        axes[0].tick_params(axis='both', which='major', labelsize=8)


        # Uncertainty of 'A'
        axes[1].errorbar(0, prediction['predicted_a'], yerr=uncertainty_a, fmt='ro', markersize=8, capsize=5, elinewidth=2)
        axes[1].set_xticks([])  # Remove x-ticks
        axes[1].set_ylabel('A')
        axes[1].set_title('Uncertainty of A')
        axes[1].set_xlim([-1, 1])

        plt.tight_layout()
        plt.savefig(os.path.join(self.config_dir, self.output_dir) + f'/uncertainty_A_{id}.png', dpi=300)
        plt.savefig(os.path.join(self.config_dir, self.output_dir) + f'/uncertainty_A_{id}.svg')
        plt.close()


    # def objective(self, EXAFS_experiment, params):
    #     # params are expected to be in the original scale
    #     synth_EXAFS = self.build_synth_spectra(params)
    #     mse = self.get_MSE_error(EXAFS_experiment, synth_EXAFS)
    #     return mse

    # def compute_hessian_with_numdifftools(self, EXAFS_experiment, params, sigma=None):
    #     """
    #     Uses numdifftools to compute the Hessian of the chi-squared objective 
    #     with respect to 'params'.
    #     """
    #     def wrapped(p):
    #         return self.objective_chisq(EXAFS_experiment, p, sigma=sigma)

    #     Hfun = nd.Hessian(wrapped, method='central', step=1e-4)
    #     return Hfun(params)

    def compute_hessian_with_numdifftools(self, EXAFS_experiment, params, bounds=None, sigma=None):
        """
        Uses numdifftools to compute the Hessian of the chi-squared objective 
        with respect to 'params', handling parameter transformations between
        free/internal and bounded/external parameter spaces.
        
        Parameters
        ----------
        EXAFS_experiment : object
            The experimental data object
        params : array-like
            The parameters in their external (bounded) form as numerical values
        bounds : list of tuples, optional
            List of (min, max) bounds for each parameter. If None, assumes unbounded
        sigma : array-like, optional
            Uncertainties in the data
            
        Returns
        -------
        numpy.ndarray
            The Hessian matrix computed at the given parameters
        """
        def setup_bounds(param_val, param_min, param_max):
            """Set up Minuit-style parameter transformation."""
            if param_min is None:
                param_min = -np.inf
            if param_max is None:
                param_max = np.inf
                
            # Convert external (bounded) parameter to internal (free) parameter
            if param_min == -np.inf and param_max == np.inf:
                internal_val = param_val
                from_internal = lambda val: val
            elif param_max == np.inf:
                internal_val = np.sqrt((param_val - param_min + 1.0)**2 - 1)
                from_internal = lambda val: param_min - 1.0 + np.sqrt(val*val + 1)
            elif param_min == -np.inf:
                internal_val = np.sqrt((param_max - param_val + 1.0)**2 - 1)
                from_internal = lambda val: param_max + 1 - np.sqrt(val*val + 1)
            else:
                internal_val = np.arcsin(2*(param_val - param_min)/(param_max - param_min) - 1)
                from_internal = lambda val: param_min + (np.sin(val) + 1) * (param_max - param_min) / 2.0
                
            if abs(internal_val) < 1e-15:  # tiny
                internal_val = 0.0
                
            return internal_val, from_internal
        
        # Handle case where no bounds are provided
        if bounds is None:
            bounds = [(None, None)] * len(params)
        
        # Convert external parameters to internal parameters
        internal_values = []
        transformations = []
        
        for val, (min_val, max_val) in zip(params, bounds):
            internal_val, transform = setup_bounds(val, min_val, max_val)
            internal_values.append(internal_val)
            transformations.append(transform)
        
        def wrapped(internal_p):
            # Convert internal parameters back to external parameters
            external_p = [transform(val) for transform, val in zip(transformations, internal_p)]
            return self.objective_chisq(EXAFS_experiment, external_p, sigma=sigma)
        
        Hfun = nd.Hessian(wrapped, method='central', step=1e-4)
        return Hfun(np.array(internal_values))

    # def objective_chisq(self, EXAFS_experiment, params, sigma=None): # attempt 1
    #     """
    #     Objective function returning the chi-squared:
    #         chi2 = sum( ((y_i - f_i)/sigma_i)^2 ).

    #     Parameters
    #     ----------
    #     EXAFS_experiment : array-like
    #         Experimental EXAFS data over k-range.
    #     params : list or array
    #         Parameter values for building the synthetic spectrum.
    #     sigma : array-like or float, optional
    #         Uncertainties in the experimental data. If None or scalar, 
    #         assume a constant sigma for all points.

    #     Returns
    #     -------
    #     chi2 : float
    #         The chi-squared value (not reduced).
    #     """
    #     EXAFS_experiment_re, EXAFS_experiment_im = EXAFS_experiment

    #     #print(f"EXAFS_experiment_re: {EXAFS_experiment_re}")
    #     #print(f"EXAFS_experiment_im: {EXAFS_experiment_im}")

    #     # 1) Build synthetic EXAFS
    #     synth_EXAFS_re, synth_EXAFS_im = self.build_synth_spectra(params, space_type='both')

    #     # 2) Get data in the k-range
    #     kmin, kmax = self.k_range
    #     k_grid = np.array(self.config["neural_network"]["k_grid"], dtype=np.float32)

    #     exp_re_in_range = [val for (k, val) in zip(k_grid, EXAFS_experiment_re) 
    #                     if kmin <= k <= kmax]
    #     exp_im_in_range = [val for (k, val) in zip(k_grid, EXAFS_experiment_im)
    #                     if kmin <= k <= kmax]
    #     synth_re_in_range = [val for (k, val) in zip(k_grid, synth_EXAFS_re) 
    #                       if kmin <= k <= kmax]
    #     synth_im_in_range = [val for (k, val) in zip(k_grid, synth_EXAFS_im)
    #                         if kmin <= k <= kmax]
        

    #     exp_re_in_range = np.array(exp_re_in_range)
    #     exp_im_in_range = np.array(exp_im_in_range)
    #     synth_re_in_range = np.array(synth_re_in_range)
    #     synth_im_in_range = np.array(synth_im_in_range)

    #     n_data = len(exp_re_in_range) # N
    #     n_varys = 4
    #     n_free = n_data - n_varys
    #     #print(f"kmin: {kmin}")
    #     #print(f"kmax: {kmax}")
    #     delta_k = kmax - kmin
    #     #print(f"delta_k: {delta_k}")
    #     delta_r = 3.34 - 1.5 # Please for the love of God don't forget to code this correctly #TODO: fix this
    #     #print(f"delta_r: {delta_r}")
    #     n_ind = 1 + int((2*delta_k*delta_r) / np.pi)
    #     #print(f"n_ind: {n_ind}")

    #     self.exp_obj_params.append([n_data, n_ind])
        

    def objective_chisq(self, EXAFS_experiment, params, sigma=None):

        measured_chi_k = EXAFS_experiment
        # covert tensor to numpy array
        measured_chi_k = measured_chi_k.detach().cpu().numpy()[0][0]
        #print(f"measured_chi_k: {measured_chi_k.shape}")

        # 1) Build synthetic EXAFS

        kmin = self.k_range[0]
        kmax = self.k_range[1]
        rmin = self.r_range[0]
        rmax = self.r_range[1]
        k_grid = self.config["neural_network"]["k_grid"] # this is actually a list of strings #TODO: make sure this is not an issue elsewhere in the code
        k_grid = np.array(k_grid, dtype=np.float32)
        k_weight = self.config["experiment"]["k_weight"]

        #print(f"params: {params}")

        predicted_a, predicted_deltar, predicted_sigma2, predicted_e0 = params
        
        path_predicted = feffpath(self.feff_path_file)
        path_predicted.s02 = 1
        path_predicted.degen = predicted_a
        path_predicted.deltar = predicted_deltar
        path_predicted.sigma2 = predicted_sigma2
        path_predicted.e0 = predicted_e0
        
        path2chi(path_predicted)

        # fig, ax = plt.subplots()
        # ax.plot(path_predicted.k, path_predicted.chi, label="path2chi")
        # ax.legend()
        # ax.set_xlim(kmin, kmax)
        # plt.savefig("/home/nick/Projects/crowpeas/tests/Ag_unc_test/path2chi.png")
        # plt.close()


        #print(f"path_predicted.k: {path_predicted.k.shape}")
        #print(f"path_predicted.chi: {path_predicted.chi.shape}")

        path_predicted.chi =  interpolate_spectrum(path_predicted.k, path_predicted.chi, k_grid)
        
        path_predicted.k = k_grid

        kmini = np.where(k_grid >= kmin)[0][0]
        kmaxi = np.where(k_grid <= kmax)[0][-1]

        # truncate data between k_min and k_max
        path_predicted.k = path_predicted.k[kmini:kmaxi]
        path_predicted.chi = path_predicted.chi[kmini:kmaxi]
        measured_chi_k = measured_chi_k[kmini:kmaxi]

        #print(f"path_predicted.k: {path_predicted.k.shape}")
        #print(f"path_predicted.chi: {path_predicted.chi.shape}")
        #print(f"measured_chi_k: {measured_chi_k.shape}")

        # fig, ax = plt.subplots()
        # ax.plot(path_predicted.k, path_predicted.chi, label=f"predicted params {params}")
        # ax.plot(path_predicted.k, measured_chi_k, label="measured")


        path_predicted.chi = measured_chi_k - path_predicted.chi

        # ax.plot(path_predicted.k, path_predicted.chi, label="diff")
        # ax.legend()

        # plt.savefig("/home/nick/Projects/crowpeas/tests/Ag_unc_test/path_predicted.png")
        # plt.close()



        #print(f"path_predicted.k: {path_predicted.k.shape}")
        #print(f"path_predicted.chi: {path_predicted.chi.shape}")

        xftf(path_predicted, kweight=k_weight, kmin=kmin, kmax=kmax)

        #print(f"path_predicted.chir: {path_predicted.chir}")
        #print(f'sigma {sigma}')

        path_predicted.chir = path_predicted.chir / sigma


        xftr(path_predicted, rmin=rmin, rmax=rmax)

        chiq = path_predicted.chiq

        chiq_in_range = [val for (k, val) in zip(k_grid, chiq) if kmin <= k <= kmax]
        chiq_in_range = np.array(chiq_in_range)

        # fig, ax = plt.subplots()
        # ax.plot(chiq_in_range)
        # plt.savefig("/home/nick/Projects/crowpeas/tests/Ag_unc_test/chiq_in_range.png")
        # plt.close()

        def realimag(arr):
            "return real array of real/imag pairs from complex array"
            return np.array([(i.real, i.imag) for i in arr]).flatten()
        
        concat_re_im = realimag(chiq_in_range)[::2]
        #save fig of concat_re_im
        # plt.plot(concat_re_im)
        # plt.savefig("/home/nick/Projects/crowpeas/tests/Ag_unc_test/concat_re_im.png")
        # plt.close()

        

        self.n_data = len(concat_re_im) # N
        self.n_varys = 4
        self.n_free = self.n_data - self.n_varys
        #print(f"kmin: {kmin}")
        #print(f"kmax: {kmax}")
        delta_k = kmax - kmin
        #print(f"delta_k: {delta_k}")
        delta_r = rmax - rmin
        #print(f"delta_r: {delta_r}")
        self.n_ind = 1 + int((2*delta_k*delta_r) / np.pi)

        self.chisqr = np.sum((concat_re_im)**2)

        #print(f"chi2: {chi2}")

        return self.chisqr



    # def compute_hessian_with_numdifftools(self, EXAFS_experiment, params, sigma=None, min_scale=1e-6):
    #     """
    #     Computes the Hessian of 'self.objective' at the given 'params' in original units
    #     using numdifftools, but internally normalizes parameters to improve numerical stability.

    #     Parameters
    #     ----------
    #     EXAFS_experiment : array-like
    #         Your experimental data.
    #     params : list or array
    #         The parameter values in their original units.
    #     min_scale : float
    #         Minimum absolute scale to use if a parameter is near zero.

    #     Returns
    #     -------
    #     hessian : 2D numpy array
    #         Hessian matrix in the original parameter units.
    #     """

    #     # 1) Figure out a per-parameter scaling factor
    #     #    (avoid zero-scale by enforcing a minimal scale)
    #     param_scales = [max(abs(p), min_scale) for p in params]

    #     # 2) Define a wrapper that UN-normalizes before calling the real objective
    #     def wrapped(norm_params, sigma=sigma):
    #         # Convert normalized -> original
    #         unnorm_params = [n * s for n, s in zip(norm_params, param_scales)]
    #         return self.objective_chisq(EXAFS_experiment, unnorm_params, sigma=sigma)

    #     # 3) Create normalized_params to pass into numdifftools
    #     normalized_params = [p / s for p, s in zip(params, param_scales)]

    #     # 4) Let numdifftools compute the Hessian in normalized space
    #     Hfun = nd.Hessian(wrapped, method='central')
    #     raw_hessian = Hfun(normalized_params)

    #     # 5) Rescale Hessian back to original units.
    #     #    For second derivatives, each entry scales as (scale_i * scale_j).
    #     n = len(params)
    #     hessian = np.zeros((n, n), dtype=float)
    #     for i in range(n):
    #         for j in range(n):
    #             hessian[i, j] = raw_hessian[i, j] * param_scales[i] * param_scales[j]

    #     return hessian


    def compute_hessian(self, func, EXAFS_experiment, params, scale_factor=0.01, max_scale=1.0):
        """
        Compute the Hessian matrix of a scalar function with respect to its parameters.
        'params' should be normalized parameters here. We'll scale them back to original units
        before each function call.
        """
        n = len(params)
        hessian = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                # Compute step sizes in normalized space
                epsilon_i = scale_factor * (abs(params[i]) if abs(params[i]) > 1e-8 else 1e-6)
                epsilon_j = scale_factor * (abs(params[j]) if abs(params[j]) > 1e-8 else 1e-6)

                # Create perturbed parameter sets in normalized space
                params_ij = params.copy()
                params_ip = params.copy()
                params_jp = params.copy()

                params_ij[i] += epsilon_i
                params_ij[j] += epsilon_j
                params_ip[i] += epsilon_i
                params_jp[j] += epsilon_j

                # Convert normalized parameters back to original scale before objective call
                def denorm(pars):
                    return [p * max_scale for p in pars]

                f_ij = func(EXAFS_experiment, denorm(params_ij))
                f_i = func(EXAFS_experiment, denorm(params_ip))
                f_j = func(EXAFS_experiment, denorm(params_jp))
                f_orig = func(EXAFS_experiment, denorm(params))

                hessian[i, j] = (f_ij - f_i - f_j + f_orig) / (epsilon_i * epsilon_j)

        return hessian

    # def compute_hessian_central(self, func, EXAFS_experiment, params,
    #                             scale_factor=0.01, min_step=1e-6, max_scale=1.0):
    #     """
    #     Compute the Hessian matrix of a scalar function with respect to its parameters
    #     using a central-difference approximation. Only the upper triangle is computed
    #     (including the diagonal), and then the Hessian is symmetrized to fill in the lower.

    #     Parameters
    #     ----------
    #     func : callable
    #         The objective function signature: func(EXAFS_experiment, params).
    #         'params' here are expected to be in original scale. We'll handle
    #         the normalization/denormalization inside this function.
    #     EXAFS_experiment : np.ndarray or similar
    #         Experimental data needed by func.
    #     params : list or np.ndarray
    #         Normalized parameters. We'll denormalize them for each function call.
    #     scale_factor : float
    #         Multiplier for the parameter-based step size, e.g., 0.01 (1%).
    #     min_step : float
    #         Minimum absolute step size, to avoid too-small steps if a parameter is near zero.
    #     max_scale : float
    #         Used to denormalize parameters inside each function call.
        
    #     Returns
    #     -------
    #     hessian : np.ndarray
    #         Approximation of the Hessian (n x n).
    #     """

    #     def denorm(pars):
    #         return [p * max_scale for p in pars]
        
    #     n = len(params)
    #     hessian = np.zeros((n, n), dtype=float)

    #     for i in range(n):
    #         # step size for parameter i in normalized space
    #         eps_i = scale_factor * max(abs(params[i]), min_step)

    #         for j in range(i, n):
    #             # step size for parameter j in normalized space
    #             eps_j = scale_factor * max(abs(params[j]), min_step)

    #             # We want:
    #             #   f(x + e_i + e_j), f(x + e_i - e_j),
    #             #   f(x - e_i + e_j), f(x - e_i - e_j).

    #             # Copy the base parameter set
    #             p_plus_plus = params.copy()
    #             p_plus_minus = params.copy()
    #             p_minus_plus = params.copy()
    #             p_minus_minus = params.copy()

    #             p_plus_plus[i]  += eps_i
    #             p_plus_plus[j]  += eps_j
    #             p_plus_minus[i] += eps_i
    #             p_plus_minus[j] -= eps_j
    #             p_minus_plus[i] -= eps_i
    #             p_minus_plus[j] += eps_j
    #             p_minus_minus[i] -= eps_i
    #             p_minus_minus[j] -= eps_j

    #             f_pp = func(EXAFS_experiment, denorm(p_plus_plus))
    #             f_pm = func(EXAFS_experiment, denorm(p_plus_minus))
    #             f_mp = func(EXAFS_experiment, denorm(p_minus_plus))
    #             f_mm = func(EXAFS_experiment, denorm(p_minus_minus))

    #             # Central difference for mixed partial derivative:
    #             # d^2 f / dx_i dx_j ≈ [f(x+ e_i + e_j) - f(x+ e_i - e_j)
    #             #                     - f(x- e_i + e_j) + f(x- e_i - e_j)] / [4*eps_i*eps_j]
    #             hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps_i * eps_j)

    #             # For the diagonal terms (i == j), that same formula simplifies to
    #             # a standard second derivative in one dimension. But the formula
    #             # is consistent for all pairs.

    #     # Symmetrize (since H should be symmetric if f is nicely behaved)
    #     # We only did j in [i..n), so fill in the lower triangle.
    #     for i in range(n):
    #         for j in range(i):
    #             hessian[i, j] = hessian[j, i]

    #     return hessian

    # def objective_reduced_chisq(self, EXAFS_experiment, params, sigma=None):
    #     """
    #     Objective function that returns the reduced chi-squared:
    #         chi2_reduced = (1 / dof) * sum( ((y_i - f_i)/sigma_i)^2 )
    #     where dof = N - p (p is number of free params, N is number of data points).
        
    #     Parameters
    #     ----------
    #     EXAFS_experiment : array-like
    #         Experimental EXAFS data over k-range.
    #     params : list or array
    #         Parameter values for which we build the synthetic spectrum.
    #     sigma : array-like or float, optional
    #         Uncertainties in the experimental data. If None or scalar, 
    #         we assume a constant sigma for all data points.

    #     Returns
    #     -------
    #     chi2_reduced : float
    #         The reduced chi-squared value.
    #     """
    #     # 1) Build synthetic EXAFS
    #     synth_EXAFS = self.build_synth_spectra(params)

    #     # 2) Restrict both data sets to the relevant k-range (similar to your get_MSE_error)
    #     kmin, kmax = self.k_range
    #     k_grid = np.array(self.config["neural_network"]["k_grid"], dtype=np.float32)

    #     exp_in_range = [val for (k, val) in zip(k_grid, EXAFS_experiment) 
    #                     if kmin <= k <= kmax]
    #     synth_in_range = [val for (k, val) in zip(k_grid, synth_EXAFS) 
    #                       if kmin <= k <= kmax]

    #     exp_in_range = np.array(exp_in_range)
    #     synth_in_range = np.array(synth_in_range)

    #     # 3) Number of data points and parameters
    #     N = len(exp_in_range)
    #     p = len(params)
    #     dof = max(1, N - p)  # avoid dividing by zero if N <= p

    #     # 4) If sigma is None, assume constant sigma = 1
    #     #    If sigma is a single float, apply it to every data point
    #     #    If sigma is an array, it should match exp_in_range in length
    #     if sigma is None:
    #         sigma = 1.0
    #     if np.isscalar(sigma):
    #         sigma_array = np.full_like(exp_in_range, fill_value=sigma, dtype=float)
    #     else:
    #         sigma_array = np.array(sigma, dtype=float)
        
    #     # 5) Compute reduced chi-squared
    #     #    sum of squared residuals, each normalized by sigma_i^2
    #     residuals = (synth_in_range - exp_in_range) / sigma_array
    #     chi2 = np.sum(residuals**2)
    #     chi2_reduced = chi2 / dof

    #     return chi2_reduced

    # def compute_covariance_matrix(self, EXAFS_experiment, params, sigma=None, lambda_reg=1e-6):
    #     """
    #     Example method to invert the Hessian to get a covariance matrix,
    #     with an optional small regularization term on the diagonal.
    #     If your objective is e.g. sum of squares, you might multiply
    #     by N/2 (depending on how you define it).
    #     """
    #     hessian = self.compute_hessian_with_numdifftools(EXAFS_experiment, params, sigma=sigma)
    #     print("hessian calculated")

    #     N = len(EXAFS_experiment)

    #     # (Optional) regularize if near-singular
    #     hessian_reg = hessian + np.eye(len(hessian)) * lambda_reg

    #     #cov_matrix = np.linalg.inv(hessian_reg) * N / 2.0
    #     cov_matrix = np.linalg.inv(hessian_reg) * 2

    #     return cov_matrix

    def compute_covariance_matrix(self, EXAFS_experiment, params, sigma=None, lambda_reg=1e-6):
        hessian = self.compute_hessian_with_numdifftools(EXAFS_experiment, params, sigma=sigma)
        
        # Check condition number before regularization
        cond_before = np.linalg.cond(hessian)
        print(f"Condition number before regularization: {cond_before}")
        
        # Examine eigenvalues
        eigenvals = np.linalg.eigvals(hessian)
        print(f"Eigenvalues before regularization: {eigenvals}")
        
        # Adaptive regularization based on matrix scale
        matrix_scale = np.max(np.abs(hessian))
        lambda_reg_adaptive = matrix_scale * 1e-6  # adjust this factor if needed
        
        # Regularize
        hessian_reg = hessian + np.eye(len(hessian)) * lambda_reg_adaptive
        
        # Check condition number after regularization
        cond_after = np.linalg.cond(hessian_reg)
        print(f"Condition number after regularization: {cond_after}")
        
        cov_matrix = np.linalg.inv(hessian_reg) # *2 might be needed
        
        # Enforce symmetry and positive diagonal
        cov_matrix = (cov_matrix + cov_matrix.T) / 2  # ensure symmetry
        np.fill_diagonal(cov_matrix, np.abs(np.diag(cov_matrix)))  # force positive diagonal
        
        return cov_matrix

    def compute_covariance_matrix_svd(self, EXAFS_experiment, params, sigma=None, rcond=1e-6):  # increased rcond
        """
        Compute covariance matrix using SVD for better handling of correlations.
        """
        # Compute Hessian
        hessian = self.compute_hessian_with_numdifftools(EXAFS_experiment, params, sigma=sigma)
        
        # Perform SVD
        U, s, Vh = np.linalg.svd(hessian, full_matrices=False)
        
        # Diagnostic information
        condition_number = s[0] / s[-1]
        singular_values = s
        
        # Find small singular values - more aggressive threshold
        threshold = rcond * s[0]  
        small_singvals = s < threshold
        
        # Create inverse with SVD, handling small singular values
        s_inv = np.zeros_like(s)
        s_inv[~small_singvals] = 1/s[~small_singvals]
        
        # Reconstruct covariance matrix
        cov_matrix = (U * s_inv) @ Vh * 2.0
        
        # Ensure symmetry and handle potential negative values
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        # Handle negative diagonal elements if any
        diag = np.diag(cov_matrix)
        if np.any(diag <= 0):
            print(f"Warning: Found {np.sum(diag <= 0)} negative diagonal elements")
            np.fill_diagonal(cov_matrix, np.abs(diag))
        
        # Collect diagnostic information
        diagnostics = {
            'condition_number': condition_number,
            'singular_values': singular_values,
            'num_small_singvals': np.sum(small_singvals),
            'rank': np.sum(~small_singvals),
            'rcond': rcond,
            'threshold': threshold
        }
        
        return cov_matrix, diagnostics

    def analyze_covariance_results(self, cov_matrix, diagnostics):
        """Helper function to analyze the results"""
        print(f"Matrix condition number: {diagnostics['condition_number']:.2e}")
        print(f"Effective rank: {diagnostics['rank']} out of {len(diagnostics['singular_values'])}")
        print(f"Number of small singular values: {diagnostics['num_small_singvals']}")
        print("\nSingular values:")
        for i, sv in enumerate(diagnostics['singular_values']):
            print(f"  σ_{i+1}: {sv:.2e}")
        
        # Compute correlation matrix more safely
        diag = np.diag(cov_matrix)
        correlation = np.zeros_like(cov_matrix)
        
        for i in range(len(cov_matrix)):
            for j in range(len(cov_matrix)):
                if diag[i] > 0 and diag[j] > 0:
                    correlation[i,j] = cov_matrix[i,j] / np.sqrt(diag[i] * diag[j])
                else:
                    correlation[i,j] = np.nan
        
        print("\nCorrelation matrix:")
        print(correlation)
        return correlation

    # def compute_hessian_with_numdifftools(self, EXAFS_experiment, params, sigma=None, min_scale=1e-6):
    #     """
    #     Computes the Hessian of 'self.objective_chisq' at 'params' using numdifftools,
    #     but includes parameter NORMALIZATION internally to avoid ill-conditioning.

    #     Returns the Hessian in ORIGINAL (unnormalized) parameter units.
    #     """
    #     # 1) Determine scale for each parameter so they are ~O(1)
    #     param_scales = [max(abs(p), min_scale) for p in params]

    #     # 2) Define wrapper that expects normalized params, then un-normalizes them
    #     def wrapped(norm_params):
    #         # Convert normalized -> original
    #         unnorm_params = [n * s for n, s in zip(norm_params, param_scales)]
    #         return self.objective_chisq(EXAFS_experiment, unnorm_params, sigma=sigma)

    #     # 3) Create normalized_params from the original ones
    #     normalized_params = [p / s for p, s in zip(params, param_scales)]

    #     # 4) Use numdifftools on the WRAPPED function
    #     Hfun = nd.Hessian(wrapped, method='central')
    #     raw_hessian = Hfun(normalized_params)

    #     # 5) Rescale the Hessian to original units
    #     n = len(params)
    #     hessian = np.zeros((n, n), dtype=float)
    #     for i in range(n):
    #         for j in range(n):
    #             hessian[i, j] = raw_hessian[i, j] * param_scales[i] * param_scales[j]

    #     return hessian

    # def compute_covariance_matrix(self, EXAFS_experiment, params, sigma=None, lambda_reg=1e-6):
    #     """
    #     Invert the Hessian from the chi-squared objective. The resulting inverse 
    #     is the approximate covariance matrix (if your data uncertainties are 
    #     properly accounted for).
    #     """
    #     hessian = self.compute_hessian_with_numdifftools(EXAFS_experiment, params, sigma=sigma)

    #     # Regularize to handle near-singular cases
    #     hessian_reg = hessian + np.eye(len(params)) * lambda_reg
    #     cov_matrix = np.linalg.inv(hessian_reg)

    #     return cov_matrix

    def get_MLP_uncertainty(self, sequence = False):
        if not sequence:
            print("Calculating MLP uncertainties")
            predictions = self.run_predictions()
        else:
            predictions = self.run_predictions_S()
        
        self.mlp_uncertainties = []
        self.init_obj_params()

        for idx, pred in enumerate(predictions):
            #interpolated_chi_q_re = pred['interpolated_chi_q_re']
            #interpolated_chi_q_im = pred['interpolated_chi_q_im']
            interpolated_chi_k = pred['interpolated_chi_k']
            predicted_parameter_array = [
                pred['predicted_a'], 
                pred['predicted_deltar'], 
                pred['predicted_sigma2'], 
                pred['predicted_e0']
            ]

            # # # Normalize parameters
            # # max_scale = max(abs(p) for p in predicted_parameter_array)
            # # if max_scale < 1e-8:
            # #     max_scale = 1e-6
            # # normalized_params = [p / max_scale for p in predicted_parameter_array]

            # max_scale = 1

            # #print("Normalized Parameters:", normalized_params)

            # obj_func = lambda EXAFS_experiment, params: self.objective_chisq(EXAFS_experiment = EXAFS_experiment, sigma = pred["noise_k"], params = params)


            # # Compute Hessian in normalized space
            # normalized_hessian = self.compute_hessian(
            #     obj_func, 
            #     interpolated_chi_k, 
            #     #normalized_params,
            #     predicted_parameter_array, 
            #     scale_factor=.0001, 
            #     max_scale=max_scale
            # )



            # # The Hessian is currently in terms of normalized parameters.
            # # To convert back to the original parameters, multiply by (max_scale^2)
            # # because Hessian second derivatives scale as 1/(units_of_params^2).
            # hessian_rescaled = normalized_hessian * (max_scale ** 2)
            # print(f"max_scale = {max_scale}")

            # # Regularize the Hessian for numerical stability
            # lambda_reg = 1e-10
            # regularized_hessian = hessian_rescaled + np.eye(len(hessian_rescaled)) * lambda_reg

            # parameters used in objective function
            #print(self.exp_obj_params[idx])
            #n_fit, n_ind = self.exp_obj_params[idx]

            # Compute covariance matrix (inverse of Hessian)
            #covariance_matrix = (n_fit/n_ind) * np.linalg.inv(regularized_hessian)
            #covariance_matrix = np.linalg.inv(regularized_hessian) * self.err_scale
            #covariance_matrix = np.linalg.inv(regularized_hessian)

            #uvars = uncertainties.correlated_values(predicted_parameter_array, covariance_matrix)
            #print(f"Uncertainties (Prediction {idx}):", uvars)

            covariance_matrix = self.compute_covariance_matrix(interpolated_chi_k, predicted_parameter_array, sigma = pred["noise_r"])
            print(f"covariance_matrix: {covariance_matrix}")

            # Compute covariance matrix with diagnostics
            #covariance_matrix, diagnostics = self.compute_covariance_matrix_svd(interpolated_chi_k, predicted_parameter_array, sigma=pred["noise_k"])

            # Analyze the results
            #self.analyze_covariance_results(covariance_matrix, diagnostics)

            self.objective_chisq(EXAFS_experiment = interpolated_chi_k, sigma = pred["noise_r"], params = predicted_parameter_array) # run one time to get self.err_scale

            #print(f"residuals: {residuals}")
            self.chi2 = (self.n_ind/self.n_data) * self.chisqr * 1/self.n_free

            self.redchi = self.chi2/(self.n_ind - self.n_varys) 
            #err_scale = reduced_chi2*(n_data - n_varys)/(n_ind - n_varys)
            err_scale = self.redchi*(self.n_free)/(self.n_ind - self.n_varys)
            self.err_scale = err_scale



            print(f"chi2: {self.chi2}")
            print(f"redchi: {self.redchi}")

            print(f"err_scale = {self.err_scale}")

            # Extract uncertainties as sqrt of diagonal of covariance
            uncertainty = np.sqrt(np.diag(covariance_matrix) * self.err_scale)
            self.mlp_uncertainties.append(uncertainty)
            
            print(f"Uncertainties (Prediction {idx}):", uncertainty)

            # Debug thing
            #  # if uncertainty is too large, make it zero. too large is more than 80% of the predicted value
            # for i in range(len(uncertainty)):
            #     if uncertainty[i] > 0.8 * predicted_parameter_array[i]:
            #         uncertainty[i] = 0.0
            

            # Optional debugging
            #condition_number = np.linalg.cond(regularized_hessian)
            #print(f"Condition Number of Regularized Hessian (Prediction {idx}):", condition_number)
            #condition_number = np.linalg.cond(normalized_hessian)
            #print(f"Condition Number of Normalized Hessian (Prediction {idx}):", condition_number)
            #print("Uncertainties:", uncertainty)
            #self.visualize_uncertainty_estimation2(pred, 'predicted_a', 'predicted_deltar')
            #self.visualize_4d_objective_landscape(pred, idx)
            #self.visualize_uncertainty_of_A(pred, uncertainty, idx)
            #self.visualize_marginal_distribution_of_A(pred, uncertainty, idx)


    def model_summary(self):
        model_summary = torchinfo.summary(self.model, 
                                    input_size=(1, 4096),  # Batch size 1, input dim 4096
                                    verbose=2,
                                    col_names=["input_size", "output_size", "num_params", "kernel_size"],
                                    row_settings=["var_names"])
        print(model_summary)

    def print_model_info(self):
        # 1. Detailed model summary
        model_summary = torchinfo.summary(
            self.model,
            input_size=(1, 4096),
            depth=10,  # Increase depth to show nested layers
            verbose=2,
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            row_settings=["var_names", "depth"]
        )
        print(model_summary)
        
        # 2. Print model architecture
        print("\nModel Architecture:")
        print(self.model)
        
        # 3. Add shape tracking hooks
        def hook_fn(module, input, output):
            print(f"\n{module.__class__.__name__}")
            print(f"Input shape: {input[0].shape}")
            print(f"Output shape: {output.shape}")
        
        # Register hooks for all layers
        for name, layer in self.model.named_modules():
            if hasattr(layer, 'weight'):
                layer.register_forward_hook(hook_fn)
                
        # 4. Run test forward pass
        print("\nForward Pass Shape Tracking:")
        with torch.no_grad():
            test_input = torch.randn(1, 4096)
            _ = self.model(test_input)

    def print_training_example(self):
        # # Print a single training example
        # x_train, y_train, _ = next(iter(self.data_loader.train_dataloader()))
        # print("Training Example:")
        # print("Input Shape:", len(x_train))
        # print("Output Shape:", len(y_train))
        # print("Input Data:")
        # print("Input index 0", x_train[0])
        # print("Input index 0 rows with all zeros")
        # count = 0
        # for i in range(20):
        #     if x_train[0][i].sum() == 0:
        #         count += 1
        # print("count", count)
        # print("length of Input index 0", len(x_train[0]))
        # print("Output Data:")
        # print("Output index 0", y_train[0])
        # print("Output index 0 rows with all zeros")
        # count = 0
        # for i in range(20):
        #     if y_train[0][i].sum() == 0:
        #         count += 1
        # print("count", count)
        # print("length of Output index 0", len(y_train[0]))
        
        # plot_x_data = x_train[0].detach().cpu().numpy()[:20-count, :]
        # plt_y_data = y_train[0].detach().cpu().numpy()[:20-count, :]

        # plt_y_data_param_0 = plt_y_data[:, 0]
        # plt_y_data_param_1 = plt_y_data[:, 1]
        # plt_y_data_param_2 = plt_y_data[:, 2]
        # plt_y_data_param_3 = plt_y_data[:, 3]

        # print(f"Plotting Training Example: {plot_x_data.shape}, {plt_y_data.shape}")

        # print raw synthetic data
        mask_bool = self.synthetic_spectra.masks[0].astype(bool)
        #print(self.synthetic_spectra.spectra[0])
        print(self.synthetic_spectra.spectra[0][mask_bool].shape)
        print(self.synthetic_spectra.parameters[0][mask_bool].shape)

        plot_x_data = self.synthetic_spectra.spectra[0][mask_bool]
        plt_y_data = self.synthetic_spectra.parameters[0][mask_bool]

        plt_y_data_param_0 = plt_y_data[:, 0]
        plt_y_data_param_1 = plt_y_data[:, 1]
        plt_y_data_param_2 = plt_y_data[:, 2]
        plt_y_data_param_3 = plt_y_data[:, 3]

        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1, 8)  # Create 8 columns to work with

        # First subplot takes 4 columns
        ax0 = plt.subplot(gs[0, 0:4])
        for spectrum in plot_x_data:
            ax0.plot(spectrum)
        ax0.set_title("Input Spectra")
        ax0.set_xlabel("k")
        ax0.set_ylabel("Intensity")

        # Remaining subplots take 1 column each
        ax1 = plt.subplot(gs[0, 4])
        ax1.plot(plt_y_data_param_0)
        ax1.set_title("Parameter 0")
        ax1.set_xlabel("Spectra Index")
        ax1.set_ylabel("Value")

        ax2 = plt.subplot(gs[0, 5])
        ax2.plot(plt_y_data_param_1)
        ax2.set_title("Parameter 1")
        ax2.set_xlabel("Spectra Index")
        ax2.set_ylabel("Value")

        ax3 = plt.subplot(gs[0, 6])
        ax3.plot(plt_y_data_param_2)
        ax3.set_title("Parameter 2")
        ax3.set_xlabel("Spectra Index")
        ax3.set_ylabel("Value")

        ax4 = plt.subplot(gs[0, 7])
        ax4.plot(plt_y_data_param_3)
        ax4.set_title("Parameter 3")
        ax4.set_xlabel("Spectra Index")
        ax4.set_ylabel("Value")

        plt.tight_layout()
        save_path = os.path.join(self.config_dir, self.output_dir) + "/training_example.png"
        plt.savefig(save_path)

        
    def print_seq_example(self):
        #example_index = 9
        # print raw synthetic data
        #mask_bool = self.synthetic_spectra.masks[example_index].astype(bool)
        #print(self.synthetic_spectra.spectra[0])
        #print(self.synthetic_spectra.spectra[0][mask_bool].shape)
        #print(self.synthetic_spectra.parameters[0][mask_bool].shape)

        #vary_dict = {"degen": (5,12,"sinusoidal"), "deltar": (-0.02,0.00,"quadratic"), "sigma2": (0.003,0.01,"log"), "e0": (5,5,"sqrt")}
        #const_dict = {"s02": 1}

        vary_dict = {"degen": (11,11,"sinusoidal"), "deltar": (-0.02,0.00,"linear"), "sigma2": (0.006,0.006,"log"), "e0": (3,7,"sinusoidal")}
        const_dict = {"s02": 1}

        # novo_seq, novo_params = self.synthetic_spectra.generate_one_sequence(feff_path_file=self.feff_path_file,
        # sequence_length=20,
        # parameter_profiles=vary_dict,
        # fixed_parameters=const_dict)
        self.synthetic_spectraS = SyntheticSpectrumS(
            feff_path_file=self.feff_path_file,
            param_ranges=self.param_ranges,
            training_mode=self.training_mode,
            num_examples=self.num_examples,
            k_weight=self.k_weight,
            k_range=self.k_range,
            spectrum_noise=self.spectrum_noise,
            noise_range=self.noise_range,
            seed=self.seed,
            generate=False,
        )
        novo_seq, novo_params = self.synthetic_spectraS.generate_glitched_sequence(feff_path_file=self.feff_path_file,
        sequence_length=20,
        parameter_profiles=vary_dict,
        fixed_parameters=const_dict,
        n_glitches=0)

        #print(novo_seq.shape)
        #print(novo_params.shape)

        #plot_x_data = self.synthetic_spectra.spectra[example_index][mask_bool]
        #plt_y_data = self.synthetic_spectra.parameters[example_index][mask_bool]

        plot_x_data = novo_seq
        plt_y_data = novo_params

        plt_y_data_param_0 = plt_y_data[:, 0]
        plt_y_data_param_1 = plt_y_data[:, 1]
        plt_y_data_param_2 = plt_y_data[:, 2]
        plt_y_data_param_3 = plt_y_data[:, 3]

        #print(plt_y_data_param_0)

        # get predictions
        normalized_x_data, _ = normalize_spectra(plot_x_data, self.norm_params_spectra)
        normalized_x_data = torch.tensor(normalized_x_data)
        with torch.no_grad():
            predictions = self.model(normalized_x_data.to(self.model.device))
        predictions = predictions.cpu().numpy()
        predictions = denormalize_data(predictions, self.norm_params_parameters)

        plt_y_data_param_0_pred = predictions[:, 0]
        plt_y_data_param_1_pred = predictions[:, 1]
        plt_y_data_param_2_pred = predictions[:, 2]
        plt_y_data_param_3_pred = predictions[:, 3]

        print(plt_y_data_param_0_pred)

        fig = plt.figure(figsize=(9, 3))
        gs = gridspec.GridSpec(1, 8)

        # Create colorblind-friendly gradient
        n_spectra = len(plot_x_data)
        colors = plt.cm.viridis(np.linspace(0, 1, n_spectra))

        k_grid = self.nn_k_grid
        k_grid = np.array(k_grid, dtype=np.float32)

        # First subplot - spectra
        ax0 = plt.subplot(gs[0, 0:4])
        for idx, spectrum in enumerate(plot_x_data):
            ax0.plot(k_grid, spectrum, color=colors[idx], linewidth=1.5)
        ax0.set_xlim([0, 20])
        ax0.set_title("Input Spectra")
        ax0.set_xlabel("k (Å$^{-1}$)")
        ax0.set_ylabel("$\chi(k)k^2$ (Å$^{-2}$)")

        # Parameter plots with matching colors
        ax1 = plt.subplot(gs[0, 4])
        for idx in range(n_spectra):
            ax1.plot(idx, plt_y_data_param_0[idx], 'o', color=colors[idx], markersize=8)
        ax1.plot(plt_y_data_param_0_pred, '--', color='black', label='Predicted', linewidth=2)
        ax1.set_ylim([4.9, 12.1])
        ax1.set_title("$A$")
        #ax1.set_xlabel("Spectra Index")
        #ax1.set_ylabel("Value")
        #ax1.legend(bbox_to_anchor=(0.5, -0.2), fontsize=7)

        ax2 = plt.subplot(gs[0, 5])
        for idx in range(n_spectra):
            ax2.plot(idx, plt_y_data_param_1[idx], 'o', color=colors[idx], markersize=8)
        ax2.plot(plt_y_data_param_1_pred, '--', color='black', label='Predicted', linewidth=2)
        ax2.set_ylim([-0.025, 0.01])
        ax2.set_title("$\Delta R (Å)$")
        # make values scientific notation
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        #ax2.set_xlabel("Spectra Index")
        #ax2.set_ylabel("Value")
        #ax2.legend(bbox_to_anchor=(0.5, -0.2), fontsize=7)

        ax3 = plt.subplot(gs[0, 6])
        for idx in range(n_spectra):
            ax3.plot(idx, plt_y_data_param_2[idx], 'o', color=colors[idx], markersize=8)
        ax3.plot(plt_y_data_param_2_pred, '--', color='black', label='Predicted', linewidth=2)
        ax3.set_ylim([0.0015, 0.011])
        ax3.set_title("$\sigma^2 (Å^{2})$")
        ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        #ax3.set_xlabel("Spectra Index")
        #ax3.set_ylabel("Value")
        #ax3.legend(bbox_to_anchor=(0.5, -0.2), fontsize=7)

        ax4 = plt.subplot(gs[0, 7])
        for idx in range(n_spectra):
            ax4.plot(idx, plt_y_data_param_3[idx], 'o', color=colors[idx], markersize=8)
        ax4.plot(plt_y_data_param_3_pred, '--', color='black', label='Predicted', linewidth=2)
        ax4.set_ylim([2.5, 7.5])
        ax4.set_title("$\Delta E_0 (eV)$")
        #ax4.set_xlabel("Spectra Index")
        #ax4.set_ylabel("Value")
        #ax4.legend(bbox_to_anchor=(0.5, -0.2), fontsize=7)

        plt.tight_layout()
        save_path = os.path.join(self.config_dir, self.output_dir) + "/figE0_example.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace(".png", ".svg"), dpi=300, bbox_inches='tight')


    def analyze_noise_sensitivity(self, sequence_length=20, n_trials=5, max_glitches=0):
        """Analyze how prediction accuracy varies with noise level."""
        
        vary_dict = {"degen": (5,12,"sinusoidal"), 
                    "deltar": (-0.02,0.00,"quadratic"), 
                    "sigma2": (0.003,0.01,"log"), 
                    "e0": (5,5,"sqrt")}
        const_dict = {"s02": 1}
        
        glitch_counts = range(0, max_glitches + 1, 1)
        param_errors = {param: [] for param in vary_dict.keys()}
        
        for n_glitches in glitch_counts:
            trial_errors = {param: [] for param in vary_dict.keys()}
            
            for trial in range(n_trials):
                novo_seq, novo_params = self.synthetic_spectra.generate_glitched_sequence(
                    feff_path_file=self.feff_path_file,
                    sequence_length=sequence_length,
                    parameter_profiles=vary_dict,
                    fixed_parameters=const_dict,
                    n_glitches=n_glitches
                )
                
                # Normalize both input spectra and parameters
                normalized_x_data, _ = normalize_spectra(novo_seq, self.norm_params_spectra)
                denormalized_params = novo_params
                
                normalized_x_data = torch.tensor(normalized_x_data)
                with torch.no_grad():
                    predictions = self.model(normalized_x_data.to(self.model.device))
                    predictions = predictions.cpu().numpy()
                    predictions = denormalize_data(predictions, self.norm_params_parameters)
                
                # Debug prints for zero noise case
                if n_glitches == 0 and trial == 0:
                    print("\nZero noise validation (normalized values):")
                    for i, param in enumerate(vary_dict.keys()):
                        print(f"\n{param}:")
                        print(f"True values: {denormalized_params[:3, i]}")
                        print(f"Predictions: {predictions[:3, i]}")
                if n_glitches == 20 and trial == 0:
                    print("\nHigh noise validation (normalized values):")
                    for i, param in enumerate(vary_dict.keys()):
                        print(f"\n{param}:")
                        print(f"True values: {denormalized_params[:3, i]}")
                        print(f"Predictions: {predictions[:3, i]}")
                
                # Calculate errors using normalized values
                for i, param in enumerate(vary_dict.keys()):
                    errors = []
                    for j in range(len(denormalized_params)):
                        true_val = denormalized_params[j, i]
                        pred_val = predictions[j, i]
                        if abs(true_val) > 1e-10:
                            error = abs((true_val - pred_val) / true_val) * 100
                            errors.append(error)
                    
                    avg_error = np.mean(errors) if errors else 0
                    trial_errors[param].append(avg_error)
            
            # Average errors across trials
            for param in vary_dict.keys():
                param_errors[param].append(np.mean(trial_errors[param]))
        
        # Plot results
        plt.figure(figsize=(10, 6))
        for param in vary_dict.keys():
            plt.plot(glitch_counts, param_errors[param], marker='o', label=f'{param}')
        
        plt.xlabel('Number of Glitches')
        plt.ylabel('Mean Percentage Error')
        plt.title('Parameter-wise Prediction Error vs Noise Level')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 100)
        
        save_path = os.path.join(self.config_dir, self.output_dir) + "/noise_sensitivity.png"
        plt.savefig(save_path)
        plt.close()

    def analyze_noise_sensitivity_method2(self, sequence_length=20, n_trials=10, max_noise=100):
        """Analyze how prediction accuracy varies with noise level."""
        
        vary_dict = {"degen": (5,12,"sinusoidal"), 
                    "deltar": (-0.02,0.00,"quadratic"), 
                    "sigma2": (0.003,0.01,"log"), 
                    "e0": (5,5,"sqrt")}
        const_dict = {"s02": 1}
        plot_names = ["$A$", "$\Delta R$", "$\sigma^2$", "$\Delta E_0$"]
        
        noise_levels = np.linspace(0.1, max_noise, 30)  # 10 points from 0 to max_noise
        param_errors = {param: [] for param in vary_dict.keys()}

        self.synthetic_spectraS = SyntheticSpectrumS(
            feff_path_file=self.feff_path_file,
            param_ranges=self.param_ranges,
            training_mode=self.training_mode,
            num_examples=self.num_examples,
            k_weight=self.k_weight,
            k_range=self.k_range,
            spectrum_noise=self.spectrum_noise,
            noise_range=self.noise_range,
            seed=self.seed,
            generate=False,
        )
        # novo_seq, novo_params = self.synthetic_spectraS.generate_glitched_sequence(feff_path_file=self.feff_path_file,
        # sequence_length=20,
        # parameter_profiles=vary_dict,
        # fixed_parameters=const_dict,
        # n_glitches=0)

        for noise_level in noise_levels:
            trial_errors = {param: [] for param in vary_dict.keys()}
            
            for trial in range(n_trials):
                novo_seq, novo_params = self.synthetic_spectraS.generate_glitched_sequence(
                    feff_path_file=self.feff_path_file,
                    sequence_length=sequence_length,
                    parameter_profiles=vary_dict,
                    fixed_parameters=const_dict,
                    n_glitches=0,  # No glitches, using noise instead
                    noise_level=noise_level
                )
                
                # Normalize input spectra for model
                normalized_x_data, _ = normalize_spectra(novo_seq, self.norm_params_spectra)
                denormalized_params = novo_params
                
                normalized_x_data = torch.tensor(normalized_x_data)
                with torch.no_grad():
                    predictions = self.model(normalized_x_data.to(self.model.device))
                    predictions = predictions.cpu().numpy()
                    predictions = denormalize_data(predictions, self.norm_params_parameters)
                
                # # Debug prints
                # if noise_level == 0 and trial == 0:
                #     print("\nZero noise validation (normalized values):")
                #     for i, param in enumerate(vary_dict.keys()):
                #         print(f"\n{param}:")
                #         print(f"True values: {denormalized_params[:3, i]}")
                #         print(f"Predictions: {predictions[:3, i]}")
                #         errors = []
                #         for j in range(len(denormalized_params)):
                #             true_val = denormalized_params[j, i]
                #             pred_val = predictions[j, i]
                #             if abs(true_val) > 1e-10:
                #                 error = abs((true_val - pred_val) / true_val) * 100
                #                 errors.append(error)
                #         avg_error = np.mean(errors) if errors else 0
                #         print(f"errors: {errors}")
                #         print(f"Average error: {avg_error}")
                
                # Calculate errors using normalized values
                for i, param in enumerate(vary_dict.keys()):
                    errors = []
                    for j in range(len(denormalized_params)):
                        true_val = denormalized_params[j, i]
                        pred_val = predictions[j, i]
                        if abs(true_val) > 1e-10:
                            error = abs((true_val - pred_val) / true_val) * 100
                            errors.append(error)
                    
                    #avg_error = np.mean(errors) if errors else 0
                    # median
                    avg_error = np.median(errors) if errors else 0
                    trial_errors[param].append(avg_error)
            
            # Average errors across trials
            for param in vary_dict.keys():
                param_errors[param].append(np.mean(trial_errors[param]))
        
        # Plot results
        plt.figure(figsize=(3, 3))
        for idx, param in enumerate(vary_dict.keys()):

            plt.plot(noise_levels, param_errors[param], marker='o', markersize=3, label=plot_names[idx])
        
        plt.xlabel('SNR (dB)')
        plt.ylabel('% Error')
        plt.title('Prediction Error vs SNR')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 40)
        
        save_path = os.path.join(self.config_dir, self.output_dir) + "/noise_sensitivity_SNR.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace(".png", ".svg"), dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_glitch_sensitivity(self, sequence_length=20, n_trials=10, max_noise=100):
        """Analyze how prediction accuracy varies with noise level."""
        
        vary_dict = {"degen": (5,12,"sinusoidal"), 
                    "deltar": (-0.02,0.00,"quadratic"), 
                    "sigma2": (0.003,0.01,"log"), 
                    "e0": (5,5,"sqrt")}
        const_dict = {"s02": 1}
        plot_names = ["$A$", "$\Delta R$", "$\sigma^2$", "$\Delta E_0$"]
        
        #noise_levels = np.linspace(0.1, max_noise, 30)  # 10 points from 0 to max_noise
        # noise levels but need to be integers
        noise_levels = range(0, max_noise + 1, 10)
        param_errors = {param: [] for param in vary_dict.keys()}

        self.synthetic_spectraS = SyntheticSpectrumS(
            feff_path_file=self.feff_path_file,
            param_ranges=self.param_ranges,
            training_mode=self.training_mode,
            num_examples=self.num_examples,
            k_weight=self.k_weight,
            k_range=self.k_range,
            spectrum_noise=False,
            noise_range=[0,0],
            seed=self.seed,
            generate=False,
        )
        # novo_seq, novo_params = self.synthetic_spectraS.generate_glitched_sequence(feff_path_file=self.feff_path_file,
        # sequence_length=20,
        # parameter_profiles=vary_dict,
        # fixed_parameters=const_dict,
        # n_glitches=0)

        for noise_level in noise_levels:
            trial_errors = {param: [] for param in vary_dict.keys()}
            
            for trial in range(n_trials):
                novo_seq, novo_params = self.synthetic_spectraS.generate_glitched_sequence(
                    feff_path_file=self.feff_path_file,
                    sequence_length=sequence_length,
                    parameter_profiles=vary_dict,
                    fixed_parameters=const_dict,
                    n_glitches=noise_level,  # No glitches, using noise instead
                    noise_level=0
                )
                
                # Normalize input spectra for model
                normalized_x_data, _ = normalize_spectra(novo_seq, self.norm_params_spectra)
                denormalized_params = novo_params
                
                normalized_x_data = torch.tensor(normalized_x_data)
                with torch.no_grad():
                    predictions = self.model(normalized_x_data.to(self.model.device))
                    predictions = predictions.cpu().numpy()
                    predictions = denormalize_data(predictions, self.norm_params_parameters)
                
                # Calculate errors using normalized values
                for i, param in enumerate(vary_dict.keys()):
                    errors = []
                    for j in range(len(denormalized_params)):
                        true_val = denormalized_params[j, i]
                        pred_val = predictions[j, i]
                        if abs(true_val) > 1e-10:
                            error = abs((true_val - pred_val) / true_val) * 100
                            errors.append(error)
                    
                    #avg_error = np.mean(errors) if errors else 0
                    # median
                    avg_error = np.median(errors) if errors else 0
                    trial_errors[param].append(avg_error)
            
            # Average errors across trials
            for param in vary_dict.keys():
                param_errors[param].append(np.mean(trial_errors[param]))
        
        # Plot results
        plt.figure(figsize=(3, 3))
        for idx, param in enumerate(vary_dict.keys()):

            plt.plot(noise_levels, param_errors[param], marker='o', markersize=3, label=plot_names[idx])
        
        plt.xlabel('Number of Glitches')
        plt.ylabel('% Error')
        plt.title('Prediction Error vs Number of Glitches')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 40)
        
        save_path = os.path.join(self.config_dir, self.output_dir) + "/glitch_sensitivity.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace(".png", ".svg"), dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_noise_levels(self, max_noise=100, n_levels=3):
        """Visualize how different noise levels affect a single spectrum."""
        
        # Use same parameter settings as analyze_noise_sensitivity
        vary_dict = {"degen": (5,12,"sinusoidal"), 
                    "deltar": (-0.02,0.00,"quadratic"), 
                    "sigma2": (0.003,0.01,"log"), 
                    "e0": (5,5,"sqrt")}
        const_dict = {"s02": 1}
        
        # Generate noise levels
        noise_levels = np.linspace(0.1, max_noise, n_levels)

        self.synthetic_spectraS = SyntheticSpectrumS(
            feff_path_file=self.feff_path_file,
            param_ranges=self.param_ranges,
            training_mode=self.training_mode,
            num_examples=self.num_examples,
            k_weight=self.k_weight,
            k_range=self.k_range,
            spectrum_noise=self.spectrum_noise,
            noise_range=self.noise_range,
            seed=self.seed,
            generate=False,
        )

        # Create subplots
        fig, axes = plt.subplots(n_levels, 1, figsize=(8, 1.5*n_levels))
        fig.suptitle('Effect of Noise Levels on Spectrum')
        
        # Generate base spectrum (no noise)
        base_seq, _ = self.synthetic_spectraS.generate_glitched_sequence(
            feff_path_file=self.feff_path_file,
            sequence_length=1,  # Just one spectrum
            parameter_profiles=vary_dict,
            fixed_parameters=const_dict,
            n_glitches=0,
            glitch_height=(.5,3),
            noise_level=0
        )

        k_grid = self.nn_k_grid
        k_grid = np.array(k_grid, dtype=np.float32)
        
        # Plot spectra with different noise levels
        for idx, noise_level in enumerate(noise_levels):
            noisy_seq, _ = self.synthetic_spectraS.generate_glitched_sequence(
                feff_path_file=self.feff_path_file,
                sequence_length=1,
                parameter_profiles=vary_dict,
                fixed_parameters=const_dict,
                n_glitches=0,
                glitch_height=(.5,3),
                noise_level=noise_level
            )
            
            axes[idx].plot(k_grid, base_seq[0], 'b-', label='Clean', alpha=0.5)
            axes[idx].plot(k_grid, noisy_seq[0], 'r-', label='Noisy', alpha=0.7)
            axes[idx].set_xlim([0, 20])
            axes[idx].set_title(f'Noise Level: {noise_level:.3f}')
            axes[idx].set_xlabel('k (Å$^{-1}$)')
            axes[idx].set_ylabel('$\chi(k)k^2$')
            axes[idx].legend()
            axes[idx].grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.config_dir, self.output_dir) + "/noise_visualization_SNR.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace(".png", ".svg"), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_loss_curves(self, metrics_file, train_rate=7, val_rate=80, best_rate=1):
        """
        Plots training, validation, and best validation loss curves in a colorblind-friendly manner with a log scale.
        
        Args:
            metrics_file (str): Path to the CSV file containing metrics data.
        """
        # Load the data
        data = pd.read_csv(metrics_file)
        
        # Sample data points at different rates
        train_data = data.iloc[::train_rate]
        val_data = data.iloc[::val_rate]
        best_data = data.iloc[::best_rate]
        
        # Plotting
        plt.figure(figsize=(5, 3))
        
        # Use colorblind-friendly colors with different sampling rates
        plt.plot(train_data['epoch'], train_data['train/loss'], 
                label="Training Loss", marker='o', color="#D55E00", markersize=3, linestyle=' ')
        plt.plot(val_data['epoch'], val_data['val/loss'], 
                label="Validation Loss", marker='o', color="#0072B2", markersize=3, linestyle=' ')
        plt.plot(best_data['epoch'], best_data['val/loss_best'], 
                label="Best Validation Loss", marker='o', linestyle='--', color="#009E73", markersize=1)
        
        # set all font sizes to 12
        plt.rcParams.update({'font.size': 12})

        plt.yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("Log(Loss)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config_dir, self.output_dir) + "/loss_curves.png")
        #plt.savefig(os.path.join(self.config_dir, self.output_dir) + "/loss_curves.svg")
        plt.close()


    def laplace(self):

        self.la = Laplace(self.model, "regression",
                    subset_of_weights="all",
                    hessian_structure="diag")
        self.la.fit(self.data_loader.train_dataloader(batch_size=16))
        self.la.optimize_prior_precision(
            method="gridsearch",
            pred_type="glm",
            link_approx="probit",
            val_loader=self.data_loader.val_dataloader(batch_size=16),
        )

    def get_MLP_predictions_laplace(self, sequence = False):

        self.laplace()

        if not sequence:
            predictions = self.run_predictions()
        else:
            predictions = self.run_predictions_S()
        
        self.mlp_pred_laplace = []

        for idx, pred in enumerate(predictions):
            normalized_chi_k = pred['normalized_chi_k']
            normalized_chi_k = torch.tensor(normalized_chi_k)
            #print(normalized_chi_k.shape)
            pred, covar_matrix = self.la(normalized_chi_k, pred_type="glm", link_approx="probit")
            pred = pred.cpu().numpy()
            cov_matrix = covar_matrix[0]
            variances = torch.diagonal(cov_matrix)
            uncertainties = torch.sqrt(variances) # values might be negative. bad.
            uncertainties = uncertainties.cpu().numpy()
            result = [pred, uncertainties]

            self.mlp_pred_laplace.append(result)

            # Optional debugging
            #condition_number = np.linalg.cond(regularized_hessian)
            #print(f"Condition Number of Regularized Hessian (Prediction {idx}):", condition_number)
            #condition_number = np.linalg.cond(normalized_hessian)
            #print(f"Condition Number of Normalized Hessian (Prediction {idx}):", condition_number)
            print("Uncertainties:", self.mlp_pred_laplace)




    def visualize_uncertainty_estimation(self, n_samples=50):
        """
        Visualize uncertainty estimation with two subplots:
        1. Multiple curves showing variation within uncertainty bounds
        2. Parameter sensitivity visualization
        """
        predictions = self.run_predictions()
        self.get_MLP_uncertainty()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # For the first prediction
        pred = predictions[0]
        uncertainty = self.mlp_uncertainties[0]
        params = np.array([
            pred['predicted_a'],
            pred['predicted_deltar'],
            pred['predicted_sigma2'],
            pred['predicted_e0']
        ])
        
        # Left plot: Multiple curves with uncertainty
        q_range = pred['interpolated_chi_q_re']
        base_curve = self.objective(q_range, *params)
        ax1.plot(q_range, base_curve, 'k-', label='Predicted', linewidth=2)
        
        # Generate samples within uncertainty bounds
        samples = multivariate_normal.rvs(
            mean=params, 
            cov=np.diag(uncertainty**2), 
            size=n_samples
        )
        
        for sample in samples:
            curve = self.objective(q_range, *sample)
            ax1.plot(q_range, curve, 'b-', alpha=0.1)
        
        ax1.set_xlabel('q')
        ax1.set_ylabel('Intensity')
        ax1.set_title('Parameter Uncertainty Effects')
        
        # Right plot: Parameter sensitivity
        param_names = ['a', 'Δr', 'σ²', 'E₀']
        ax2.errorbar(
            range(len(params)),
            params,
            yerr=uncertainty,
            fmt='o',
            capsize=5,
            capthick=2,
            markersize=8
        )
        ax2.set_xticks(range(len(params)))
        ax2.set_xticklabels(param_names)
        ax2.set_title('Parameter Uncertainties')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config_dir, self.output_dir) + "/uncertainty_estimation.png")
        plt.close()

# function for ploting Pd for the paper

    def plot_results_Pd(self, sequence=False, title_size=14, label_size=14, tick_size=12, legend_size=6):
        dataset_names = self.config["experiment"]["dataset_names"]
        if not sequence:
            predictions = self.run_predictions()
        else:
            predictions = self.run_predictions_S()
        num_predictions = len(predictions)
        kmin = self.k_range[0]
        kmax = self.k_range[1]
        
        # Create grayscale colors for experimental data
        exp_colors = ['black', 'blue', 'darkred']
        predicted_colors = ['gray', 'lightblue', 'orange']
        labels = ['Pd Foil', 'Pd NP / He 298 K', 'Pd NP / $H_2$ 298 K']
        network_type = self.config["neural_network"]["architecture"]["type"]

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))

        for idx, (pred, exp_color, predicted_color, lab) in enumerate(zip(predictions, exp_colors, predicted_colors, labels)):
            k_grid = np.array(self.config["neural_network"]["k_grid"], dtype=np.float32)
            
            interpolated_chi_q = pred['interpolated_chi_q_re']
            interpolated_artemis = self.build_synth_spectra(pred['artemis_result'])
            predicted_parameter_array = [pred['predicted_a'], pred['predicted_deltar'], 
                                    pred['predicted_sigma2'], pred['predicted_e0']]
            interpolated_nn = self.build_synth_spectra(predicted_parameter_array)
            
            mse_error_artemis = self.get_MSE_error(interpolated_artemis, interpolated_chi_q)
            mse_error_nn = self.get_MSE_error(interpolated_nn, interpolated_chi_q)

            # Experimental data (black/gray)
            ax1.plot(k_grid, interpolated_chi_q, color=exp_color, label=f'{lab}', linewidth=2.5)
            ax2.plot(k_grid, interpolated_chi_q, color=exp_color, label=f'{lab}', linewidth=2.5)

            # Artemis and NN predictions with distinct colors
            ax1.plot(k_grid, interpolated_artemis,
                    label=f'MSE={mse_error_artemis:.3f}', 
                    color=predicted_color, linestyle='--', alpha=0.7, linewidth=1.5)
            
            ax2.plot(k_grid, interpolated_nn,
                    label=f'MSE={mse_error_nn:.3f}', 
                    color=predicted_color, linestyle='--', alpha=0.7, linewidth=1.5)

        for ax in [ax1, ax2]:
            ax.set_xlim(kmin, kmax)
            ax.legend(fontsize=legend_size, loc='best', ncol=1)
            ax.set_xlabel('k (Å$^{-1}$)', fontsize=label_size)
            ax.set_ylabel(r'$Re[\chi(q)] \ (\mathrm{\AA}^{-2})$', fontsize=label_size)
            ax.tick_params(axis='both', which='major', labelsize=tick_size)
            

        ax1.set_title('Artemis Results', fontsize=title_size, pad=10)
        ax2.set_title('Neural Network Results', fontsize=title_size, pad=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config_dir, self.output_dir) + '/qspace_pd.png', dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.config_dir, self.output_dir) + '/qspace_pd.svg', dpi=300, bbox_inches='tight')
        plt.close(fig)

class HistoryLogger(pl.Callback):
    def __init__(self, history):
        super().__init__()
        self.history = history
        
    def on_train_epoch_end(self, trainer, pl_module):
        self.history["train/loss"].append(trainer.callback_metrics["train/loss"].item())
        
    def on_validation_epoch_end(self, trainer, pl_module):
        self.history["val/loss"].append(trainer.callback_metrics["val/loss"].item())

    def plot_chi(self, dataset_dir):
        data = read_ascii(dataset_dir)

        k_weight = self.k_weight

        if k_weight == 0:
            data.chi0 = data.chi
            plt_t.plot(data.k, data.chi0, label='Chi0')

        elif k_weight == 1:
            data.chi1 = data.chi * data.k
            plt_t.plot(data.k, data.chi1, label='Chi1')


        elif k_weight == 2:
            data.chi2 = data.chi * data.k ** 2
            plt_t.plot(data.k, data.chi2, label='Chi2')

        else:
            raise ValueError("Invalid k-weight")

        plt_t.xlabel('k')
        plt_t.xfrequency(20)
        plt_t.ylabel('Chi')
        plt_t.title(f'{dataset_dir}')
        plt_t.show()
    
    def plot_r(self, dataset_dir):
        data = read_ascii(dataset_dir)

        k_weight = self.k_weight
        kmin = self.kmin
        kmax = self.kmax

        plt_t.clear_figure()

        if k_weight == 0:
            data.chi = data.chi

        elif k_weight == 1:
            data.chi = data.chi * data.k


        elif k_weight == 2:
            data.chi = data.chi * data.k ** 2

        else:
            raise ValueError("Invalid k-weight")

        xftf(data, kweight=k_weight, kmin=kmin, kmax=kmax)

        plt_t.plot(data.r, data.chir_mag, label='FT-EXAFS')
        plt_t.xlabel('r')
        plt_t.xfrequency(12)
        plt_t.ylabel('chir_mag')
        plt_t.title(f'{dataset_dir}')
        plt_t.show()


# def main():

#     #path = "/home/nick/Projects/crowpeas/MLP_Pd/output_2L_decay_0_qminqmaxNone_t2_n01/training10k_qspace.toml"
#     #path = "/home/nick/Projects/crowpeas/MLP_Pt/output_2L_decay_0_qminqmaxNone_t2_n01/training10k_qspace.toml"
#     path = "/home/nick/Projects/crowpeas/MLP_Rh/output_2L_decay_0_qminqmaxNone_t2_n01/training10k_qspace.toml"
#     print(path)
#     crowpeas = CrowPeas()

#     # (        # train loop
#     #     crowpeas.load_config(path)
#     #     .load_and_validate_config()
#     #     .init_synthetic_spectra()
#     #     .save_training_data()
#     #     .prepare_dataloader()
#     #     .save_config()
#     #     .train()
#     #     .load_model()
#     #     .validate_model()

#     # )
#     #crowpeas.plot_training_history()

#     # (        # testing loop
#     #     crowpeas.load_config(path)
#     #     .load_and_validate_config()
#     #     .load_synthetic_spectra()
#     #     .prepare_dataloader()
#     #     .load_model()

#     # )
#     # crowpeas.plot_parity()
#     # crowpeas.plot_results()

#     # (        # testing loop laplace
#     #     crowpeas.load_config(path)
#     #     .load_and_validate_config()
#     #     .load_synthetic_spectra()
#     #     .prepare_dataloader()
#     #     .load_model()
#     #     .get_MLP_predictions_laplace()

#     # )

#     # (        # testing loop unc fig
#     #     crowpeas.load_config(path)
#     #     .load_and_validate_config()
#     #     .load_synthetic_spectra()
#     #     .prepare_dataloader()
#     #     .load_model()
#     #     #.get_MLP_uncertainty()

#     # ) 

#     (        # SNR
#         crowpeas.load_config(path)
#         .load_and_validate_config()
#         .load_synthetic_spectra()
#         .prepare_dataloader()
#         .load_model() 
        
#     )    
#     #crowpeas.plot_parity()
#     #crowpeas.plot_results()
#     #crowpeas.plot_results_Pd()
#     #crowpeas.print_training_example()
#     #crowpeas.print_seq_example()
#     #crowpeas.analyze_noise_sensitivity_method2()
#     #crowpeas.analyze_glitch_sensitivity()
#     #crowpeas.visualize_noise_levels()
#     crowpeas.plot_loss_curves("/home/nick/Projects/crowpeas/MLP_Rh/lightning_logs/version_5/metrics.csv")
#     #crowpeas.visualize_noise_impact()
#     #crowpeas.plot_parity()
#     #crowpeas.plot_parity2()
#     #crowpeas.plot_training_history()


# if __name__ == "__main__":
#    main()
