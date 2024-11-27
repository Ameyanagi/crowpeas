"""Main module."""

import toml
import os
import math
from typing import Sequence, Literal
import json

from .data_generator import SyntheticSpectrum
from .data_loader import CrowPeasDataModule
import lightning as pl
from .model.BNN import BNN
from .model.MLP import MLP
from .model.HetMLPNM import hetMLP
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np

from .utils import (
    normalize_data,
    denormalize_data,
    normalize_spectra,
    denormalize_spectra,
    interpolate_spectrum,
    predict_with_uncertainty,
    predict_with_uncertainty_hetMLP
)
from larch.xafs import xftf, xftr, feffpath, path2chi, ftwindow
from larch.io import read_ascii
from larch.fitting import param, guess, param_group
from larch import Group 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

import plotext as plt_t

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

    k: np.ndarray

    # parameters related to general neural network
    neural_network: dict
    model_name: str
    model_dir: str
    checkpoint_dir: str
    checkpoint_name: str
    model: pl.LightningModule

    # parameters related to neural network architecture
    nn_type: Literal["MLP", "BNN", "hetMLP"]
    nn_activation: str
    nn_output_activation: str
    nn_output_dim: int
    nn_hidden_dims: Sequence[int]
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
    noise_range: Sequence[float]
    k_range: Sequence[float]
    r_range: Sequence[float]
    k_weight: int
    train_size_ratio: float
    val_size_ratio: float
    test_size_ratio: float

    # parameters related to neural network hyperparameters
    epochs: int
    batch_size: int
    learning_rate: float

    # synthetic spectra for training
    synthetic_spectra: SyntheticSpectrum
    data_loader: CrowPeasDataModule
    history: dict = {"train_loss": [], "val_loss": []}

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
        required_for_general = ["title", "mode"]
        required_for_training = [
            "feffpath",
            "param_ranges",
            "training_set_dir",
            "num_examples",
            "k_weight",
            "spectrum_noise",
            "noise_range",
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

        self.nn_type = self.config["neural_network"]["architecture"]["type"]
        self.nn_activation = self.config["neural_network"]["architecture"]["activation"]
        self.nn_output_activation = self.config["neural_network"]["architecture"][
            "output_activation"
        ]
        self.nn_output_dim = self.config["neural_network"]["architecture"]["output_dim"]
        self.nn_hidden_dims = self.config["neural_network"]["architecture"][
            "hidden_dims"
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
            self.k_range = self.config["training"]["k_range"]
            self.k_weight = self.config["training"]["k_weight"]
            self.r_range = self.config["training"]["r_range"]
            self.seed = self.config["training"].get("seed", None)
            self.train_size_ratio = self.config["training"].get("train_size_ratio", 0.8)
            self.val_size_ratio = self.config["training"].get("val_size_ratio", 0.1)
            self.test_size_ratio = self.config["training"].get("test_size_ratio", 0.1)


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

    def save_config(self, path: str | None = None):

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
                    "k_range": self.k_range,
                    "k_weight": self.k_weight,
                    "r_range": self.r_range,
                    "train_size_ratio": self.train_size_ratio,
                    "val_size_ratio": self.val_size_ratio,
                    "test_size_ratio": self.test_size_ratio,
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
                    "k_range": self.k_range,
                    "k_weight": self.k_weight,
                    "r_range": self.r_range,
                    "train_size_ratio": self.train_size_ratio,
                    "val_size_ratio": self.val_size_ratio,
                    "test_size_ratio": self.test_size_ratio,
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
                        "k_range": self.k_range,
                        "k_weight": self.k_weight,
                        "r_range": self.r_range,
                        "train_size_ratio": self.train_size_ratio,
                        "val_size_ratio": self.val_size_ratio,
                        "test_size_ratio": self.test_size_ratio,
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
                    },
                    "k_grid": self.nn_k_grid,
                },
                
            }

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
                with open(self.config_filename, "w") as f:
                    toml.dump(self.config, f)
            elif self.config_filename.endswith("json"):
                with open(self.config_filename, "w") as f:
                    json.dump(self.config, f)
            else:
                raise ValueError(
                    "File format not supported. Only TOML and JSON are supported."
                )

        return self

    def init_synthetic_spectra(self, generate=True):
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

        print("Loading training data")

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

        # TODO: read normal_params
        normalized_spectra, norm_spectra_params = normalize_spectra(
            self.synthetic_spectra.spectra
        )
        normalized_parameters, norm_parameters_params = normalize_data(
            self.synthetic_spectra.parameters
        )

        self.norm_params_spectra = norm_spectra_params
        self.norm_params_parameters = norm_parameters_params

        self.data_loader = CrowPeasDataModule(
            spectra=normalized_spectra,
            parameters=normalized_parameters,
            random_seed=seed,
        )
        self.data_loader.setup(setup)

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

        return self

    def load_model(self):
        if self.nn_type.lower().startswith("bnn"):
            self.model = BNN.load_from_checkpoint(
                os.path.join(self.checkpoint_dir, self.checkpoint_name)
            )
        elif self.nn_type.lower().startswith("mlp"):
            self.model = MLP.load_from_checkpoint(
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
                k_min=self.k_range[0],
                k_max=self.k_range[1],
                r_min=self.r_range[0],
                r_max=self.r_range[1],
                rmax_out=6,
                window="kaiser",
                dx=1,
                input_form="r",
                activation=self.nn_activation,
            )
        elif self.nn_type.lower().startswith("mlp"):
            self.model = MLP(
                hidden_layers=self.nn_hidden_dims,
                output_size=self.nn_output_dim,
                k_min=self.k_range[0],
                k_max=self.k_range[1],
                r_min=self.r_range[0],
                r_max=self.r_range[1],
                rmax_out=6,
                window="kaiser",
                dx=1,
                input_form="r",
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
                input_form="r",
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
        callback = self.create_checkpoint_callback()

        history_logger = HistoryLogger(self.history)
        callbacks = [callback, history_logger]  # Add to existing checkpoint callback
        trainer = pl.Trainer(max_epochs=self.epochs, callbacks=callbacks)
        trainer.fit(
            self.model,
            self.data_loader.train_dataloader(batch_size=self.batch_size),
            self.data_loader.val_dataloader(batch_size=self.batch_size),
        )

        best_checkpoint = callback.best_model_path
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

        self.x_test, self.y_test = next(iter(self.data_loader.test_dataloader()))

        self.model.eval()

        with torch.no_grad():
            self.test_pred = self.model(self.x_test.to(self.model.device))

        if self.nn_type.lower().startswith("het"):
            self.test_pred_mu, self.test_pred_sigma = self.test_pred # test_pred is a tuple for hetMLP
            self.denormalized_x_test = self.denormalize_spectra(self.x_test)
            self.denormalized_y_test = self.denormalize_data(self.y_test)
            self.denormalized_test_pred = self.denormalize_data(self.test_pred_mu)
        else:
            self.denormalized_x_test = self.denormalize_spectra(self.x_test)
            self.denormalized_y_test = self.denormalize_data(self.y_test)
            self.denormalized_test_pred = self.denormalize_data(self.test_pred)

        return self

    def plot_parity(self, save_path="./parity.png"):
        parameter_name_dict = {0: "A", 1: "deltar", 2: "sigma2", 3: "e0"}

        if not hasattr(self, "denormalized_y_test") or not hasattr(
            self, "denormalized_test_pred"
        ):
            self.validate_model()

        num_parameters = self.denormalized_y_test.shape[1]

        # Plot predicted vs. true values for each parameter in a single row
        fig, axs = plt.subplots(1, num_parameters, figsize=(5 * num_parameters, 5))

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
            axs[i].set_xlabel("True Values")
            axs[i].set_ylabel("Predicted Values")
            axs[i].set_title(f"Parameter {parameter_name_dict[i]}: Predicted vs. True")

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path)

        return fig

    def plot_test_spectra(self, index, save_path="./spectra.png"):

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
        k_weight = self.k_weight
        kmin = self.k_range[0]
        kmax = self.k_range[1]
        rmin = self.r_range[0]
        rmax = self.r_range[1]
        k_grid = self.config["neural_network"]["k_grid"] # this is actually a list of strings #TODO: make sure this is not an issue elsewhere in the code
        k_grid = np.array(k_grid, dtype=np.float32)
        k_weight = self.config["experiment"]["k_weight"]

        xftf(data, kweight=k_weight, kmin=kmin, kmax=kmax)
        xftr(data, rmin=rmin, rmax=rmax)

        if k_weight == 2:
            data.chi2 = data.chi * data.k ** 2
        
        interpolated_chi_k = interpolate_spectrum(data.k, data.chi2, k_grid)
        interpolated_chi_k = torch.tensor(interpolated_chi_k).unsqueeze(0)
        xftf(data, kweight=k_weight, kmin=kmin, kmax=kmax)
        xftr(data, rmin=rmin, rmax=rmax)
        interpolated_chi_q = interpolate_spectrum(data.q, data.chiq_re, k_grid)

        return interpolated_chi_k, interpolated_chi_q

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
            interpolated_chi_k, interpolated_chi_q = self.process_exp_data(self.load_exp_data(dataset_dir))
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
                'interpolated_chi_q': interpolated_chi_q,
                'artemis_result': artemis_result,
                'artemis_unc': artemis_unc_entry,
                'dataset_name': dataset_name
            })

        return self.predictions

    def build_synth_spectra(self, predicted_params: list | np.ndarray):

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

        interpolated_predicted = interpolate_spectrum(path_predicted.q, path_predicted.chiq_re, k_grid)

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

        e2 = np.mean((interpolated_artemis_in_range - interpolated_exp_in_range)**2)

        return e2



    def save_predictions_to_toml(self, filename):
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
                file.write(f"interpolated_chi_q = {tensor_to_list(prediction['interpolated_chi_q'])}\n")
                file.write(f"artemis_result = \"{prediction['artemis_result']}\"\n")
                file.write(f"artemis_unc = \"{prediction['artemis_unc']}\"\n")
                file.write("\n")




    def plot_results(self):
        dataset_names = self.config["experiment"]["dataset_names"]
        predictions = self.run_predictions()
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
            axs_params[0].set_title('Delta A')
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
            axs_params[2].set_title('Sigma2')
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
            axs_params[3].set_title('E0')
            axs_params[3].set_xlabel('NN')
            axs_params[3].set_ylabel('Artemis')
            axs_params[3].tick_params(axis='both', which='major', labelsize=8)

        axs_params[0].legend()
        plt.tight_layout()
        plt.savefig('parameters.png')
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

            interpolated_chi_q = pred['interpolated_chi_q']
            interpolated_artemis = self.build_synth_spectra(pred['artemis_result'])

            mse_error = self.get_MSE_error(interpolated_chi_q, interpolated_artemis)

            network_type = self.config["neural_network"]["architecture"]["type"]

            ax.plot(k_grid, interpolated_chi_q, label=f'{network_type} @ {dataset_names[idx]} MSE = {mse_error:.3f}', color=color)
            ax.plot(
                k_grid, interpolated_artemis,
                label='Artemis', color="black"
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
        plt.savefig('qspace.png')
        plt.close(fig_q)

    def plot_training_history(self, save_path="training_history.png"):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history["train_loss"], label="Training Loss")
        plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return plt.gcf()


class HistoryLogger(pl.Callback):
    def __init__(self, history):
        super().__init__()
        self.history = history
        
    def on_train_epoch_end(self, trainer, pl_module):
        self.history["train_loss"].append(trainer.callback_metrics["train/loss"].item())
        
    def on_validation_epoch_end(self, trainer, pl_module):
        self.history["val_loss"].append(trainer.callback_metrics["val/loss"].item())


    # def plot_results(self):
    #     dataset_names = self.config["experiment"]["dataset_names"]
    #     predictions = self.run_predictions()
    #     num_predictions = len(predictions)

    #     # ============================
    #     # Plot Parameters
    #     # ============================
    #     num_param_plots = 4
    #     fig_params, axs_params = plt.subplots(1, num_param_plots, figsize=(20, 4))

    #     # find min and max artimis values
    #     all_artemis = [pred['artemis_result'] for pred in predictions]
    #     all_artemis = np.array(all_artemis)
    #     min_artemis = np.min(all_artemis, axis=0)
    #     max_artemis = np.max(all_artemis, axis=0)

    #     for pred in predictions:
    #         # Plot Delta A
    #         axs_params[0].errorbar(
    #             pred['predicted_a'], pred['artemis_result'][0],
    #             xerr=0, yerr=pred['artemis_unc'][0],
    #             fmt='o', label='Predicted', color='red'
    #         )
    #         axs_params[0].plot(
    #             [min_artemis[0]-1, max_artemis[0]+1],
    #             [min_artemis[0]-1, max_artemis[0]+1],
    #             'r--'
    #         )
    #         axs_params[0].set_title('Delta A')
    #         axs_params[0].set_xlabel('NN')
    #         axs_params[0].set_ylabel('Artemis')
    #         axs_params[0].tick_params(axis='both', which='major', labelsize=8)

    #         # Plot Delta R
    #         axs_params[1].errorbar(
    #             pred['predicted_deltar'], pred['artemis_result'][1],
    #             xerr=0, yerr=pred['artemis_unc'][1],
    #             fmt='o', label='Predicted', color='red'
    #         )
    #         axs_params[1].plot(
    #             [min_artemis[1]-0.1, max_artemis[1]+0.1],
    #             [min_artemis[1]-0.1, max_artemis[1]+0.1],
    #             'r--'
    #         )
    #         axs_params[1].set_title('Delta R')
    #         axs_params[1].set_xlabel('NN')
    #         axs_params[1].set_ylabel('Artemis')
    #         axs_params[1].tick_params(axis='both', which='major', labelsize=8)

    #         # Plot Sigma2
    #         axs_params[2].errorbar(
    #             pred['predicted_sigma2'], pred['artemis_result'][2],
    #             xerr=0, yerr=pred['artemis_unc'][2],
    #             fmt='o', label='Predicted', color='red'
    #         )
    #         axs_params[2].plot(
    #             [min_artemis[2]-0.01, max_artemis[2]+0.01],
    #             [min_artemis[2]-0.01, max_artemis[2]+0.01],
    #             'r--'
    #         )
    #         axs_params[2].set_title('Sigma2')
    #         axs_params[2].set_xlabel('NN')
    #         axs_params[2].set_ylabel('Artemis')
    #         axs_params[2].tick_params(axis='both', which='major', labelsize=8)

    #         # Plot E0
    #         axs_params[3].errorbar(
    #             pred['predicted_e0'], pred['artemis_result'][3],
    #             xerr=0, yerr=pred['artemis_unc'][3],
    #             fmt='o', label='Predicted', color='red'
    #         )
    #         axs_params[3].plot(
    #             [-10, 10], [-10, 10], 'r--'
    #         )
    #         axs_params[3].set_title('E0')
    #         axs_params[3].set_xlabel('NN')
    #         axs_params[3].set_ylabel('Artemis')
    #         axs_params[3].tick_params(axis='both', which='major', labelsize=8)

    #     plt.tight_layout()
    #     plt.savefig('parameters.png')
    #     plt.close(fig_params)  # Close the figure to free memory

    #     # ============================
    #     # Plot Q-space
    #     # ============================
    #     # Determine grid size based on number of predictions
    #     cols_q = 3  # Number of columns in the grid
    #     rows_q = math.ceil(num_predictions / cols_q)

    #     fig_q, axs_q = plt.subplots(rows_q, cols_q, figsize=(5 * cols_q, 4 * rows_q))
    #     axs_q = axs_q.flatten() if num_predictions > 1 else [axs_q]

    #     for idx, pred in enumerate(predictions):
    #         ax = axs_q[idx]
    #         k_grid = self.config["neural_network"]["k_grid"]
    #         k_grid = np.array(k_grid, dtype=np.float32) # TODO fix this

    #         interpolated_chi_q = pred['interpolated_chi_q']
    #         interpolated_artemis = self.build_synth_spectra(pred['artemis_result'])

    #         mse_error = self.get_MSE_error(interpolated_chi_q, interpolated_artemis)

    #         ax.plot(k_grid, interpolated_chi_q, label='Experimental', color='black')
    #         ax.plot(
    #             k_grid, interpolated_artemis,
    #             label=f'Artemis MSE = {mse_error:.3f}', color='blue'
    #         )
    #         ax.set_xlim(2, 14)
    #         ax.set_title(f'Q-space Prediction {idx + 1}')
    #         ax.legend()
    #         ax.set_xlabel('k')
    #         ax.set_ylabel('Chi Q')
    #         ax.tick_params(axis='both', which='major', labelsize=8)

    #     # Remove any unused subplots
    #     for idx in range(num_predictions, rows_q * cols_q):
    #         fig_q.delaxes(axs_q[idx])

    #     plt.tight_layout()
    #     plt.savefig('qspace.png')
    #     plt.close(fig_q) 



    # def predict_on_experimental_data(self):

    #     network_type = self.config["neural_network"]["architecture"]["type"]
    #     krange = self.config["experiment"]["k_range"]
    #     kmin = krange[0]
    #     kmax = krange[1]
    #     r_range = self.config["experiment"]["r_range"]
    #     rmin = r_range[0]
    #     rmax = r_range[1]

    #     k_grid = self.config["neural_network"]["k_grid"] # this is actually a list of strings #TODO: make sure this is not an issue elsewhere in the code
    #     k_grid = np.array(k_grid, dtype=np.float32)
    #     k_weight = self.config["experiment"]["k_weight"]

    #     exp_data_path = self.config["experiment"]["dataset_dir"]
    #     pt_data  = read_ascii(exp_data_path)

    #     if k_weight == 2:
    #         pt_data.chi2 = pt_data.chi*pt_data.k**2

    #     interpolated_chi_k = interpolate_spectrum(pt_data.k, pt_data.chi2, k_grid)
    #     interpolated_chi_k = torch.tensor(interpolated_chi_k).unsqueeze(0)
    #     xftf(pt_data, kweight=k_weight, kmin=kmin, kmax=kmax)
    #     xftr(pt_data, rmin=rmin, rmax=rmax)
    #     interpolated_chi_q = interpolate_spectrum(pt_data.q, pt_data.chiq_re, k_grid)


    #     artemis_results = self.config["artemis"]["result"]
    #     artemis_unc = self.config["artemis"]["unc"]

    #     if network_type == "BNN":
    #         self.predict_and_denormalize_BNN(interpolated_chi_k/self.norm_params_spectra["max_abs_val"])
    #         preds, uncs = self.denormalized_test_pred
    #         predicted_a, predicted_deltar, predicted_sigma2, predicted_e0 = preds
    #         a_unc, deltar_unc, sigma2_unc, e0_unc = uncs

    #     if network_type == "MLP":
    #         self.predict_and_denormalize(interpolated_chi_k/self.norm_params_spectra["max_abs_val"])
    #         predicted_a, predicted_deltar, predicted_sigma2, predicted_e0 = self.denormalized_test_pred[0]
    #         a_unc, deltar_unc, sigma2_unc, e0_unc = [0,0,0,0]


    #     path_predicted = feffpath(self.feff_path_file)
    #     path_predicted.s02 = 1
    #     path_predicted.degen = predicted_a
    #     path_predicted.deltar = predicted_deltar
    #     path_predicted.sigma2 = predicted_sigma2
    #     path_predicted.e0 = predicted_e0
    #     path2chi(path_predicted)
    #     xftf(path_predicted, kweight=k_weight, kmin=kmin, kmax=kmax)
    #     xftr(path_predicted, rmin=rmin, rmax=rmax)

    #     interpolated_predicted = interpolate_spectrum(path_predicted.q, path_predicted.chiq_re, k_grid)

    #     artemis_results = self.config["artemis"]["result"]
    #     artemis_unc = self.config["artemis"]["unc"]

    #     artemis_a = self.config["artemis"]["result"][0]
    #     artemis_deltar = self.config["artemis"]["result"][1]
    #     artemis_sigma2 = self.config["artemis"]["result"][2]
    #     artemis_e0 = self.config["artemis"]["result"][3]
    

    #     path_artemis = feffpath(self.feff_path_file)
    #     path_artemis.s02 = 1
    #     path_artemis.degen = artemis_a
    #     path_artemis.deltar = artemis_deltar
    #     path_artemis.sigma2 = artemis_sigma2
    #     path_artemis.e0 = artemis_e0
    #     path2chi(path_artemis)
    #     xftf(path_artemis, kweight=k_weight, kmin=kmin, kmax=kmax)
    #     xftr(path_artemis, rmin=rmin, rmax=rmax)

    #     interpolated_artemis = interpolate_spectrum(path_artemis.q, path_artemis.chiq_re, k_grid)

    #     self.exp_prediction = {
    #         "predicted_a": predicted_a,
    #         "predicted_deltar": predicted_deltar,
    #         "predicted_sigma2": predicted_sigma2,
    #         "predicted_e0": predicted_e0,
    #         "interpolated_chi_k": interpolated_chi_k,
    #         "interpolated_chi_q": interpolated_chi_q,
    #         "interpolated_predicted": interpolated_predicted,
    #         "interpolated_artemis": interpolated_artemis
    #     }

    #     def get_MSE_error(interpolated_artemis, interpolated_exp, k_grid, krange):
            
    #         kmin = krange[0]
    #         kmax = krange[1]

    #         # get exp in range
    #         interpolated_exp_in_range = [i[1] for i in zip(k_grid, interpolated_exp) if kmin <= i[0] <= kmax]
    #         interpolated_exp_in_range = np.array(interpolated_exp_in_range)

    #         # MSE error between predicted and artemis with nano
    #         interpolated_artemis_in_range = [i[1] for i in zip(k_grid, interpolated_artemis) if kmin <= i[0] <= kmax]
    #         interpolated_artemis_in_range = np.array(interpolated_artemis_in_range)

    #         e2 = np.mean((interpolated_artemis_in_range - interpolated_exp_in_range)**2)

    #         return e2
        
    #     mse_artemis = get_MSE_error(interpolated_artemis, interpolated_chi_q, k_grid, krange)
        
    #     # Define the overall figure
    #     fig = plt.figure(figsize=(10, 5))

    #     # Create a grid with 2 rows and 3 columns, with the last column being used for the single plot
    #     gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 2])

    #     # First 2x2 grid for the subplots 1-4
    #     ax1 = fig.add_subplot(gs[0, 0])
    #     ax2 = fig.add_subplot(gs[0, 1])
    #     ax3 = fig.add_subplot(gs[1, 0])
    #     ax4 = fig.add_subplot(gs[1, 1])

    #     # The fifth plot occupying the third column
    #     ax5 = fig.add_subplot(gs[:, 2])

    #     # Plot the first four subplots in a 2x2 grid
    #     ax1.errorbar(predicted_a, artemis_results[0], xerr=a_unc, yerr=artemis_unc[0], fmt='o', label='Predicted', color='red')
    #     ax1.plot([artemis_results[0]-2, artemis_results[0]+2], [artemis_results[0]-2, artemis_results[0]+2], 'r--')
    #     #ax1.set_xlim(8, 10)
    #     #ax1.set_ylim(8, 10)
    #     ax1.set_title('A')
    #     ax1.set_xlabel('NN')
    #     ax1.set_ylabel('Artemis')
    #     # change tick font size
    #     ax1.tick_params(axis='both', which='major', labelsize=8)


    #     ax2.errorbar(predicted_deltar, artemis_results[1], xerr=deltar_unc, yerr=artemis_unc[1], fmt='o', label='Predicted', color='red')
    #     ax2.plot([artemis_results[1]-0.1, artemis_results[1]+0.1], [artemis_results[1]-0.1, artemis_results[1]+0.1], 'r--')
    #     #ax2.set_xlim(-0.02, 0)
    #     #ax2.set_ylim(-0.02, 0)
    #     ax2.set_title('Delta R')
    #     ax2.set_xlabel('NN')
    #     ax2.set_ylabel('Artemis')
    #     ax2.tick_params(axis='both', which='major', labelsize=8)

    #     ax3.errorbar(predicted_sigma2, artemis_results[2], xerr=sigma2_unc, yerr=artemis_unc[2], fmt='o', label='Predicted', color='red')
    #     ax3.plot([artemis_results[2]-0.02, artemis_results[2]+0.02], [artemis_results[2]-0.02, artemis_results[2]+0.02], 'r--')
    #     #ax3.set_xlim(0.003, 0.005)
    #     #ax3.set_ylim(0.003, 0.005)
    #     ax3.set_title('Sigma2')
    #     ax3.set_xlabel('NN')
    #     ax3.set_ylabel('Artemis')
    #     ax3.tick_params(axis='both', which='major', labelsize=8)

    #     ax4.errorbar(predicted_e0, artemis_results[3], xerr=e0_unc, yerr=artemis_unc[3], fmt='o', label='Predicted', color='red')
    #     ax4.plot([-10, 10], [-10, 10], 'r--')
    #     #ax4.set_xlim(-10, 10)
    #     #ax4.set_ylim(-10, 10)
    #     ax4.set_title('enot')
    #     ax4.set_xlabel('NN')
    #     ax4.set_ylabel('Artemis')
    #     ax4.tick_params(axis='both', which='major', labelsize=8)

    #     # Plot the fifth subplot
    #     ax5.plot(k_grid, interpolated_chi_q, label='Experimental', color='black')
    #     ax5.plot(k_grid, interpolated_artemis, label=f'Artemis MSE = {mse_artemis:.3f}', color='blue')
    #     ax5.set_xlim(2, 14)
    #     ax5.set_title('Q-space')
    #     ax5.legend()

    #     # Adjust layout
    #     plt.tight_layout()
    #     plt.savefig('exp_agreement.png')

    #     return print(f"{predicted_a=}, {predicted_deltar=}, {predicted_sigma2=}, {predicted_e0=}")




    def plot_chi(self, dataset_dir):
        data = read_ascii(dataset_dir)

        k_weight = 2

        if k_weight == 2:
            data.chi2 = data.chi * data.k ** 2

        plt_t.plot(data.k, data.chi2, label='Chi2')
        plt_t.xlabel('k')
        plt_t.xfrequency(20)
        plt_t.ylabel('Chi2')
        plt_t.title(f'{dataset_dir}')
        plt_t.show()
    
    def plot_r(self, dataset_dir):
        data = read_ascii(dataset_dir)

        k_weight = 2
        kmin = 2
        kmax = 14

        plt_t.clear_figure()

        if k_weight == 2:
            data.chi2 = data.chi * data.k ** 2
        xftf(data, kweight=k_weight, kmin=kmin, kmax=kmax)

        plt_t.plot(data.r, data.chir_mag, label='FT-EXAFS')
        plt_t.xlabel('r')
        plt_t.xfrequency(12)
        plt_t.ylabel('chir_mag')
        plt_t.title(f'{dataset_dir}')
        plt_t.show()




    # def fit_laplace(self):
    #     if not hasattr(self, "synthetic_spectra") or self.synthetic_spectra is None:
    #         self.load_synthetic_spectra()
    #
    #     if not hasattr(self, "data_loader") or self.data_loader is None:
    #         self.prepare_dataloader()
    #
    #     if not hasattr(self, "model") or self.model is None:
    #         self.load_model()
    #
    #     la_model = deepcopy(self.model)
    #
    #     la = Laplace(
    #         la_model,
    #         "regression",
    #         subset_of_weights="last_layer",
    #         hessian_structure="diag",
    #     )
    #
    #     la.fit(self.data_loader.train_dataloader(batch_size=self.batch_size))
    #     print("fit finished")
    #
    #     la.optimize_prior_precision("glm")
    #     print("optimize_prior_precision")
    #
    #     la.fit(self.data_loader.train_dataloader(batch_size=self.batch_size))
    #
    #     x_test, y_test = next(iter(self.data_loader.test_dataloader()))
    #
    #     f_mu, f_var = la(x_test.to(device=self.model.device))
    #
    #     f_mu = f_mu.squeeze().detach().cpu().numpy()
    #
    #     f_sigma = f_var.squeeze().detach().cpu().numpy()
    #
    #     pred_std = np.sqrt(f_sigma**2)
    #
    #     denormalize_data = self.denormalize_data(f_mu)
    #     denormalize_std = self.denormalize_data(pred_std)
    #     denormalize_true = self.denormalize_data(y_test)
    #
    #     for data, std, true in zip(denormalize_data, denormalize_std, denormalize_true):
    #         print(f"Predicted: {data}")
    #         print(f"Predicted std: {std}")
    #         print(f"True: {true}")
    #
    #         print()
    #     #
    #     # print(pred_std.shape)
    #     #
    #     # print()
    #     #
    #     # print(la.sigma_noise.item())
    #


# def main():
#
#     path = "./examples/training.toml"
#     crowpeas = CrowPeas()
#     # (
#     #     crowpeas.load_config(path)
#     #     .load_and_validate_config()
#     #     .init_synthetic_spectra()
#     #     .save_training_data()
#     #     .save_config(path)
#     # )
#     #
#     (
#         crowpeas.load_config(path)
#         .load_and_validate_config()
#         .load_synthetic_spectra()
#         .prepare_dataloader()
#         .save_config(path)
#         .train()
#     )
#
#
# if __name__ == "__main__":
#     main()
