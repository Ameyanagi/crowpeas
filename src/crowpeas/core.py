"""Main module."""

import toml
import os
from typing import Sequence, Literal
import json

from .data_generator import SyntheticSpectrum
from .data_loader import CrowPeasDataModule
import lightning as pl
from .model.BNN import BNN
from .model.MLP import MLP
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np

from .utils import (
    normalize_data,
    denormalize_data,
    normalize_spectra,
    denormalize_spectra,
    interpolate_spectrum,
)
from larch.xafs import xftf, xftr, feffpath, path2chi, ftwindow
from larch.io import read_ascii
from larch.fitting import param, guess, param_group
from larch import Group 
import matplotlib.pyplot as plt

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
    nn_type: Literal["MLP", "BNN"]
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
        required_for_expertiment = [
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
            for param in required_for_expertiment:
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

        trainer = pl.Trainer(max_epochs=self.epochs, callbacks=[callback])
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

    def predict_on_experimental_data(self):

        krange = self.config["experiment"]["k_range"]
        kmin = krange[0]
        kmax = krange[1]
        k_grid = self.config["neural_network"]["k_grid"]

        exp_data_path = self.config["experiment"]["dataset_dir"]
        pt_data  = read_ascii(exp_data_path)
        interpolated_chi_k = interpolate_spectrum(pt_data.k, pt_data.chi, k_grid)
        interpolated_chi_k = torch.tensor(interpolated_chi_k).unsqueeze(0)
        xftf(pt_data, kweight=0, kmin=kmin, kmax=kmax)
        xftr(pt_data, rmin=1.7, rmax=3.2)
        interpolated_chi_q = interpolate_spectrum(pt_data.q, pt_data.chiq_re, k_grid)

        self.predict_and_denormalize(interpolated_chi_k)

        predicted_a, predicted_deltar, predicted_sigma2, predicted_e0 = self.denormalized_test_pred[0]

        # path_predicted = feffpath(self.feff_path_file)
        # path_predicted.s02 = 1
        # path_predicted.degen = predicted_a
        # path_predicted.deltar = predicted_deltar
        # path_predicted.sigma2 = predicted_sigma2
        # path_predicted.e0 = predicted_e0
        # path2chi(path_predicted)
        # xftf(path_predicted, kweight=2, kmin=kmin, kmax=kmax)
        # xftr(path_predicted, rmin=1.7, rmax=3.2)

        # interpolated_predicted = interpolate_spectrum(path_predicted.q, path_predicted.chiq_re, k_grid)


        # path_artemis = feffpath(self.feff_path_file)
        # path_artemis.s02 = 1
        # path_artemis.degen = 9.276
        # path_artemis.deltar = -0.007
        # path_artemis.sigma2 = 0.00417
        # path_artemis.e0 = 9.022
        # path2chi(path_artemis)
        # xftf(path_artemis, kweight=2, kmin=kmin, kmax=kmax)
        # xftr(path_artemis, rmin=1.7, rmax=3.2)

        # interpolated_artemis = interpolate_spectrum(path_artemis.q, path_artemis.chiq_re, k_grid)

        return print(f"{predicted_a=}, {predicted_deltar=}, {predicted_sigma2=}, {predicted_e0=}")


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
