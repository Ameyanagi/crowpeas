"""Main module."""

import toml
import os
from typing import Sequence, Literal
import json

from data_generator import SyntheticSpectrum
from data_loader import CrowPeasDataModule
import lightning as pl
from model.BNN import BNN
from model.MLP import MLP
import torch
from lightning.pytorch.callbacks import ModelCheckpoint


class CrowPeas:

    # general parameters
    title: str = ""
    config_filename: str
    config_dir: str
    config: dict
    norm_params: dict

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

        self.title = self.config["general"]["title"]
        self.norm_params = self.config["general"].get("norm_params", None)

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
        self.config = {
            "general": {
                "title": self.title,
                "mode": "training" if self.training_mode else "inference",
                "seed": self.seed,
                "norm_params": self.norm_params,
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

        return self

    def load_synthetic_spectra(self):
        if not hasattr(self, "synthetic_spectra") or self.synthetic_spectra is None:
            self.init_synthetic_spectra(generate=False)

        self.synthetic_spectra.load_training_data(
            os.path.join(self.training_set_dir, self.training_data_prefix)
        )

        return self

    def prepare_dataloader(self, setup="fit"):
        if self.synthetic_spectra is None:
            self.init_synthetic_spectra()

        if self.seed is None:
            seed = 42
        else:
            seed = self.seed

        self.data_loader = CrowPeasDataModule(
            spectra=self.synthetic_spectra.spectra,
            parameters=self.synthetic_spectra.parameters,
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
                r_min=self.param_ranges["deltar"][0],
                r_max=self.param_ranges["deltar"][1],
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
                r_min=self.param_ranges["deltar"][0],
                r_max=self.param_ranges["deltar"][1],
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

    def train(self):

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
        checkpoint = os.path.join(self.checkpoint_dir, self.checkpoint_name)

        trainer = pl.Trainer(max_epochs=self.epochs, callbacks=[callback])
        trainer.fit(
            self.model,
            self.data_loader.train_dataloader(batch_size=self.batch_size),
            self.data_loader.val_dataloader(batch_size=self.batch_size),
        )

        return self


def main():

    path = "./examples/training.toml"
    crowpeas = CrowPeas()
    # (
    #     crowpeas.load_config(path)
    #     .load_and_validate_config()
    #     .init_synthetic_spectra()
    #     .save_training_data()
    #     .save_config(path)
    # )
    #
    (
        crowpeas.load_config(path)
        .load_and_validate_config()
        .load_synthetic_spectra()
        .prepare_dataloader()
        .train()
    )


if __name__ == "__main__":
    main()
