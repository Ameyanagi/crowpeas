"""Console script for crowpeas.

This module implements the command-line interface for the crowpeas package,
providing commands for configuration generation, training, validation, and
prediction on experimental data.
"""

import typer
from rich.console import Console
from typing import Annotated, Optional
from pathlib import Path
import os
import shutil


# Directory containing package resources
DIR_PATH = Path(__file__).parent


# Create Typer app with help displayed when no arguments are provided
app = typer.Typer(
    no_args_is_help=True,
    help="EXAFS fitting tool using Neural Networks",
)

# Rich console for formatted output
console = Console()


@app.command()
def main(
    config: Annotated[
        Optional[str], typer.Argument(help="Config file path [toml or json]")
    ] = None,
    generate: Annotated[
        bool, typer.Option("--generate", "-g", help="Generate config file")
    ] = False,
    training: Annotated[
        bool,
        typer.Option(
            "--training",
            "-t",
            help="Option to train the model. True: Always train, False(default): Train only when configured in the config",
        ),
    ] = False,
    dataset: Annotated[
        bool,
        typer.Option(
            "--dataset",
            "-d",
            help="Option to generate dataset. True: Always generate, False(default): Generate only when configured in the config",
        ),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            "-r",
            help="Option to resume training. True: Resume training, False(default): Start training from scratch",
        ),
    ] = False,
    validate: Annotated[
        bool,
        typer.Option(
            "--validate",
            "-v",
            help="Option to validate the config file. True: Validate only, False(default): Validate and run",
        ),
    ] = False,
    experiment: Annotated[
        bool,
        typer.Option(
            "--experiment",
            "-e",
            help="Option to use a trained model to make predictions on experimental data. True: predict on experimental data, False(default): Do not predict on experimental data",
        ),
    ] = False,
    plot: Annotated[
        bool,
        typer.Option(
            "--plot",
            "-p",
            help="Option to plot experimental data. True: plot experimental data, False(default): Do not plot experimental data",
        ),
    ] = False,
    data_path: Annotated[
        Optional[str], typer.Option("--data-path", help="Data path required when using --plot option")
    ] = None,
):
    """
    Crowpeas: Neural Network-based EXAFS Analysis Tool

    This command-line interface provides functionality for training neural networks
    on EXAFS data, making predictions on experimental data, and evaluating results.

    Usage Examples:
        crowpeas -g                      # Generate a config file
        crowpeas -d -t -v config.toml    # Generate dataset, train model, and validate
        crowpeas -e config.toml          # Run predictions on experimental data
        crowpeas -p --data-path data.dat # Plot experimental data
    """
    if plot and not data_path:
        raise typer.BadParameter("Ploting requires data path!")

    if generate:
        if config is None:
            config = "config.toml"

        console.print("Generating config file")
        generate_config(config)
        console.print(
            "Config file generated. Please fill in the details and run the train command.\n"
        )

        console.print("Command: crowpeas -d -t -v <path_to_config_file>\n")
        console.print("For more help: crowpeas --help")

        return

    if config is not None:
        run_crowpeas(config, training, dataset, resume, validate, experiment)
        return

    if config is None and not plot:
        typer.echo(app.rich_help_panel)
        # console.print(app.rich_help_panel)

    if plot:
        plot_crowpeas(data_path)


def generate_config(config: str) -> None:
    """
    Generate a sample configuration file and required FEFF file.

    This function copies the sample config and FEFF files from the package
    to the specified location, providing a starting point for users.

    Args:
        config: Path where the configuration file should be created
    """
    config_path = os.path.join(DIR_PATH, "./sample_config.toml")
    feff_path = os.path.join(DIR_PATH, "./Pt_feff0001.dat")

    console.print(f"Generating config file at {config}")
    shutil.copyfile(config_path, config)
    console.print(f"Generating feff file at ./Pt_feff0001.dat")
    shutil.copyfile(feff_path, "./Pt_feff0001.dat")


def run_crowpeas(
    config: str,
    training: bool,
    dataset: bool,
    resume: bool,
    validate: bool,
    experiment: bool,
) -> None:
    """
    Run the main crowpeas workflow based on command-line options.

    This function orchestrates the crowpeas workflow, including dataset generation,
    model training, validation, and prediction on experimental data.

    Args:
        config: Path to the configuration file
        training: Whether to force training mode
        dataset: Whether to force dataset generation
        resume: Whether to resume training from a checkpoint
        validate: Whether to validate the model
        experiment: Whether to run on experimental data
    """
    from crowpeas.core import CrowPeas

    console.print(f"Running crowpeas with config file: {config}")

    # Initialize CrowPeas and load configuration
    cp = CrowPeas()
    cp.load_config(config)
    cp.load_and_validate_config()

    # Handle dataset generation/loading
    if dataset:
        console.print("Force dataset generation mode")
        cp.init_synthetic_spectra(generate=True)
        cp.save_training_data()
        cp.save_config()
    else:
        console.print("Trying to load dataset, if exists")
        cp.load_synthetic_spectra()
        cp.save_config()

    # Handle model loading for resuming training
    if resume:
        console.print("Read checkpoint and resume training")
        cp.load_model()

    # Handle model training
    if training:
        console.print("Force training mode")
        cp.training_mode = True
        cp.prepare_dataloader(setup="fit")
        cp.save_config()

        console.print("Training the model")
        cp.train()
        cp.plot_training_history()

    # Handle model validation
    if validate:
        console.print("Validating the config file")
        cp.validate_model()
        cp.plot_parity("/parity.png")
        console.print("The parity plot is generated at parity.png")
        cp.plot_test_spectra(1, save_path="/1.png")

    # Handle experimental prediction
    if experiment:
        console.print("Looking for experimental config file")
        cp.plot_results()
        cp.save_predictions_to_toml("predictions.toml")


def plot_crowpeas(dataset_dir: str) -> None:
    """
    Plot EXAFS data using crowpeas.

    This function creates visualizations of EXAFS data in both k-space and r-space.

    Args:
        dataset_dir: Path to the dataset directory containing EXAFS data
    """
    from crowpeas.core import CrowPeas

    console.print(f"Plotting {dataset_dir} with crowpeas")

    cp = CrowPeas()
    console.print("Chi k2")
    cp.plot_chi(dataset_dir)
    console.print("R")
    cp.plot_r(dataset_dir)

if __name__ == "__main__":
    app()
