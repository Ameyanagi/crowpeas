"""Console script for crowpeas."""

import typer
from rich.console import Console
from typing import Annotated
from pathlib import Path
import os
import shutil


DIR_PATH = Path(__file__).parent


app = typer.Typer(
    no_args_is_help=True,
    # callback=lambda: typer.echo(app.rich_help_panel),
)
console = Console()


@app.command()
def main(
    config: Annotated[
        str | None, typer.Argument(help="Config file path [toml or json]")
    ] = None,
    generate: Annotated[
        bool, typer.Option("--generate", "-g", help="Generate config file")
    ] = False,
    training: Annotated[
        bool,
        typer.Option(
            "--training",
            "-t",
            help="Option to train the model. True: Always train, False(default): Train only when it is configured in the config",
        ),
    ] = False,
    dataset: Annotated[
        bool,
        typer.Option(
            "--dataset",
            "-d",
            help="Option to generate dataset. True: Always generate, False(default): Generate only when it is configured in the config",
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
            help="Option to use a trained model to make predictions on experimnetal data. True: predict on experimental data, False(default): Do not predict on experimental data",
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
        str | None, typer.Option("--data-path", help="Data path required when using --plot option")
    ] = None,
):
    """
    Crowpeas is a tool to perform a neural network based EXAFS analysis.
    This cli will provide a training, prediction and evaluation interface.
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


def generate_config(config: str):
    """
    Generate a config file for the crowpeas tool.
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
):
    from crowpeas.core import CrowPeas

    console.print(f"Running crowpeas with config file: {config}")

    cp = CrowPeas()
    cp.load_config(config)
    cp.load_and_validate_config()

    if dataset:
        console.print("Force dataset generation mode")
        cp.init_synthetic_spectra(generate=True)
        cp.save_training_data()
        cp.save_config()

    else:
        console.print("Trying to load dataset, if exists")
        cp.load_synthetic_spectra()
        cp.save_config()

    if resume:
        console.print("Read checkpoint and resume training")
        cp.load_model()

    if training:
        console.print("Force training mode")
        cp.training_mode = True
        cp.prepare_dataloader(setup="fit")
        cp.save_config()

        console.print("Training the model")
        cp.train()

    if validate:
        console.print("Validating the config file")
        cp.validate_model()
        cp.plot_parity("./parity.png")
        console.print("The parity plot is generated at parity.png")
        cp.plot_test_spectra(1, save_path="./1.png")

    if experiment:
        console.print("Looking for experimental config file")
        #cp.predict_on_experimental_data()
        cp.plot_results()
        cp.save_predictions_to_toml("predictions.toml")
        

    return

def plot_crowpeas(
    dataset_dir: str | None,
):
    from crowpeas.core import CrowPeas

    console.print(f"Plotting {dataset_dir} with crowpeas")

    cp = CrowPeas()
    console.print("Chi k2")
    cp.plot_chi(dataset_dir)
    console.print("R")
    cp.plot_r(dataset_dir)

if __name__ == "__main__":
    app()
