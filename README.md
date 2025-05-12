# crowpeas

[![PyPI Version](https://img.shields.io/pypi/v/crowpeas.svg)](https://pypi.python.org/pypi/crowpeas)
[![Build Status](https://img.shields.io/travis/Ameyanagi/crowpeas.svg)](https://travis-ci.com/Ameyanagi/crowpeas)
[![Documentation Status](https://readthedocs.org/projects/crowpeas/badge/?version=latest)](https://crowpeas.readthedocs.io/en/latest/?version=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## EXAFS Fitting using Neural Networks

Crowpeas is a Python package for Extended X-ray Absorption Fine Structure (EXAFS) fitting using neural networks. It provides tools to generate synthetic spectral data, train various neural network models, and make predictions on experimental data.

![EXAFS Fitting Illustration](https://raw.githubusercontent.com/Ameyanagi/crowpeas/main/images/illustration.png)

## Features

- Generate synthetic EXAFS spectra for training
- Train neural network models:
  - Multi-Layer Perceptron (MLP)
  - Convolutional Neural Network (CNN) [in testing]
- Make predictions on experimental data with uncertainty estimation
- Visualize results and compare with traditional fitting approaches
- Optional PyTorch model compilation for faster execution (requires C++ compiler)

## Installation

### Automated Installation (Recommended)

The easiest way to install crowpeas is using the provided installation script, which:
1. Checks for and installs the uv package manager if needed
2. Sets up a virtual environment
3. Installs PyTorch with appropriate CUDA support
4. Installs crowpeas and its dependencies

```bash
# Clone the repository
git clone https://github.com/Ameyanagi/crowpeas.git
cd crowpeas

# Run the installation script
python install.py
```

After installation, activate the virtual environment:

```bash
# On Linux/macOS
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

### Manual Installation

If you prefer to install manually, follow these steps:

1. **Install PyTorch** (install with appropriate CUDA support for GPU acceleration):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

2. **Install crowpeas**:

```bash
pip install git+https://github.com/Ameyanagi/crowpeas
```

## Quick Start

### 1. Generate a Configuration File

```bash
crowpeas -g
```

This will create a `config.toml` file and a `Pt_feff0001.dat` file in your current directory.

### 2. Run the Full Workflow

```bash
crowpeas -d -t -v config.toml
```

This command:
- Generates a synthetic dataset (`-d`)
- Trains a neural network model (`-t`)
- Validates the model and creates plots (`-v`)

### 3. Make Predictions on Experimental Data

```bash
crowpeas -e config.toml
```

## Configuration

Crowpeas uses TOML configuration files to specify:
- Training dataset parameters
- Neural network architecture and hyperparameters
- Experimental data paths and settings

Example configuration:

```toml
[general]
title = "Pd training"
mode = "training"
output_dir = "output"

[training]
feffpath = "Pt_feff0001.dat"
training_set_dir = "training_set"
num_examples = 10000
input_type = "q"
k_range = [2.5, 12.5]
k_weight = 2

[neural_network.architecture]
type = "MLP"
activation = "gelu" 
output_dim = 4
hidden_dims = [1024, 516, 516, 516]
```

See [the documentation](https://crowpeas.readthedocs.io/) for detailed configuration options.

## Command-Line Interface

Crowpeas provides a rich command-line interface with the following options:

- `-g, --generate`: Generate a sample configuration file
- `-d, --dataset`: Generate a synthetic dataset
- `-t, --training`: Train a neural network model
- `-r, --resume`: Resume training from a checkpoint
- `-v, --validate`: Validate the model and create plots
- `-e, --experiment`: Make predictions on experimental data
- `-p, --plot`: Plot experimental data
- `--data-path`: Specify path to experimental data (with `-p`)

## Environment Variables

- `CROWPEAS_COMPILE`: Set to "1", "true", or "yes" to enable PyTorch model compilation for faster execution. Requires a working C++ compiler (default: disabled).

## Documentation

Full documentation is available at [https://crowpeas.readthedocs.io/](https://crowpeas.readthedocs.io/)

## Development

For development, install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

Then you can run:

```bash
# Run tests
pytest

# Run code style checks
ruff check .

# Run type checks
pyright
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use crowpeas in your research, please cite:

```
@article{MARCELLA2025116145,
  title = {First shell EXAFS data analysis of nanocatalysts via neural networks},
  journal = {Journal of Catalysis},
  volume = {447},
  pages = {116145},
  year = {2025},
  issn = {0021-9517},
  doi = {https://doi.org/10.1016/j.jcat.2025.116145},
  url = {https://www.sciencedirect.com/science/article/pii/S0021951725002106},
  author = {Nicholas Marcella and Ryuichi Shimogawa and Yongchun Xiang and Anatoly I. Frenkel},
  abstract = {Understanding the mechanisms of work of nanoparticle catalysts requires the knowledge of their structural and electronic descriptors, often measured in operando X-ray absorption fine structure (XAFS) spectroscopy experiments. We introduce a neural-network-based framework for rapidly mapping the extended XAFS (EXAFS) spectra onto structural parameters as an alternative to the commonly used non-linear least-squares fitting approaches. Our method leverages a multilayer perceptron trained on theoretical EXAFS and validated against theoretical test data and experimental spectra of frequently used nanoparticle types. The network helps lower the correlation between parameters, achieves high accuracy in the presence of noise and glitches, and can provide real-time parameter predictions with minimal user intervention. Parameter uncertainties are estimated as well. This method can be readily integrated into beamline pipelines or laboratory data analysis workflow and has the potential to accelerate high-throughput catalyst characterization and testing.}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.