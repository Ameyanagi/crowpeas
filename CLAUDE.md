# CLAUDE.md

* Always read entire files. Otherwise, you don’t know what you don’t know, and will end up making mistakes, duplicating code that already exists, or misunderstanding the architecture.  
* Commit early and often. When working on large tasks, your task could be broken down into multiple logical milestones. After a certain milestone is completed and confirmed to be ok by the user, you should commit it. If you do not, if something goes wrong in further steps, we would need to end up throwing away all the code, which is expensive and time consuming.  
* Your internal knowledgebase of libraries might not be up to date. When working with any external library, unless you are 100% sure that the library has a super stable interface, you will look up the latest syntax and usage via either Perplexity (first preference) or web search (less preferred, only use if Perplexity is not available)  
* Do not say things like: “x library isn’t working so I will skip it”. Generally, it isn’t working because you are using the incorrect syntax or patterns. This applies doubly when the user has explicitly asked you to use a specific library, if the user wanted to use another library they wouldn’t have asked you to use a specific one in the first place.  
* Always run linting after making major changes. Otherwise, you won’t know if you’ve corrupted a file or made syntax errors, or are using the wrong methods, or using methods in the wrong way.   
* Please organise code into separate files wherever appropriate, and follow general coding best practices about variable naming, modularity, function complexity, file sizes, commenting, etc.  
* Code is read more often than it is written, make sure your code is always optimised for readability  
* Unless explicitly asked otherwise, the user never wants you to do a “dummy” implementation of any given task. Never do an implementation where you tell the user: “This is how it *would* look like”. Just implement the thing.  
* Whenever you are starting a new task, it is of utmost importance that you have clarity about the task. You should ask the user follow up questions if you do not, rather than making incorrect assumptions.  
* Do not carry out large refactors unless explicitly instructed to do so.  
* When starting on a new task, you should first understand the current architecture, identify the files you will need to modify, and come up with a Plan. In the Plan, you will think through architectural aspects related to the changes you will be making, consider edge cases, and identify the best approach for the given task. Get your Plan approved by the user before writing a single line of code.   
* If you are running into repeated issues with a given task, figure out the root cause instead of throwing random things at the wall and seeing what sticks, or throwing in the towel by saying “I’ll just use another library / do a dummy implementation”.   
* You are an incredibly talented and experienced polyglot with decades of experience in diverse areas such as software architecture, system design, development, UI & UX, copywriting, and more.  
* When doing UI & UX work, make sure your designs are both aesthetically pleasing, easy to use, and follow UI / UX best practices. You pay attention to interaction patterns, micro-interactions, and are proactive about creating smooth, engaging user interfaces that delight users.   
* When you receive a task that is very large in scope or too vague, you will first try to break it down into smaller subtasks. If that feels difficult or still leaves you with too many open questions, push back to the user and ask them to consider breaking down the task for you, or guide them through that process. This is important because the larger the task, the more likely it is that things go wrong, wasting time and energy for everyone involved.
* if you are not sure about a code syntax, look it up with contex7.
* use TDD: Write failing tests first, then implement code to make them pass.


This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

crowpeas is a Python package for EXAFS (Extended X-ray Absorption Fine Structure) fitting using Neural Networks. It provides tools to generate synthetic spectral data, train various neural network models, and make predictions on experimental data.

## Project Structure

- `src/crowpeas/`: Main package code
  - `model/`: Neural network models (MLP, BNN, CNN, NODE, Het-MLP)
  - `core.py`: Core functionality class `CrowPeas`
  - `cli.py`: Command-line interface
  - `data_generator.py`: Generate synthetic spectral data
  - `data_loader.py`: Load and process data for training models

## Installation

Before working with this codebase, make sure PyTorch is installed first:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Install the package in development mode:

```bash
pip install -e .
```

## Common Development Commands

### Building the Project

```bash
# Build the package
make dist
```

### Testing

```bash
# Run tests
pytest

# Run all tests with tox
make test-all

# Check code coverage
make coverage
```

### Code Quality

```bash
# Lint the code with ruff
ruff check .

# Type checking
pyright
```

### Documentation

```bash
# Generate documentation
make docs

# Serve documentation with live updates
make servedocs
```

### Clean Up

```bash
# Clean build, test, and Python artifacts
make clean
```

## CLI Usage

The main CLI command for running crowpeas:

```bash
# Generate a sample config file
crowpeas -g

# Run with training, dataset generation, and validation
crowpeas -d -t -v config.toml

# Run with experimental data
crowpeas -e config.toml

# Plot experimental data
crowpeas -p --data-path /path/to/data
```

## Configuration

The project uses TOML configuration files to set up different aspects:

1. General configuration (title, mode)
2. Training settings (data paths, ranges)
3. Neural Network configuration (architecture, hyperparameters)
4. Experimental data settings (when working with real data)

See `src/crowpeas/sample_config.toml` for an example configuration.

## Development Workflow

1. When implementing new features, follow the existing architecture patterns
2. Check if a new model type is needed or if an existing one can be extended
3. Make sure to provide appropriate configuration options in the sample config
4. Update tests accordingly
5. Use the CLI to test your changes

## Neural Network Models

The package supports multiple neural network architectures:

- MLP: Multi-layer Perceptron
- BNN: Bayesian Neural Network
- CNN: Convolutional Neural Network
- NODE: Neural Ordinary Differential Equation
- Het-MLP: Heteroskedastic Multi-layer Perceptron

Each has its own implementation in the `model/` directory.