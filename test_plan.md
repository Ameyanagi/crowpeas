# Test Plan for Crowpeas Refactoring

This document outlines the testing approach we'll use to ensure functionality isn't broken during the refactoring process.

## Core Functionality Tests

1. **Configuration Loading**
   - Test loading TOML configuration files
   - Test loading JSON configuration files
   - Test with missing required fields
   - Test with invalid field values

2. **Data Generation**
   - Test synthetic spectra generation
   - Test with and without noise
   - Test with different parameter ranges
   - Test generation of training/validation/test splits

3. **Model Training**
   - Test training MLP model with minimal epochs
   - Test training with checkpointing
   - Test resuming from checkpoints
   - Test validation during training

4. **Model Inference**
   - Test prediction on synthetic data
   - Test prediction on experimental data
   - Test with checkpointed models

5. **Visualization**
   - Test plotting of results
   - Test parity plot generation
   - Test training history visualization

## Test Cases for Different Model Types

For each model type (MLP, BNN, CNN, NODE, Het-MLP):
1. Test model initialization with valid parameters
2. Test forward pass with sample data
3. Test training with minimal epoch count
4. Test prediction on new data

## Integration Tests

1. **End-to-End Workflow**
   - Generate synthetic data
   - Train model
   - Save model checkpoint
   - Load model checkpoint
   - Predict on test data
   - Visualize results

2. **CLI Functionality**
   - Test `crowpeas -g` (config generation)
   - Test `crowpeas -d -t -v config.toml` (dataset, training, validation)
   - Test `crowpeas -e config.toml` (experimental prediction)
   - Test `crowpeas -p --data-path /path/to/data` (plotting)

## Testing Environment

We'll use the example data in the `examples/` folder with reduced training parameters:
- Small number of examples (10)
- Few epochs (10)
- Small batch size (2)
- Simple network architecture

## Approach

1. Create baseline tests for current functionality
2. Run tests after each significant refactoring step
3. Compare results to ensure consistency
4. Document any intentional behavioral changes

## Metrics to Track

1. Training loss curves
2. Validation loss
3. Prediction accuracy on test data
4. Model parameter outputs for known inputs