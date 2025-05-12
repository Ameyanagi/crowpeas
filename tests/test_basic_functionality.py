"""Basic functionality tests for crowpeas."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np

# Add src to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crowpeas.core import CrowPeas
from crowpeas.data_generator import SyntheticSpectrum
from crowpeas.model.MLP import MLP


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality to ensure it's not broken during refactoring."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.test_dir, "test_config.toml")

        # Create a valid config file for testing
        with open(self.test_config_path, "w") as f:
            f.write('''[general]
title = "Test training"
mode = "training"
output_dir = "output"

[training]
feffpath = "Pt_feff0001.dat"
training_set_dir = "training_set"
training_data_prefix = "Test training_data"
num_examples = 10
spectrum_noise = false
noise_range = [0.0, 0.0]
input_type = "q"
k_range = [2.5, 12.5]
k_weight = 2
r_range = [1.7, 3.0]
train_size_ratio = 0.8
val_size_ratio = 0.2
test_size_ratio = 0.0
training_noise = true
training_noise_range = [0.0, 0.01]

[neural_network]
model_name = "Test"
model_dir = "model"
checkpoint_dir = "checkpoint"
checkpoint_name = "Test"

[training.param_ranges]
s02 = [0.75, 1.2]
degen = [4.0, 13.0]
deltar = [-0.2, 0.2]
sigma2 = [0.001, 0.02]
e0 = [-10.0, 10.0]

[neural_network.hyperparameters]
epochs = 10
batch_size = 2
learning_rate = 0.001

[neural_network.architecture]
type = "MLP"
activation = "leakyrelu"
output_activation = "linear"
output_dim = 4
hidden_dims = [516, 516]
dropout_rates = [0.0, 0.0]
weight_decay = 0.0
filter_sizes = [0]
kernel_sizes = [0]
''')

        # Copy feff file to test directory
        crowpeas_dir = Path(__file__).parent.parent
        feff_path = crowpeas_dir / "src" / "crowpeas" / "Pt_feff0001.dat"
        shutil.copy(feff_path, os.path.join(self.test_dir, "Pt_feff0001.dat"))

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def test_config_loading(self):
        """Test that configuration loading works."""
        cp = CrowPeas()
        cp.load_config(self.test_config_path)
        self.assertIsNotNone(cp.config)
        self.assertIn("general", cp.config)
        self.assertIn("neural_network", cp.config)
        self.assertIn("training", cp.config)

    @unittest.skip("Synthetic spectrum test needs more specific setup")
    def test_synthetic_spectrum(self):
        """Test synthetic spectrum generation."""
        # This test is skipped for now as it requires more specific setup
        pass

    def test_model_creation(self):
        """Test model creation."""
        # Create a simple MLP model
        model = MLP(
            hidden_layers=[10, 10],
            output_size=4,
            k_min=2.5,
            k_max=12.5,
            r_min=1.7,
            r_max=3.2,
            input_form="r",
            activation="relu",
        )
        
        # Check model attributes
        self.assertEqual(len(model.hidden_layers), 2)
        self.assertEqual(model.hidden_layers[0], 10)
        self.assertEqual(model.hidden_layers[1], 10)

    @unittest.skip("Model forward test requires more complex setup")
    def test_model_forward(self):
        """Test model forward pass."""
        # This test is skipped for now as it requires more complex setup
        pass


if __name__ == "__main__":
    unittest.main()