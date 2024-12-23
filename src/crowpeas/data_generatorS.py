import random
import numpy as np
from math import sin, pi, log, sqrt
from larch.xafs import path2chi, feffpath
import pandas as pd
from pathlib import Path
import json

class SyntheticSpectrumS:

    training_mode: bool
    num_examples: int
    spectra: np.ndarray
    parameters: np.ndarray
    k: np.ndarray | None
    k_weight: int
    spectrum_noise: bool
    k_range: tuple | list
    k_grid: np.ndarray

    def __init__(
        self,
        feff_path_file,
        path_parameters=None,
        param_ranges=None,
        training_mode=False,
        num_examples=1,
        spectrum_noise=False,
        noise_range=(0, 0.01),
        k_range=(2.5, 12.5),
        k_weight=2,
        seed=None,
        generate=True,
    ):
        self.training_mode = training_mode
        self.num_examples = num_examples

        path1 = feffpath(feff_path_file)
        self.k = path1.k

        # This should be k_weighted_chi and add k_weight
        self.kweighted_chi = None
        self.k_weight = k_weight
        self.spectrum_noise = spectrum_noise
        self.noise_range = noise_range
        self.k_range = k_range
        self.k_grid = np.linspace(k_range[0], k_range[1], num=100)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if generate:
            if self.training_mode:
                self.generate_training_examples(feff_path_file, param_ranges)
            else:
                self.generate_spectrum(feff_path_file, path_parameters, param_ranges)

    def generate_spectrum(
        self, feff_path_file, path_parameters=None, param_ranges=None
    ):
        path1 = feffpath(feff_path_file)

        if param_ranges:
            path_parameters = [
                random.uniform(param_ranges[key][0], param_ranges[key][1])
                for key in ["s02", "degen", "deltar", "sigma2", "e0"]
            ]

        if path_parameters:
            path1.s02 = path_parameters[0]
            path1.degen = path_parameters[1]
            path1.A = path1.s02 * path1.degen
            path1.deltar = path_parameters[2]
            path1.sigma2 = path_parameters[3]
            path1.e0 = path_parameters[4]

        path2chi(path1)

        if self.k is None:
            self.k = path1.k

        self.kweighted_chi = path1.chi * path1.k**self.k_weight

        if self.spectrum_noise:
            noise_level = random.uniform(self.noise_range[0], self.noise_range[1])
            noise = np.random.normal(0, noise_level, len(self.kweighted_chi))
            self.kweighted_chi += noise

        return path1.s02 * path1.degen, path1.deltar, path1.sigma2, path1.e0

    # def generate_training_examples(
    #     self, 
    #     feff_path_file, 
    #     param_ranges, 
    #     min_spectra_in_sequence=5, 
    #     max_spectra_in_sequence=20
    # ):
    #     """
    #     Generate training examples that consist of sequences of spectra with structured parameter variations.
    #     For each sequence:
    #     - Choose 1 to 3 parameters from ["s02", "degen", "deltar", "sigma2", "e0"] to vary.
    #     - Apply a random variation profile (linear, quadratic, sqrt, log, sinusoidal) over a randomly chosen
    #         number of steps between min_spectra_in_sequence and max_spectra_in_sequence.
    #     - Non-varying parameters remain constant.
    #     """

    #     parameters_list = ["s02", "degen", "deltar", "sigma2", "e0"]
    #     profile_types = ["linear_increase", "linear_decrease", "quadratic", "sqrt", "log", "sinusoidal"]

    #     tmp_spectra = []
    #     tmp_parameters = []
        
    #     for _ in range(self.num_examples):
    #         # Randomly choose sequence length for this example
    #         length = random.randint(min_spectra_in_sequence, max_spectra_in_sequence)

    #         # Choose how many parameters to vary (1 to 3)
    #         num_vary = random.randint(1, 3)
    #         vary_params = random.sample(parameters_list, num_vary)
    #         # The others remain constant
    #         fixed_params = [p for p in parameters_list if p not in vary_params]

    #         # Assign fixed parameters
    #         fixed_values = {}
    #         for p in fixed_params:
    #             start_val = random.uniform(param_ranges[p][0], param_ranges[p][1])
    #             fixed_values[p] = start_val

    #         # Assign start/end values and profile for varying parameters
    #         vary_values = {}
    #         for p in vary_params:
    #             start_val = random.uniform(param_ranges[p][0], param_ranges[p][1])
    #             end_val = random.uniform(param_ranges[p][0], param_ranges[p][1])
    #             profile_type = random.choice(profile_types)
    #             vary_values[p] = self._generate_profile(length, start_val, end_val, profile_type)

    #         # Now build the sequence of parameters and spectra
    #         sequence_spectra = []
    #         sequence_params = []
    #         for i in range(length):
    #             # Construct the parameter set for this step
    #             step_params = {}
    #             for p in fixed_params:
    #                 step_params[p] = fixed_values[p]
    #             for p in vary_params:
    #                 step_params[p] = vary_values[p][i]

    #             # Compute A = s02 * degen
    #             A = step_params["s02"] * step_params["degen"]
    #             # Generate one spectrum
    #             # generate_spectrum expects path_parameters in order: s02, degen, deltar, sigma2, e0
    #             path_parameters = [
    #                 step_params["s02"],
    #                 step_params["degen"],
    #                 step_params["deltar"],
    #                 step_params["sigma2"],
    #                 step_params["e0"],
    #             ]
    #             self.generate_spectrum(feff_path_file, path_parameters=path_parameters)
    #             sequence_spectra.append(self._interpolate_if_needed(self.k, self.kweighted_chi, self.k_grid))
    #             # Store parameters as (A, deltar, sigma2, e0) to match original code
    #             sequence_params.append([A, step_params["deltar"], step_params["sigma2"], step_params["e0"]])

    #         tmp_spectra.append(sequence_spectra)
    #         tmp_parameters.append(sequence_params)

    #     self.spectra = np.array(tmp_spectra)      # shape: [num_examples, variable_length, len(k_grid)]
    #     self.parameters = np.array(tmp_parameters) # shape: [num_examples, variable_length, 4]

    # def generate_training_examples(
    #     self, 
    #     feff_path_file, 
    #     param_ranges, 
    #     min_spectra_in_sequence=5, 
    #     max_spectra_in_sequence=20
    # ):
    #     """
    #     Generate training examples that consist of sequences of spectra with structured parameter variations.
    #     For each sequence:
    #     - Choose 1 to 3 parameters to vary.
    #     - Apply a random variation profile.
    #     - Sequence lengths vary between min_spectra_in_sequence and max_spectra_in_sequence.
    #     """

    #     parameters_list = ["s02", "degen", "deltar", "sigma2", "e0"]
    #     profile_types = ["linear_increase", "linear_decrease", "quadratic", "sqrt", "log", "sinusoidal"]

    #     tmp_spectra = []
    #     tmp_parameters = []
        
    #     for _ in range(self.num_examples):
    #         # Randomly choose sequence length for this example
    #         length = random.randint(min_spectra_in_sequence, max_spectra_in_sequence)

    #         # Choose how many parameters to vary (1 to 3)
    #         num_vary = random.randint(1, 3)
    #         vary_params = random.sample(parameters_list, num_vary)
    #         # The others remain constant
    #         fixed_params = [p for p in parameters_list if p not in vary_params]

    #         # Assign fixed parameters
    #         fixed_values = {}
    #         for p in fixed_params:
    #             start_val = random.uniform(param_ranges[p][0], param_ranges[p][1])
    #             fixed_values[p] = start_val

    #         # Assign start/end values and profile for varying parameters
    #         vary_values = {}
    #         for p in vary_params:
    #             start_val = random.uniform(param_ranges[p][0], param_ranges[p][1])
    #             end_val = random.uniform(param_ranges[p][0], param_ranges[p][1])
    #             profile_type = random.choice(profile_types)
    #             vary_values[p] = self._generate_profile(length, start_val, end_val, profile_type)

    #         # Now build the sequence of parameters and spectra
    #         sequence_spectra = []
    #         sequence_params = []
    #         for i in range(length):
    #             # Construct the parameter set for this step
    #             step_params = {fp: fixed_values[fp] for fp in fixed_params}
    #             for vp in vary_params:
    #                 step_params[vp] = vary_values[vp][i]

    #             # Compute A = s02 * degen
    #             A = step_params["s02"] * step_params["degen"]
    #             # generate_spectrum expects path_parameters in order: s02, degen, deltar, sigma2, e0
    #             path_parameters = [
    #                 step_params["s02"],
    #                 step_params["degen"],
    #                 step_params["deltar"],
    #                 step_params["sigma2"],
    #                 step_params["e0"],
    #             ]
    #             self.generate_spectrum(feff_path_file, path_parameters=path_parameters)
    #             #interp_spectrum = self._interpolate_if_needed(self.k, self.kweighted_chi, self.k_grid)
    #             #sequence_spectra.append(interp_spectrum)
    #             sequence_spectra.append(self.kweighted_chi)
    #             # Store parameters as (A, deltar, sigma2, e0) to match original code
    #             sequence_params.append([A, step_params["deltar"], step_params["sigma2"], step_params["e0"]])

    #         tmp_spectra.append(sequence_spectra)
    #         tmp_parameters.append(sequence_params)

    #     # Now we have lists of sequences with variable lengths
    #     # Find max length
    #     max_len = max(len(seq) for seq in tmp_spectra)

    #     ## Get the dimension of k_grid and parameters per step
    #     #k_len = len(self.k_grid)
    #     k_len = 401  # Hardcoded for now
    #     param_dim = 4  # (A, deltar, sigma2, e0)

    #     # Initialize arrays with zeros for padding
    #     # shape: [num_examples, max_len, k_len] and [num_examples, max_len, 4]
    #     padded_spectra = np.zeros((self.num_examples, max_len, k_len), dtype=float)
    #     padded_parameters = np.zeros((self.num_examples, max_len, param_dim), dtype=float)

    #     for i, (spec_seq, param_seq) in enumerate(zip(tmp_spectra, tmp_parameters)):
    #         seq_len = len(spec_seq)
    #         padded_spectra[i, :seq_len, :] = spec_seq
    #         padded_parameters[i, :seq_len, :] = param_seq

    #     self.spectra = padded_spectra
    #     self.parameters = padded_parameters

    def generate_training_examples(
        self, 
        feff_path_file, 
        param_ranges, 
        min_spectra_in_sequence=5, 
        max_spectra_in_sequence=20
    ):
        """
        Generate training examples that consist of sequences of spectra with structured parameter variations.
        Also generates masks indicating valid (non-padded) entries.
        """
        parameters_list = ["s02", "degen", "deltar", "sigma2", "e0"]
        profile_types = ["linear_increase", "linear_decrease", "quadratic", "sqrt", "log", "sinusoidal"]

        tmp_spectra = []
        tmp_parameters = []
        tmp_masks = []

        for _ in range(self.num_examples):
            # Randomly choose sequence length for this example
            length = random.randint(min_spectra_in_sequence, max_spectra_in_sequence)

            # Choose how many parameters to vary (1 to 3)
            num_vary = random.randint(1, 3)
            vary_params = random.sample(parameters_list, num_vary)
            # The others remain constant
            fixed_params = [p for p in parameters_list if p not in vary_params]

            # Assign fixed parameters
            fixed_values = {}
            for p in fixed_params:
                fixed_values[p] = random.uniform(param_ranges[p][0], param_ranges[p][1])

            # Assign start/end values and profile for varying parameters
            vary_values = {}
            for p in vary_params:
                start_val = random.uniform(param_ranges[p][0], param_ranges[p][1])
                end_val = random.uniform(param_ranges[p][0], param_ranges[p][1])
                profile_type = random.choice(profile_types)
                vary_values[p] = self._generate_profile(length, start_val, end_val, profile_type)

            # Now build the sequence of parameters and spectra
            sequence_spectra = []
            sequence_params = []
            for i in range(length):
                # Construct the parameter set for this step
                step_params = {fp: fixed_values[fp] for fp in fixed_params}
                for vp in vary_params:
                    step_params[vp] = vary_values[vp][i]

                # Compute A = s02 * degen
                A = step_params["s02"] * step_params["degen"]
                # generate_spectrum expects path_parameters in order: s02, degen, deltar, sigma2, e0
                path_parameters = [
                    step_params["s02"],
                    step_params["degen"],
                    step_params["deltar"],
                    step_params["sigma2"],
                    step_params["e0"],
                ]
                self.generate_spectrum(feff_path_file, path_parameters=path_parameters)
                sequence_spectra.append(self.kweighted_chi)
                # Store parameters as (A, deltar, sigma2, e0) to match original code
                sequence_params.append([A, step_params["deltar"], step_params["sigma2"], step_params["e0"]])

            # Pad the sequences to `max_spectra_in_sequence` with zeros
            # Assuming kweighted_chi is of length k_len (401 as in original code)
            # sequence_spectra is a list of arrays, each of length k_len
            # sequence_params is a list of lists, each of length 4

            # Pad spectra
            if len(sequence_spectra) < max_spectra_in_sequence:
                num_padding = max_spectra_in_sequence - len(sequence_spectra)
                padding_spectra = [np.zeros_like(sequence_spectra[0]) for _ in range(num_padding)]
                sequence_spectra.extend(padding_spectra)
            else:
                # Truncate if necessary
                sequence_spectra = sequence_spectra[:max_spectra_in_sequence]

            # Pad parameters
            if len(sequence_params) < max_spectra_in_sequence:
                num_padding = max_spectra_in_sequence - len(sequence_params)
                padding_params = [[0.0, 0.0, 0.0, 0.0] for _ in range(num_padding)]
                sequence_params.extend(padding_params)
            else:
                # Truncate if necessary
                sequence_params = sequence_params[:max_spectra_in_sequence]

            # Create mask: 1 for valid entries, 0 for padded
            mask = [1] * length + [0] * (max_spectra_in_sequence - length)

            tmp_spectra.append(sequence_spectra)
            tmp_parameters.append(sequence_params)
            tmp_masks.append(mask)

        # Convert to numpy arrays
        self.spectra = np.array(tmp_spectra)      # shape: [num_examples, max_len, k_len]
        self.parameters = np.array(tmp_parameters) # shape: [num_examples, max_len, 4]
        self.masks = np.array(tmp_masks)         # shape: [num_examples, max_len]



    def _generate_profile(self, length, start_val, end_val, profile_type):
        """
        Generate a profile of `length` steps from start_val to end_val according to profile_type.
        Available profiles: linear_increase, linear_decrease, quadratic, sqrt, log, sinusoidal
        """
        x = np.linspace(0, 1, length)
        if profile_type == "linear_increase":
            y = start_val + (end_val - start_val)*x
        elif profile_type == "linear_decrease":
            y = start_val + (end_val - start_val)*x
            # Decreasing if end_val < start_val. If not guaranteed, we rely on values chosen.
        elif profile_type == "quadratic":
            # Quadratic from start to end, let's say y = start_val + (end_val - start_val)*x^2
            y = start_val + (end_val - start_val)*(x**2)
        elif profile_type == "sqrt":
            # sqrt profile: vary from start to end using sqrt(x). Ensure start_val and end_val are > 0
            # If they're not, we can shift them, but let's assume they are positive or handle negative by absolute value.
            # We'll do a simple sqrt scaling:
            y = start_val + (end_val - start_val)*np.sqrt(x)
        elif profile_type == "log":
            # Log profile: from start to end using log(1 + 9x) to have a range
            # Make sure start_val and end_val are positive for log
            # If not, we could offset them, but let's assume param_ranges ensure positivity for these params when needed
            y = start_val + (end_val - start_val)*(np.log(1+9*x)/np.log(10))  # log base 10 scale
        elif profile_type == "sinusoidal":
            # Oscillate between start_val and end_val using a sine wave
            # sine wave oscillation: mean = (start+end)/2, amplitude = (end-start)/2
            mean_val = (start_val + end_val)/2
            amp = (end_val - start_val)/2
            y = mean_val + amp * np.sin(2*pi*x)
        else:
            # fallback linear
            y = start_val + (end_val - start_val)*x
        return y

    def _interpolate_if_needed(self, original_k, original_spectrum, target_k):
        """
        Interpolate the spectrum onto target_k if needed.
        """
        if len(original_k) != len(target_k) or not np.allclose(original_k, target_k):
            return np.interp(target_k, original_k, original_spectrum)
        else:
            return original_spectrum


    def get_training_data(self):
        if not self.training_mode:
            raise ValueError("Training mode is not active. No training data available.")

        if self.spectra is None or self.parameters is None or self.masks is None:
            raise ValueError("No training data available.")
            
        return self.spectra, self.parameters, self.masks


    def save_training_data(self, prefix="./synthetic"):
        if not self.training_mode:
            raise ValueError("Training mode is not active. No data to save.")

        filename_spectra = f"{prefix}_spectra.feather"
        filename_parameters = f"{prefix}_parameters.feather"
        filename_masks = f"{prefix}_masks.feather"
        filename_meta = f"{prefix}_metadata.json"

        # Save original shapes
        spectra_shape = self.spectra.shape
        params_shape = self.parameters.shape
        masks_shape = self.masks.shape

        # Flatten if multi-dimensional
        spectra_2d = self.spectra.reshape(spectra_shape[0], -1)
        parameters_2d = self.parameters.reshape(params_shape[0], -1)
        masks_2d = self.masks.reshape(masks_shape[0], -1)

        spectra_df = pd.DataFrame(spectra_2d)
        parameters_df = pd.DataFrame(parameters_2d)
        masks_df = pd.DataFrame(masks_2d)

        # Handle parameter column naming as before
        if params_shape[-1] == 4 and len(params_shape) == 3:
            # If shape is (num_examples, seq_length, 4)
            # After reshape: (num_examples, seq_length*4)
            # Columns can be named systematically.
            seq_length = params_shape[1]
            param_cols = []
            param_names = ["A", "deltar", "sigma2", "e0"]
            for s in range(seq_length):
                for name in param_names:
                    param_cols.append(f"{name}_step{s}")
            parameters_df = pd.DataFrame(parameters_2d, columns=param_cols)
        elif params_shape[-1] == 4 and len(params_shape) == 2:
            # Simple 2D case: (num_examples, 4)
            parameters_df = pd.DataFrame(parameters_2d, columns=["A", "deltar", "sigma2", "e0"])
        else:
            # Generic case: no named columns
            parameters_df = pd.DataFrame(parameters_2d)

        spectra_df.to_feather(filename_spectra)
        parameters_df.to_feather(filename_parameters)
        masks_df.to_feather(filename_masks)

        # Save metadata (shapes) so we can restore later
        meta_data = {
            "spectra_shape": spectra_shape,
            "parameters_shape": params_shape,
            "masks_shape": masks_shape,
        }
        with open(filename_meta, "w") as f:
            json.dump(meta_data, f)

        print(f"Spectra data saved to {filename_spectra}")
        print(f"Parameters data saved to {filename_parameters}")
        print(f"Masks data saved to {filename_masks}")
        print(f"Metadata saved to {filename_meta}")


    def exists_training_data(self, prefix="./synthetic"):
        filename_spectra = f"{prefix}_spectra.feather"
        filename_parameters = f"{prefix}_parameters.feather"
        filename_meta = f"{prefix}_metadata.json"

        if Path(filename_spectra).is_file() and Path(filename_parameters).is_file() and Path(filename_meta).is_file():
            return True
        return False

    def load_training_data(self, prefix="./synthetic"):
        filename_spectra = f"{prefix}_spectra.feather"
        filename_parameters = f"{prefix}_parameters.feather"
        filename_masks = f"{prefix}_masks.feather"
        filename_meta = f"{prefix}_metadata.json"

        if not Path(filename_spectra).is_file() or not Path(filename_parameters).is_file() \
        or not Path(filename_masks).is_file() or not Path(filename_meta).is_file():
            raise FileNotFoundError("Required training data or metadata files are missing.")

        spectra_2d = pd.read_feather(filename_spectra).values
        parameters_2d = pd.read_feather(filename_parameters).values
        masks_2d = pd.read_feather(filename_masks).values

        with open(filename_meta, "r") as f:
            meta_data = json.load(f)

        spectra_shape = tuple(meta_data["spectra_shape"])
        parameters_shape = tuple(meta_data["parameters_shape"])
        masks_shape = tuple(meta_data["masks_shape"])

        # Reshape back to original
        self.spectra = spectra_2d.reshape(spectra_shape)
        self.parameters = parameters_2d.reshape(parameters_shape)
        self.masks = masks_2d.reshape(masks_shape)

        print(f"Spectra data loaded from {filename_spectra}")
        print(f"Parameters data loaded from {filename_parameters}")
        print(f"Masks data loaded from {filename_masks}")


    def free_training_data(self):
        del self.spectra
        del self.parameters
        print("Training data has been freed.")
