import numpy as np
import torch
import matplotlib.pyplot as plt


def normalize_data(
    data: np.ndarray, 
    feature_range: tuple = (0, 1),
    parameters: dict = None
) -> tuple[np.ndarray, dict]:
    """Normalize the data to the specified feature range.

    Args:
        data (np.ndarray): The data to be normalized.
        feature_range (tuple): The desired range of transformed data.
        parameters (dict, optional): Dictionary with pre-computed normalization parameters.

    Returns:
        np.ndarray: Normalized data.
        dict: Dictionary containing the min and max values for each feature.
    """
    if parameters is None:
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        feature_min = np.min(feature_range)
        feature_max = np.max(feature_range)
        norm_params = {
            "min": min_val,
            "max": max_val,
            "feature_min": feature_min,
            "feature_max": feature_max,
        }
    else:
        norm_params = parameters
        min_val = norm_params["min"]
        max_val = norm_params["max"]
        feature_min = norm_params["feature_min"]
        feature_max = norm_params["feature_max"]

    scale = feature_max - feature_min
    normalized_data = feature_min + (data - min_val) / (max_val - min_val) * scale
    return normalized_data, norm_params

def normalize_spectra(
    spectra: np.ndarray,
    parameters: dict = None
) -> tuple[np.ndarray, dict]:
    """Normalize spectra by dividing by the maximum absolute value.

    Args:
        spectra (np.ndarray): The spectra to be normalized.
        parameters (dict, optional): Dictionary with pre-computed max_abs_val.

    Returns:
        np.ndarray: Normalized spectra.
        dict: Dictionary containing the max_abs_val.
    """
    if parameters is None:
        min_val = np.min(spectra)
        max_val = np.max(spectra)
        max_abs_val = np.max([np.abs(min_val), np.abs(max_val)])
        norm_params = {"max_abs_val": max_abs_val}
    else:
        norm_params = parameters
        max_abs_val = norm_params["max_abs_val"]

    normalized_spectra = spectra / max_abs_val
    return normalized_spectra, norm_params



def normalize_data_S(
    data: np.ndarray, 
    mask: np.ndarray, 
    feature_range: tuple = (0, 1)
) -> tuple[np.ndarray, dict]:
    """
    Normalize the data to the specified feature range, excluding padded entries.

    Args:
        data (np.ndarray): The data to be normalized. Shape: [num_examples, max_len, features]
        mask (np.ndarray): Boolean mask indicating non-padded entries. Shape: [num_examples, max_len]
        feature_range (tuple): The desired range of transformed data.

    Returns:
        np.ndarray: Normalized data with padded entries remaining zero.
        dict: Dictionary containing the min and max values for each feature.
    """
    feature_min, feature_max = feature_range
    normalized_data = np.zeros_like(data, dtype=float)

    # Reshape for easier masking
    num_examples, max_len, num_features = data.shape
    reshaped_data = data.reshape(-1, num_features)
    reshaped_mask = mask.reshape(-1)

    # Select non-padded data
    valid_data = reshaped_data[reshaped_mask]

    # Compute min and max per feature
    min_val = valid_data.min(axis=0)
    max_val = valid_data.max(axis=0)
    
    # Avoid division by zero
    scale = max_val - min_val
    scale[scale == 0] = 1.0

    # Normalize valid data
    normalized_valid = feature_min + ((valid_data - min_val) / scale) * (feature_max - feature_min)
    
    # Assign normalized values back
    reshaped_normalized = normalized_data.reshape(-1, num_features)
    reshaped_normalized[reshaped_mask] = normalized_valid

    # Reshape back to original shape
    normalized_data = reshaped_normalized.reshape(num_examples, max_len, num_features)

    norm_params = {
        "min": min_val.tolist(),
        "max": max_val.tolist(),
        "feature_min": feature_min,
        "feature_max": feature_max,
    }

    return normalized_data, norm_params

def normalize_spectra_S(
    spectra: np.ndarray, 
    mask: np.ndarray
) -> tuple[np.ndarray, dict]:
    """
    Normalize the spectra, excluding padded entries.

    Args:
        spectra (np.ndarray): The spectra data to be normalized. Shape: [num_examples, max_len, k_len]
        mask (np.ndarray): Boolean mask indicating non-padded entries. Shape: [num_examples, max_len]

    Returns:
        np.ndarray: Normalized spectra with padded entries remaining zero.
        dict: Dictionary containing the max absolute value used for normalization.
    """
    normalized_spectra = np.zeros_like(spectra, dtype=float)

    # Reshape for easier masking
    num_examples, max_len, k_len = spectra.shape
    reshaped_spectra = spectra.reshape(-1, k_len)
    reshaped_mask = mask.reshape(-1)

    # Select non-padded data
    valid_spectra = reshaped_spectra[reshaped_mask]

    # Compute max absolute value across all valid spectra
    max_abs_val = np.max(np.abs(valid_spectra))
    if max_abs_val == 0:
        max_abs_val = 1.0  # Prevent division by zero

    # Normalize valid spectra
    normalized_valid_spectra = valid_spectra / max_abs_val

    # Assign normalized spectra back
    reshaped_normalized = normalized_spectra.reshape(-1, k_len)
    reshaped_normalized[reshaped_mask] = normalized_valid_spectra

    # Reshape back to original shape
    normalized_spectra = reshaped_normalized.reshape(num_examples, max_len, k_len)

    norm_params = {
        "max_abs_val": max_abs_val,
    }

    return normalized_spectra, norm_params



def interpolate_spectrum(
    original_k: np.ndarray, original_spectrum: np.ndarray, target_k: np.ndarray
) -> np.ndarray:
    """Interpolate the original spectrum onto the target energy mesh.

    Args:
        original_k (np.ndarray): The original energy mesh.
        original_spectrum (np.ndarray): The original spectrum values.
        target_k (np.ndarray): The target energy mesh.

    Returns:
        np.ndarray: Interpolated spectrum values on the target energy mesh.
    """
    return np.interp(target_k, original_k, original_spectrum)


def denormalize_data(normalized_data: np.ndarray, norm_params: dict) -> np.ndarray:
    """Denormalize the data from the specified feature range to the original range.

    Args:
        normalized_data (np.ndarray): The data to be denormalized.
        norm_params (dict): Dictionary containing the min and max values for each feature.
        feature_range (tuple): The range of transformed data used during normalization.

    Returns:
        np.ndarray: Denormalized data.
    """

    min_val = norm_params["min"]
    min_val = np.array(min_val).astype(np.float32)
    max_val = norm_params["max"]
    max_val = np.array(max_val).astype(np.float32)
    feature_min = norm_params["feature_min"]
    feature_min = np.array(feature_min).astype(np.float32)
    feaure_max = norm_params["feature_max"]
    feaure_max = np.array(feaure_max).astype(np.float32)
    scale = feaure_max - feature_min
    denormalized_data = ((normalized_data - feature_min) / scale) * (
        max_val - min_val
    ) + min_val

    return denormalized_data


def denormalize_spectra(normalized_data: np.ndarray, norm_params: dict) -> np.ndarray:
    max_abs_val = norm_params["max_abs_val"]
    return normalized_data * max_abs_val


def predict_and_denormalize(
    model: torch.nn.Module,
    spectrum: np.ndarray,
    norm_params: dict,
    device: str | None = None,
):
    """Predict and denormalize the output for a single unknown spectrum.

    Args:
        model (torch.nn.Module): The trained model.
        spectrum (np.ndarray): The unknown spectrum of shape (100,).
        norm_params (dict): Dictionary containing the min and max values for normalization.
        device (str): The device to run the model on ('cuda', 'mps', or 'cpu').


    Returns:
        np.ndarray: Denormalized prediction.
    """

    device = model.device
    # Convert the spectrum to a tensor and add batch dimension
    spectrum_tensor = (
        torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0).to(device)
    )

    # Put the model in evaluation mode
    model.eval()

    # Make the prediction
    with torch.no_grad():
        prediction = model(spectrum_tensor)

    # Convert prediction to numpy array and remove batch dimension
    prediction = prediction.cpu().numpy().squeeze(0)

    return denormalize_data(prediction, norm_params)


def predict_with_uncertainty(model: torch.nn.Module, input_data: np.ndarray, norm_params: dict, n_samples: int=10):
    device = model.device
    input_data = input_data.to(device)

    # Perform multiple forward passes
    predictions = torch.zeros((n_samples, 4)).to(device)

    with torch.no_grad():
        for i in range(n_samples):
            predictions[i] = model(input_data)

    # Calculate mean and standard deviation across the samples
    denormalized_predictions = np.zeros_like(predictions.cpu().numpy())
    for i, pred in enumerate(predictions):
        denormalized_predictions[i] = denormalize_data(pred.cpu().numpy(), norm_params)

    mean_predictions = np.mean(denormalized_predictions, axis=0)
    uncertainty = np.std(denormalized_predictions, axis=0)

    return mean_predictions, uncertainty

def denormalize_std(std_normalized: np.ndarray, norm_params: dict) -> np.ndarray:
    """Denormalize the standard deviation from the normalized scale to the original scale.

    Args:
        std_normalized (np.ndarray): The standard deviation in the normalized scale.
        norm_params (dict): Dictionary containing the min and max values for each feature.

    Returns:
        np.ndarray: Denormalized standard deviation.
    """
    # Extract normalization parameters
    min_val = np.array(norm_params["min"]).astype(np.float32)
    max_val = np.array(norm_params["max"]).astype(np.float32)
    feature_min = np.array(norm_params["feature_min"]).astype(np.float32)
    feature_max = np.array(norm_params["feature_max"]).astype(np.float32)
    
    # Compute scaling factor 'a'
    data_range = max_val - min_val
    feature_range = feature_max - feature_min
    a = feature_range / data_range
    
    # Denormalize the standard deviation
    std_original = std_normalized / a
    
    return std_original


def predict_with_uncertainty_hetMLP(model: torch.nn.Module, input_data: np.ndarray, norm_params: dict):
    device = next(model.parameters()).device
    model.eval()
    
    # Ensure input_data is a tensor and move to the appropriate device
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data, dtype=torch.float32)
    input_data = input_data.to(device)
    
    with torch.no_grad():
        # Get mean and log variance from the model
        mean, log_var = model(input_data)
        std = torch.exp(0.5 * log_var)  # Convert log variance to standard deviation

        # Move tensors to CPU and convert to numpy arrays
        mean = mean.cpu().numpy()
        std = std.cpu().numpy()

        # Denormalize mean and standard deviation
        mean_predictions = denormalize_data(mean, norm_params)
        std_predictions = denormalize_std(std, norm_params)

    return mean_predictions, std_predictions

# TODO: refactor
def plot_MSE_error_in_krange(
    k_grid,
    k_range,
    interpolated_exp,
    interpolated_NN,
    interpolated_artemis,
    plot_label="plot_label",
):
    """Plot the interpolated data in the specified k-range and calculate the MSE error.

    Args:
        k_grid (np.ndarray): The k-grid for the interpolated data.
        k_range (tuple): The k-range to plot the data.
        interpolated_exp (np.ndarray): The interpolated experimental data.
        interpolated_NN (np.ndarray): The interpolated data predicted by the neural network.
        interpolated_artemis (np.ndarray): The interpolated data predicted by Artemis.
        plot_label (str): The label for the plot.
    """

    kmin = k_range[0]
    kmax = k_range[1]

    # MSE error between predicted with nano between k=2 and 10
    interpolated_NN_in_range = [
        i[1] for i in zip(k_grid, interpolated_NN) if kmin <= i[0] <= kmax
    ]
    interpolated_NN_in_range = np.array(interpolated_NN_in_range)

    interpolated_exp_in_range = [
        i[1] for i in zip(k_grid, interpolated_exp) if kmin <= i[0] <= kmax
    ]
    interpolated_exp_in_range = np.array(interpolated_exp_in_range)

    e1 = np.mean((interpolated_NN_in_range - interpolated_exp_in_range) ** 2)

    # MSE error between predicted and artemis with nano
    interpolated_artemis_in_range = [
        i[1] for i in zip(k_grid, interpolated_artemis) if kmin <= i[0] <= kmax
    ]
    interpolated_artemis_in_range = np.array(interpolated_artemis_in_range)

    e2 = np.mean((interpolated_artemis_in_range - interpolated_exp_in_range) ** 2)

    k_grid_truncated = [i for i in k_grid if kmin <= i <= kmax]

    plt.figure(figsize=(8, 6))

    # Plotting the data with enhanced line styles and colors
    plt.plot(
        k_grid_truncated,
        interpolated_artemis_in_range,
        linestyle="--",
        color="blue",
        linewidth=2,
        label=f"Artemis MSE = {e2:.4f}",
    )
    plt.plot(
        k_grid_truncated,
        interpolated_NN_in_range,
        linestyle="-.",
        color="green",
        linewidth=2,
        label=f"NN MSE = {e1:.4f}",
    )
    plt.plot(
        k_grid_truncated,
        interpolated_exp_in_range,
        linestyle="-",
        color="black",
        linewidth=2,
        label="Experimental",
    )

    # Adding axis labels with appropriate font sizes
    plt.xlabel(r"$ k \ \rm (\AA^{-1})$", fontsize=14)
    plt.ylabel(r"$ k^2\chi(q)$", fontsize=14)

    # Adding grid lines for better readability
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Adding a legend with a border and appropriate font size
    plt.legend(fontsize=12, loc="best", frameon=True)
    plt.title(plot_label, fontsize=16)

    # Setting the limits of the plot if needed (optional)
    # plt.xlim(min(k_grid_truncated), max(k_grid_truncated))
    # plt.ylim(min_y_value, max_y_value)

    # Setting the title of the plot (optional)
    # plt.title('Comparison of Interpolated Data', fontsize=16)

    # Display the plot
    plt.show()

def create_random_series(spectra, parameters, N_series, min_len, max_len, seed=42):
    """
    Create N_series sequences of random lengths between min_len and max_len from the given data.

    Args:
        spectra (np.ndarray): [N, 401] array of individual spectra.
        parameters (np.ndarray): [N, 4] array of parameters corresponding to each spectrum.
        N_series (int): Number of sequences to create.
        min_len (int): Minimum length of a sequence.
        max_len (int): Maximum length of a sequence.
        seed (int): Random seed for reproducibility.

    Returns:
        spectra_series (list of np.ndarray): Each element is [length, 401]
        parameters_series (list of np.ndarray): Each element is [length, 4]
        lengths (list): Lengths of each generated sequence (for debugging).
    """
    rng = np.random.default_rng(seed)
    N = spectra.shape[0]
    if min_len < 1 or max_len < min_len:
        raise ValueError("Invalid min_len/max_len values.")
    
    spectra_series = []
    parameters_series = []
    lengths = []

    for _ in range(N_series):
        length = rng.integers(min_len, max_len + 1)
        lengths.append(length)

        # Randomly select 'length' indices (with replacement)
        indices = rng.choice(N, size=length, replace=True)
        
        seq_spectra = spectra[indices]      # [length, 401]
        seq_params = parameters[indices]    # [length, 4]
        
        spectra_series.append(seq_spectra)
        parameters_series.append(seq_params)

    return spectra_series, parameters_series, lengths
