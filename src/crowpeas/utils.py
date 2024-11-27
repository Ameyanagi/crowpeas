import numpy as np
import torch
import matplotlib.pyplot as plt


def normalize_data(
    data: np.ndarray, feature_range: tuple = (0, 1)
) -> tuple[np.ndarray, dict]:
    """Normalize the data to the specified feature range.

    Args:
        data (np.ndarray): The data to be normalized.
        feature_range (tuple): The desired range of transformed data.

    Returns:
        np.ndarray: Normalized data.
        dict: Dictionary containing the min and max values for each feature.
    """
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    feature_min = np.min(feature_range)
    feature_max = np.max(feature_range)
    scale = feature_max - feature_min
    normalized_data = feature_min + (data - min_val) / (max_val - min_val) * scale

    norm_params = {
        "min": min_val,
        "max": max_val,
        "feature_min": feature_min,
        "feature_max": feature_max,
    }

    return normalized_data, norm_params


def normalize_spectra(spectra: np.ndarray) -> tuple[np.ndarray, dict]:
    min_val = np.min(spectra)
    max_val = np.max(spectra)

    max_abs_val = np.max([np.abs(min_val), np.abs(max_val)])
    normalized_spectra = spectra / max_abs_val

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


def predict_with_uncertainty(model: torch.nn.Module, input_data: np.ndarray, norm_params: dict, n_samples: int=100):
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
