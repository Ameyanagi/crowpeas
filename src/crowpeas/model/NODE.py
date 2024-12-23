import torch
from torch import nn
import lightning.pytorch as pl
from torchmetrics import MeanMetric, MinMetric
from larch.xafs import ftwindow
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint

import torch._dynamo
torch._dynamo.config.suppress_errors = True


class NODE(pl.LightningModule):
    def __init__(
        self,
        hidden_layers,
        output_size=4,
        k_min=2.5,
        k_max=12.5,
        r_min=1.7,
        r_max=3.2,
        rmax_out=6,
        window="kaiser",
        dx=1,
        input_form="r",
        activation="relu",
        learning_rate=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # FFT parameters
        self.nfft = 2048
        self.kstep = 0.05
        self.rmax_out = rmax_out
        self.k_min = k_min
        self.k_max = k_max
        self.r_min = r_min
        self.r_max = r_max

        # Create k_grid and window as buffers to avoid them being considered as parameters
        self.register_buffer("k_grid", torch.arange(0, 20.05, 0.05))
        self.register_buffer("window", torch.tensor(
            ftwindow(
                self.k_grid.cpu().numpy(),
                xmin=self.k_min,
                xmax=self.k_max,
                window=window,
                dx=dx,
            ),
            dtype=torch.float32,
        ))

        self.fft_c1 = self.kstep * np.pi
        self.fft_c2 = np.pi * np.sqrt(np.pi)
        self.rstep = np.pi / (self.kstep * self.nfft)

        # r-space grid (unused in this standard NN)
        self.r_space = self.rstep * torch.arange(int(min(self.nfft / 2, 1.01 + rmax_out / self.rstep)))

        # FFT processing layers
        # Input_dim = 2 * nfft = 4096 (real and imaginary parts)
        self.input_dim = 2 * self.nfft  # 4096

        # Define a standard feedforward neural network
        layers = []
        last_dim = self.input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU() if activation.lower() == "relu" else nn.Tanh())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_size))
        self.net = nn.Sequential(*layers)

        # Loss function
        self.loss = nn.MSELoss(reduction='none')  # We'll handle reduction manually

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_best = MinMetric()

        self.learning_rate = learning_rate

    def fft_processing(self, x):
        """
        Perform FFT processing on input data x: [batch_size, k_len=401]
        """
        batch_size = x.shape[0]

        # Pad input to nfft
        padded_x = torch.zeros(batch_size, self.nfft, dtype=torch.complex64, device=x.device)
        padded_x[:, :x.shape[1]] = x * self.window.unsqueeze(0)

        # Perform FFT
        fft = torch.fft.fft(padded_x, dim=1)
        # Keep full range to maintain correct dimension: [batch_size, 2048]
        fft = self.fft_c1 * fft[:, :self.nfft]  
        fft = fft / self.fft_c2

        # Concatenate real and imaginary parts: final shape [batch_size, 4096]
        fft_concatenated = torch.cat((fft.real, fft.imag), dim=1)
        return fft_concatenated

    def forward(self, x):
        """
        Forward pass through the StandardNN.
        
        Args:
            x (torch.Tensor): Spectra input tensor with shape [batch_size, max_len=20, k_len=401]
        
        Returns:
            torch.Tensor: Predicted parameters with shape [batch_size, max_len=20, output_size=4]
        """
        batch_size, max_len, k_len = x.shape

        # Reshape x for batch processing: [batch_size * max_len, k_len]
        x_reshaped = x.view(-1, k_len)  # Shape: [batch_size * max_len, 401]

        # FFT processing: [batch_size * max_len, 4096]
        y0 = self.fft_processing(x_reshaped)  # Shape: [batch_size * max_len, 4096]

        # Pass through the network: [batch_size * max_len, output_size]
        preds = self.net(y0)  # Shape: [batch_size * max_len, output_size]

        # Reshape back to [batch_size, max_len, output_size]
        preds = preds.view(batch_size, max_len, -1)  # Shape: [batch_size, max_len, output_size]

        return preds

    def compute_masked_loss(self, preds, targets, masks):
        """
        Computes the MSE loss only on valid (non-padded) entries.
        
        Args:
            preds (torch.Tensor): Predicted parameters, shape [batch_size, max_len, param_dim=4]
            targets (torch.Tensor): True parameters, shape [batch_size, max_len, param_dim=4]
            masks (torch.Tensor): Boolean mask, shape [batch_size, max_len]
        
        Returns:
            torch.Tensor: Scalar loss
        """
        # Compute element-wise loss
        loss = self.loss(preds, targets)  # Shape: [batch_size, max_len, param_dim=4]

        # Expand masks to match loss dimensions
        masks_expanded = masks.unsqueeze(-1)  # Shape: [batch_size, max_len, 1]

        # Apply mask
        loss = loss * masks_expanded  # Zero out losses where mask is False

        # Compute mean loss over valid entries
        total_loss = loss.sum()
        num_valid = masks_expanded.sum()

        # Avoid division by zero
        if num_valid > 0:
            mean_loss = total_loss / num_valid
        else:
            mean_loss = torch.tensor(0.0, device=preds.device)

        return mean_loss

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.
        
        Args:
            batch (tuple): Tuple containing (spectra, targets, masks)
            batch_idx (int): Index of the batch
        
        Returns:
            torch.Tensor: Scalar loss
        """
        spectra, targets, masks = batch  # spectra: [batch_size, max_len, 401]
                                         # targets: [batch_size, max_len, 4]
                                         # masks: [batch_size, max_len]

        # Forward pass
        preds = self.forward(spectra)  # Shape: [batch_size, max_len, 4]

        # Compute masked loss
        loss = self.compute_masked_loss(preds, targets, masks)

        # Update and log metrics
        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        spectra, targets, masks = batch  # Unpack the list
        #print(f"Spectra shape: {spectra.shape}, Type: {type(spectra)}")
        preds = self(spectra)
        loss = self.compute_masked_loss(preds, targets, masks)
        
        # Log the validation loss
        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss




    def test_step(self, batch, batch_idx):
        """
        Test step for a single batch.
        
        Args:
            batch (tuple): Tuple containing (spectra, targets, masks)
            batch_idx (int): Index of the batch
        """
        spectra, targets, masks = batch

        # Forward pass
        preds = self.forward(spectra)

        # Compute masked loss
        loss = self.compute_masked_loss(preds, targets, masks)

        # Update and log metrics
        self.test_loss(loss)
        self.log("test_loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        """
        Configures the optimizer.
        
        Returns:
            torch.optim.Optimizer: Optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to update and log the best validation loss.
        """
        val_loss = self.val_loss.compute()
        self.val_loss_best.update(val_loss)
        self.log("val/loss_best", self.val_loss_best.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

# Initialize the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val/loss_best",
    dirpath="/home/nick/Projects/crowpeas/tests/StandardNN/checkpoint/",
    filename="standardnn-{epoch:02d}",
    save_top_k=1,
    mode="min",
)
