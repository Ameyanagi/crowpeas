import torch
from torch import nn
from larch.xafs import ftwindow
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics import MeanMetric
from torchmetrics.aggregation import MinMetric
from typing import Sequence
from .activation import ACTIVATIONS

class CNN(pl.LightningModule):
    def __init__(
        self,
        hidden_layers: Sequence[int],
        num_filters: Sequence[int] = [32, 64],
        kernel_sizes: Sequence[int] = [3, 3],
        dropout: float = 0.2,
        output_size=4,
        k_min=2.5,
        k_max=12.5,
        r_min=1.7,
        r_max=3.2,
        rmax_out=6,
        window="kaiser",
        dx=1,
        input_form="q",
        activation="relu",
        learning_rate=1e-3,
    ):
        super().__init__()
        # Keep all the initialization from MLP
        self.save_hyperparameters()
        self.input_form = input_form
        self.learning_rate = learning_rate
        self.dropout = nn.Dropout(dropout)
        
        # Initialize metrics
        self.loss = torch.nn.MSELoss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_best = MinMetric()

        # Keep all FFT parameters and buffers
        self.nfft = 2048
        self.kstep = 0.05
        self.rmax_out = rmax_out
        self.k_min = k_min
        self.k_max = k_max
        self.r_min = r_min
        self.r_max = r_max
        
        # Register all the same buffers as in MLP
        # [Previous buffer registrations remain the same]

        self.register_buffer("k_grid", torch.arange(0, 20.05, 0.05))
        self.register_buffer("nfft_t", torch.tensor(self.nfft, dtype=torch.int32))
        self.register_buffer(
            "window",
            torch.tensor(
                ftwindow(
                    self.k_grid.cpu().numpy(),
                    xmin=self.k_min,
                    xmax=self.k_max,
                    window=window,
                    dx=dx,
                ),
                dtype=torch.float32,
            ),
        )

        # Constants
        self.register_buffer(
            "fft_c1", torch.tensor(self.kstep * np.pi, dtype=torch.float32)
        )
        self.register_buffer(
            "fft_c2", torch.tensor(np.pi * np.sqrt(np.pi), dtype=torch.float32)
        )
        self.scale = 0.5
        self.rw = 0

        self.register_buffer(
            "ffti_c1",
            torch.tensor(
                self.scale * (4 * np.sqrt(np.pi) / self.kstep), dtype=torch.float32
            ),
        )

        # r-space grid
        self.register_buffer(
            "rstep", torch.tensor(np.pi / (self.kstep * self.nfft), dtype=torch.float32)
        )
        self.register_buffer(
            "irmax",
            torch.tensor(
                int(min(self.nfft / 2, 1.01 + self.rmax_out / self.rstep)),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "r_space",
            torch.tensor(self.rstep * np.arange(self.irmax), dtype=torch.float32),
        )
        self.register_buffer(
            "r_",
            self.rstep * torch.arange(self.nfft),
        )
        self.register_buffer(
            "window_r",
            torch.tensor(
                ftwindow(
                    self.r_.cpu().numpy(),
                    xmin=self.r_min,
                    xmax=self.r_max,
                    window=window,
                    dx=dx,
                ),
                dtype=torch.float32,
            ),
        )

        # q-space grid
        self.register_buffer(
            "q",
            torch.tensor(
                np.linspace(0, 30, int(1.05 + 30 / self.kstep)), dtype=torch.float32
            ),
        )
        self.nkpts = len(self.q)
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        in_channels = 1  # Start with 1 channel for signal data
        
        for filters, kernel_size in zip(num_filters, kernel_sizes):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, filters, kernel_size, padding='same'),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
            )
            in_channels = filters

        # Calculate input size for fully connected layers
        if self.input_form == "r":
            cnn_input_size = 2 * self.nfft
        else:
            cnn_input_size = 2 * self.nkpts
            
        # Reshape factor for CNN input
        self.height = int(np.sqrt(cnn_input_size))
        self.width = self.height
        
        # Calculate CNN output size
        x = torch.zeros(1, 1, self.height, self.width)
        for conv in self.conv_layers:
            x = conv(x)
        flattened_size = int(np.prod(x.shape[1:]))

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        prev_size = flattened_size
        
        for hidden_size in hidden_layers:
            self.fc_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        self.output_layer = nn.Linear(prev_size, output_size)
        
        if activation.lower() in ACTIVATIONS.keys():
            self.activation = ACTIVATIONS[activation.lower()]

    def forward(self, x):
        batch_size = x.shape[0]

        # Keep all the FFT preprocessing from original MLP
        padded_x = torch.zeros(batch_size, self.nfft, dtype=torch.complex64).to(x)
        padded_x[:, : x.shape[1]] = (
            x * self.window.unsqueeze(0) if self.window is not None else x
        )

        # Perform FFT/IFFT as before
        fft = torch.fft.fft(padded_x, dim=1)
        fft = self.fft_c1 * fft[:, : self.nfft_t // 2]
        fft = fft / self.fft_c2

        padded_fft = torch.zeros(
            batch_size, self.nfft, dtype=torch.complex64, device=self.device
        )
        padded_fft[:, : fft.shape[1]] = fft

        r_factor = self.r_**self.rw
        window_r_factor = self.window_r if self.window_r is not None else 1

        # Process based on input form
        if self.input_form == "r":
            r_space_input = padded_fft * window_r_factor * r_factor.unsqueeze(0)
            x = torch.cat((r_space_input.real, r_space_input.imag), dim=1)
        else:
            ffti = self.ffti_c1 * torch.fft.ifft(
                padded_fft * window_r_factor * r_factor.unsqueeze(0), dim=1, norm=None
            )
            ffti = ffti[:, : torch.floor(self.nfft_t / 2).long()]
            ffti = ffti[:, : self.nkpts]
            x = torch.cat((ffti.real, ffti.imag), dim=1)

        # Reshape for CNN
        x = x.view(batch_size, 1, self.height, self.width)

        # Apply CNN layers
        for conv in self.conv_layers:
            x = conv(x)
            x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Apply FC layers with dropout
        for fc in self.fc_layers[:-1]:  # Apply to all but last layer
            x = fc(x)
            x = self.activation()(x)
            x = self.dropout(x)  # Add dropout between FC layers
        
        # Last FC layer without dropout
        if self.fc_layers:
            x = self.activation()(self.fc_layers[-1](x))

        return self.output_layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.train_loss(loss)
        self.log(
            "train/loss",  # Changed from "train_loss" to "train/loss"
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.val_loss(loss)
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def on_validation_epoch_end(self):
        val_loss = self.val_loss.compute()
        self.val_loss_best.update(val_loss)
        self.log(
            "val/loss_best",
            self.val_loss_best.compute(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.test_loss(loss)
        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # Update ModelCheckpoint to monitor val/loss instead
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",  # Changed from val/loss_best
        dirpath="./checkpoint/",
        filename="nnxanes-{epoch:02d}",
        save_top_k=1,
        mode="min",
    )