from typing import Sequence

import lightning.pytorch as pl
import numpy as np
import torch
from larch.xafs import ftwindow
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torchmetrics import MeanMetric
from torchmetrics.aggregation import MinMetric

from .activation import ACTIVATIONS


class CustomLoss(nn.Module):
    def __init__(self, norm_params, alpha_list=None):
        super(CustomLoss, self).__init__()
        self.save_hyperparameters()

        self.mse_loss = nn.MSELoss()
        self.norm_params = norm_params
        if alpha_list is None:
            alpha_list = [1, 1, 1, 1]
        self.alpha_list = alpha_list

    def forward(self, predicted_parameters, target_parameters):
        w1, w2, w3, w4 = self.alpha_list

        weighted_mse_loss = (
            w1 * self.mse_loss(predicted_parameters[:, 0], target_parameters[:, 0])
            + w2 * self.mse_loss(predicted_parameters[:, 1], target_parameters[:, 1])
            + w3 * self.mse_loss(predicted_parameters[:, 2], target_parameters[:, 2])
            + w4 * self.mse_loss(predicted_parameters[:, 3], target_parameters[:, 3])
        )

        return weighted_mse_loss


class CNN(pl.LightningModule):
    def __init__(
        self,
        hidden_layers: Sequence[int],
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
        cnn_layers=2,
        cnn_kernel_size=20,
        cnn_stride=1,
        cnn_padding="same",
        cnn_dilation=1,
        cnn_activation="relu",
        cnn_pooling="max",
        cnn_outchannels=32,
        cnn_pooling_kernel_size=3,
        cnn_pooling_stride=1,
        cnn_pooling_padding=0,
        cnn_pooling_dilation=1,
        cnn_dropout=0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_form = input_form
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate

        self.cnn_layers = cnn_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_stride = cnn_stride
        self.cnn_padding = cnn_padding
        self.cnn_dilation = cnn_dilation
        self.cnn_activation = cnn_activation
        self.cnn_pooling = cnn_pooling
        self.cnn_pooling_size = cnn_pooling_kernel_size
        self.cnn_pooling_stride = cnn_pooling_stride
        self.cnn_pooling_padding = cnn_pooling_padding
        self.cnn_pooling_dilation = cnn_pooling_dilation
        self.cnn_dropout = cnn_dropout

        if isinstance(cnn_outchannels, int):
            outchannels = [cnn_outchannels] * cnn_layers
        elif isinstance(cnn_outchannels, list):
            if len(cnn_outchannels) == cnn_layers:
                outchannels = cnn_outchannels
            else:
                raise ValueError(
                    "cnn_outchannels must have the same length as cnn_layers"
                )
        else:
            raise ValueError("cnn_outchannels must be an integer or a list of integers")

        # loss function
        self.loss = torch.nn.MSELoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

        # FFT parameters
        self.nfft = 2048
        self.kstep = 0.05
        self.rmax_out = rmax_out
        self.k_min = k_min
        self.k_max = k_max
        self.r_min = r_min
        self.r_max = r_max

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

        if input_form == "r":
            current_input_node = self.nfft
            current_input_channels = 2
        else:
            current_input_node = self.nkpts
            current_input_channels = 2

        self.cnn_modules = []

        for i in range(self.cnn_layers):
            # Add CNN layers
            #
            setattr(
                self,
                f"conv{i}",
                nn.Conv1d(
                    in_channels=current_input_channels,
                    out_channels=outchannels[i],
                    kernel_size=self.cnn_kernel_size,
                    stride=self.cnn_stride,
                    padding=self.cnn_padding,
                    dilation=self.cnn_dilation,
                ),
            )

            self.cnn_modules.append(getattr(self, f"conv{i}"))

            if self.cnn_padding == "same":
                current_input_node = current_input_node
            elif isinstance(self.cnn_padding, int):
                current_input_node = (
                    current_input_node
                    + 2 * self.cnn_padding
                    - self.cnn_dilation * (self.cnn_kernel_size - 1)
                    - 1
                ) // self.cnn_stride + 1
            else:
                raise ValueError("cnn_padding must be an integer or 'same'")

            print(current_input_node)

            if self.cnn_activation.lower() in ACTIVATIONS.keys():
                cnn_activation = ACTIVATIONS[self.cnn_activation.lower()]
                self.cnn_modules.append(cnn_activation())

            if self.cnn_pooling.lower() == "max":
                self.cnn_modules.append(
                    nn.MaxPool1d(
                        kernel_size=self.cnn_pooling_size,
                        stride=self.cnn_pooling_stride,
                        padding=self.cnn_pooling_padding,
                        dilation=self.cnn_pooling_dilation,
                    )
                )
                current_input_node = (
                    current_input_node
                    + 2 * self.cnn_pooling_padding
                    - self.cnn_pooling_dilation * (self.cnn_pooling_size - 1)
                    - 1
                ) // self.cnn_pooling_stride + 1

            elif self.cnn_pooling.lower() == "avg":
                self.cnn_modules.append(
                    nn.AvgPool1d(
                        kernel_size=self.cnn_pooling_size,
                        stride=self.cnn_pooling_stride,
                        padding=self.cnn_pooling_padding,
                    )
                )
                current_input_node = (
                    current_input_node
                    + 2 * self.cnn_pooling_padding
                    - (self.cnn_pooling_size - 1)
                    - 1
                ) // self.cnn_pooling_stride + 1

            print(current_input_node)

            if self.cnn_dropout > 0:
                self.cnn_modules.append(nn.Dropout(self.cnn_dropout))

            current_input_channels = outchannels[i]

        input_layers = [current_input_node * current_input_channels]

        for layer in hidden_layers:
            input_layers.append(layer)

        output_layers = [layer for layer in hidden_layers]
        output_layers.append(output_size)

        if activation.lower() in ACTIVATIONS.keys():
            self.activation = ACTIVATIONS[activation.lower()]

        for i, (in_layer, out_layer) in enumerate(zip(input_layers, output_layers)):
            setattr(
                self,
                f"fc{i}",
                nn.Linear(in_features=in_layer, out_features=out_layer),
            )

            if i < len(hidden_layers):
                setattr(self, f"activation_{i}", self.activation())

    def forward(self, x):
        batch_size = x.shape[0]

        # Pad input to nfft
        padded_x = torch.zeros(batch_size, self.nfft, dtype=torch.complex64).to(x)
        padded_x[:, : x.shape[1]] = (
            x * self.window.unsqueeze(0) if self.window is not None else x
        )

        # Perform FFT
        fft = torch.fft.fft(padded_x, dim=1)
        fft = self.fft_c1 * fft[:, : self.nfft_t // 2]
        fft = fft / self.fft_c2

        # Prepare for IFFT
        padded_fft = torch.zeros(
            batch_size, self.nfft, dtype=torch.complex64, device=self.device
        )
        padded_fft[:, : fft.shape[1]] = fft

        # Perform IFFT
        r_factor = self.r_**self.rw
        window_r_factor = self.window_r if self.window_r is not None else 1

        if self.input_form == "r":
            # try r space
            r_space_input = padded_fft * window_r_factor * r_factor.unsqueeze(0)

            r_space_input_con = torch.stack(
                (r_space_input.real, r_space_input.imag),
                dim=1,
            )

            for layer in self.cnn_modules:
                r_space_input_con = layer(r_space_input_con)

            r_space_input_con = r_space_input_con.view(r_space_input_con.size(0), -1)
            out = self.fc0(r_space_input_con)
        else:
            ffti = self.ffti_c1 * torch.fft.ifft(
                padded_fft * window_r_factor * r_factor.unsqueeze(0), dim=1, norm=None
            )
            ffti = ffti[:, : torch.floor(self.nfft_t / 2).long()]
            ffti = ffti[:, : self.nkpts]
            ffti_conc = torch.stack((ffti.real, ffti.imag), dim=1)

            for layer in self.cnn_modules:
                ffti_conc = layer(ffti_conc)

            ffti_conc = ffti_conc.view(ffti_conc.size(0), -1)
            out = self.fc0(ffti_conc)
        # Pass through Bayesian neural network layers

        out = self.activation_0(out)

        for i in range(1, len(self.hidden_layers)):
            out = getattr(self, f"fc{i}")(out)
            out = getattr(self, f"activation_{i}")(out)

        out = getattr(self, f"fc{len(self.hidden_layers) }")(out)

        return out

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return loss, y_hat, y

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, y_hat, y = self.model_step(batch)

        self.train_loss(loss)

        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, y_hat, y = self.model_step(batch)

        self.val_loss(loss)
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute()

        self.val_loss_best.update(loss)

        self.log(
            "val/loss_best",
            self.val_loss_best.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, y_hat, y = self.model_step(batch)

        self.test_loss(loss)
        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def on_test_epoch_end(self) -> None:
        pass

    def configure_optimizers(self) -> dict[str, any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


checkpoint_callback = ModelCheckpoint(
    monitor="val/loss_best",
    dirpath="./checkpoint/",
    filename="nnxanes-{epoch:02d}",
    save_top_k=1,
    mode="min",
)
