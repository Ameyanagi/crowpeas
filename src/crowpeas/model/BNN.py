import torch
from torch import nn
from larch.xafs import ftwindow
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics import MeanMetric, MinMetric
import torchbnn as bnn

from typing import Sequence
from .activation import ACTIVATIONS


class CustomLoss(nn.Module):
    def __init__(self, norm_params, alpha_list=None):
        super(CustomLoss, self).__init__()
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


class BNN(pl.LightningModule):
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
    ):
        super().__init__()
        self.input_form = input_form
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate

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

        # Bayesian Neural network layers
        if self.input_form == "r":
            input_size = 2 * self.nfft
        else:
            input_size = 2 * self.nkpts

        input_layers = [input_size]

        for layer in hidden_layers:
            input_layers.append(layer)

        output_layers = [layer for layer in hidden_layers[::-1]]
        output_layers.append(output_size)

        if activation.lower() in ACTIVATIONS.keys():
            self.activation = ACTIVATIONS[activation.lower()]

        for i, (in_layer, out_layer) in enumerate(zip(input_layers, output_layers)):
            setattr(
                self,
                f"fc{i}",
                bnn.BayesLinear(
                    prior_mu=0,
                    prior_sigma=0.1,
                    in_features=in_layer,
                    out_features=out_layer,
                ),
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

            r_space_input_con = torch.cat(
                (r_space_input.real, r_space_input.imag), dim=1
            )
            out = self.fc0(r_space_input_con)
        else:
            ffti = self.ffti_c1 * torch.fft.ifft(
                padded_fft * window_r_factor * r_factor.unsqueeze(0), dim=1, norm=None
            )
            ffti = ffti[:, : torch.floor(self.nfft_t / 2).long()]
            ffti = ffti[:, : self.nkpts]
            ffti_conc = torch.cat((ffti.real, ffti.imag), dim=1)
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
        self.val_loss_best(loss)

        self.log(
            "val/loss_best",
            self.val_loss_best,
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
