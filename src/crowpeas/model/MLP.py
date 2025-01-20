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


class MLP(pl.LightningModule):
    def __init__(
        self,
        hidden_layers: Sequence[int],
        output_size=4,
        k_min=2.5,
        k_max=12.5,
        r_min=1.7,
        r_max=3.2,
        rmax_out=6,
        #q_min=2, # adjust
        #q_max=13, # adjust
        window="kaiser",
        dx=1,
        input_form="r",
        activation="relu",
        learning_rate=1e-3,
        #dropout_rates: Sequence[float] = None,  # New parameter for dropout rates
        dropout_rates: Sequence[float] = None,
        weight_decay=0.0,  # Added weight decay parameter 
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_form = input_form
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize dropout rates (None or 0 means no dropout)
        if dropout_rates is None:
            dropout_rates = [0] * (len(hidden_layers) + 1)  # +1 for input layer
        elif len(dropout_rates) < len(hidden_layers) + 1:
            # Pad with zeros if not enough rates provided
            dropout_rates = list(dropout_rates) + [0] * (len(hidden_layers) + 1 - len(dropout_rates))

        self.dropout_rates = dropout_rates

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
        self.q_grid = np.linspace(0, 30, int(1.05 + 30 / self.kstep))
        self.q_min = k_min
        self.q_max = k_max

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

        self.register_buffer(
            "qwindow",
            torch.tensor(
                ftwindow(
                    self.q_grid,
                    xmin=self.q_min,
                    xmax=self.q_max,
                    window=window,
                    dx=dx,
                ),
                dtype=torch.float32,
            ),
        )     

        # Find indices where window is effectively non-zero
        self.q_nonzero_indices = torch.where(self.qwindow > 1e-10)[0]
        self.q_start_idx = self.q_nonzero_indices[0]
        self.q_end_idx = self.q_nonzero_indices[-1] + 1
        self.q_effective_size = self.q_end_idx - self.q_start_idx

        # Adjust input size based on window
        if self.input_form == "r":
            input_size = 2 * self.nfft
        elif self.input_form == "q":
            input_size = 2 * self.q_effective_size
        else:
            raise ValueError(f"Invalid input form: {self.input_form}")
            
        input_size = 2 * self.nkpts # hardcode for test
        
        input_layers = [input_size]
        for layer in hidden_layers:
            input_layers.append(layer)

        output_layers = [layer for layer in hidden_layers]
        output_layers.append(output_size)

        if activation.lower() in ACTIVATIONS.keys():
            self.activation = ACTIVATIONS[activation.lower()]

        # Create linear layers and dropout layers
        for i, (in_layer, out_layer) in enumerate(zip(input_layers, output_layers)):
            # Linear layer
            setattr(
                self,
                f"fc{i}",
                nn.Linear(in_features=in_layer, out_features=out_layer),
            )
            
            # Dropout layer (if rate > 0)
            if self.dropout_rates[i] > 0:
                setattr(
                    self,
                    f"dropout_{i}",
                    nn.Dropout(p=self.dropout_rates[i])
                )

            # Activation layer (except for output layer)
            if i < len(hidden_layers):
                setattr(self, f"activation_{i}", self.activation())

    def forward(self, x):
        batch_size = x.shape[0]

        # Pad input to nfft and ensure on device
        padded_x = torch.zeros(batch_size, self.nfft, dtype=torch.complex64, device=self.device)
        if self.window is not None:
            window = self.window.to(self.device)
            padded_x[:, : x.shape[1]] = x.to(self.device) * window.unsqueeze(0)
        else:
            padded_x[:, : x.shape[1]] = x.to(self.device)

        # Perform FFT
        fft = torch.fft.fft(padded_x, dim=1)
        fft = self.fft_c1.to(self.device) * fft[:, : self.nfft_t // 2]
        fft = fft / self.fft_c2.to(self.device)

        # Prepare for IFFT
        padded_fft = torch.zeros(
            batch_size, self.nfft, dtype=torch.complex64, device=self.device
        )
        padded_fft[:, : fft.shape[1]] = fft

        # Perform IFFT
        r_factor = (self.r_.to(self.device))**self.rw
        window_r_factor = self.window_r.to(self.device) if self.window_r is not None else 1

        if self.input_form == "r":
            r_space_input = padded_fft * window_r_factor * r_factor.unsqueeze(0)
            r_space_input_con = torch.cat(
                (r_space_input.real, r_space_input.imag), dim=1
            )
            out = self.fc0(r_space_input_con)
        else:
            ffti = self.ffti_c1.to(self.device) * torch.fft.ifft(
                padded_fft * window_r_factor * r_factor.unsqueeze(0), dim=1, norm=None
            )
            ffti = ffti[:, : torch.floor(self.nfft_t / 2).long()]
            ffti = ffti[:, : self.nkpts]

            self.qwindow = None
            if self.qwindow is not None:
                qwindow = self.qwindow.to(self.device)
                ffti = ffti[:, self.q_start_idx:self.q_end_idx]
                qwindow = qwindow[self.q_start_idx:self.q_end_idx]
                ffti = ffti * qwindow.unsqueeze(0)                
            
            ffti_conc = torch.cat((ffti.real, ffti.imag), dim=1)
            out = self.fc0(ffti_conc)

        # Apply first dropout if specified
        if hasattr(self, 'dropout_0'):
            out = getattr(self, 'dropout_0')(out)
        
        out = self.activation_0(out)

        # Process hidden layers with dropout
        for i in range(1, len(self.hidden_layers)):
            out = getattr(self, f"fc{i}")(out)
            
            # Apply dropout if specified for this layer
            if hasattr(self, f'dropout_{i}'):
                out = getattr(self, f'dropout_{i}')(out)
                
            out = getattr(self, f"activation_{i}")(out)

        # Final layer
        out = getattr(self, f"fc{len(self.hidden_layers)}")(out)
        
        # Apply final dropout if specified
        if hasattr(self, f'dropout_{len(self.hidden_layers)}'):
            out = getattr(self, f'dropout_{len(self.hidden_layers)}')(out)

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
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay  # Apply weight decay in optimizer
        )
        return optimizer


checkpoint_callback = ModelCheckpoint(
    monitor="val/loss_best",
    dirpath="./checkpoint/",
    filename="nnxanes-{epoch:02d}",
    save_top_k=1,
    mode="min",
)
