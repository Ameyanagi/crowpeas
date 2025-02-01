from torchdiffeq import odeint
import torch
from torch import nn
import lightning.pytorch as pl
from torchmetrics import MeanMetric
from torchmetrics.aggregation import MinMetric
from larch.xafs import ftwindow
import numpy as np
import torch._dynamo
from lightning.pytorch.callbacks import ModelCheckpoint

class ODEFunc(nn.Module):
    """Defines the ODE function governing the hidden state dynamics."""
    def __init__(self, input_dim, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, t, state):
        return self.net(state)


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

        # r-space grid
        self.r_space = self.rstep * torch.arange(int(min(self.nfft / 2, 1.01 + rmax_out / self.rstep)))

        # ODE input_dim = 2 * self.nfft = 4096
        input_dim = 2 * self.nfft  # 4096
        hidden_dim = hidden_layers[0]

        self.ode_func = ODEFunc(input_dim=input_dim, hidden_dim=hidden_dim)
        # fc_out should map from 4096 to output_size
        self.fc_out = nn.Linear(4096, output_size)

        # Loss function
        self.loss = nn.MSELoss()

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_best = MinMetric()

        self.learning_rate = learning_rate

    def fft_processing(self, x):
        """
        Perform FFT processing on input data x: [batch_size, 401]
        """
        batch_size = x.shape[0]

        # Pad input to nfft
        padded_x = torch.zeros(batch_size, self.nfft, dtype=torch.complex64, device=x.device)
        padded_x[:, : x.shape[1]] = x * self.window.unsqueeze(0)

        # Perform FFT
        fft = torch.fft.fft(padded_x, dim=1)
        # Keep full range to maintain correct dimension: [batch_size, 2048]
        fft = self.fft_c1 * fft[:, : self.nfft]  
        fft = fft / self.fft_c2

        # Concatenate real and imaginary parts: final shape [batch_size, 4096]
        fft_concatenated = torch.cat((fft.real, fft.imag), dim=1)
        return fft_concatenated

    def forward(self, x):
        # Assume we can generate t_span from x.shape or have it as a class attribute
        t_span = torch.linspace(0, 1, x.shape[1]).to(x.device)
        return self.ode_forward(x, t_span)

    @torch._dynamo.disable
    def ode_forward(self, spectra, t_span):
        """
        spectra: [1, num_spectra, 401]
        t_span: [num_spectra]
        """

        batch_size, num_spectra, feature_dim = spectra.shape
        # batch_size is 1 here, since we process one sequence at a time

        # Extract the initial spectrum (time step 0)
        initial_spectrum = spectra[:, 0, :]  # Shape: [1, 401]

        # FFT process the initial state
        y0 = self.fft_processing(initial_spectrum)  # [1, 4096]

        # Integrate with odeint
        # odeint output: [T, B, 4096], here T=num_spectra, B=1
        ode_out = odeint(self.ode_func, y0, t_span, method='rk4')  
        # ode_out: [num_spectra, 1, 4096]

        # Reshape to [1, num_spectra, 4096]
        ode_out = ode_out.transpose(0, 1).contiguous()  # [1, num_spectra, 4096]

        # We have one sequence: apply fc_out to each time step
        B, T, F = ode_out.shape  # B=1, T=num_spectra, F=4096
        ode_out_flat = ode_out.view(B*T, F)  # [T, 4096]
        pred_flat = self.fc_out(ode_out_flat)  # [T, 4]
        pred = pred_flat.view(B, T, -1)  # [1, T, 4]

        return pred


    def model_step(self, batch):
        spectra, target_changes, t_span = batch
        y_hat = self.ode_forward(spectra, t_span)
        loss = self.loss(y_hat, target_changes)
        return loss, y_hat, target_changes

    def training_step(self, batch, batch_idx):
        spectra_list, target_list, t_span_list = batch
        losses = []
        for spec, tar, ts in zip(spectra_list, target_list, t_span_list):
            # [seq_len, 401] -> [1, seq_len, 401]
            spec = spec.unsqueeze(0).to(self.device)
            ts = ts.to(self.device)
            tar = tar.to(self.device)

            pred = self.ode_forward(spec, ts)  # [1, seq_len, 4]
            loss = self.loss(pred, tar.unsqueeze(0))  # tar: [seq_len,4] -> [1, seq_len,4]
            losses.append(loss)

        final_loss = torch.mean(torch.stack(losses))
        self.log("train_loss", final_loss, batch_size=1)
        return final_loss

    def validation_step(self, batch, batch_idx):
        spectra_list, target_list, t_span_list = batch
        losses = []
        for spec, tar, ts in zip(spectra_list, target_list, t_span_list):
            spec = spec.unsqueeze(0).to(self.device)
            ts = ts.to(self.device)
            tar = tar.to(self.device)

            pred = self.ode_forward(spec, ts)  # [1, seq_len,4]
            loss = self.loss(pred, tar.unsqueeze(0))
            losses.append(loss)

        final_loss = torch.mean(torch.stack(losses))
        self.log("val_loss", final_loss, batch_size=1)
        return final_loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self.model_step(batch)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()
        self.val_loss_best.update(loss)
        self.log("val/loss_best", self.val_loss_best.compute(), on_step=False, on_epoch=True)

checkpoint_callback = ModelCheckpoint(
    monitor="val/loss_best",
    dirpath="/home/nick/Projects/crowpeas/tests/NODE_Rh/checkpoint/",
    filename="nnxanes-{epoch:02d}",
    save_top_k=1,
    mode="min",
)