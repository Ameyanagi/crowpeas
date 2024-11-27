import matplotlib.pyplot as plt
from tueplots import figsizes
plt.rcParams['figure.dpi']= 300
import torch
import numpy as np

from laplace.curvature.asdl import AsdlGGN
from laplace import KronLaplace
from tueplots import bundles


from hetreg.utils import TensorDataLoader
from hetreg.models import MLP
from hetreg.marglik import marglik_optimization

import pandas as pd
from torch.utils.data import TensorDataset

print("things loaded")

device = 'cuda'

path_to_x_data = "/home/nick/Projects/crowpeas/BNN_Rh/training_set/Rh training_training_data_spectra.feather"
path_to_y_data = "/home/nick/Projects/crowpeas/BNN_Rh/training_set/Rh training_training_data_parameters.feather"

# Load the data
x_df = pd.read_feather(path_to_x_data)
y_df = pd.read_feather(path_to_y_data)

print(x_df.shape)
print(y_df.shape)

# Convert to tensors
X = torch.tensor(x_df.values, dtype=torch.float64).to(device)
y = torch.tensor(y_df.values, dtype=torch.float64).to(device)

# Verify the shapes of X and y
print(f'X shape: {X.shape}')  # Expecting [num_samples, 401]
print(f'y shape: {y.shape}')  # Expecting [num_samples, 4]


# Create TensorDataLoader
ds_train = TensorDataset(X, y)
train_loader = TensorDataLoader(X, y, batch_size=32, shuffle=True)

# Check the shapes of batches from train_loader
for batch_idx, (data, target) in enumerate(train_loader):
    print(f'Batch {batch_idx} - data shape: {data.shape}, target shape: {target.shape}')
    break  # Check only the first batch


torch.manual_seed(711)

n_samples = 1000
lr = 1e-2
lr_min = 1e-5
lr_hyp = 1e-1
lr_hyp_min = 1e-1
marglik_early_stopping = True
n_epochs = 10000
n_hypersteps = 50
marglik_frequency = 50
laplace = KronLaplace 
optimizer = 'Adam'
backend = AsdlGGN
n_epochs_burnin = 100
prior_prec_init = 1e-3
use_wandb = False


#ds_train = Skafte(n_samples=n_samples, double=True)
#train_loader = TensorDataLoader(ds_train.data.to(device), ds_train.targets.to(device), batch_size=-1)
#xl, xr = ds_train.x_bounds
#offset = 3
#x = torch.linspace(xl-offset, xr+offset, 1000).to(device).double().unsqueeze(-1)


# Heteroscedastic
model = MLP(401, 100, 1, output_size=2, activation='tanh', head='natural', head_activation='softplus').to(device).double()
print(model)
la, model, margliksh, _, _ = marglik_optimization(
    model, train_loader, likelihood='heteroscedastic_regression', lr=lr, lr_min=lr_min, lr_hyp=lr_hyp, early_stopping=marglik_early_stopping,
    lr_hyp_min=lr_hyp_min, n_epochs=n_epochs, n_hypersteps=n_hypersteps, marglik_frequency=marglik_frequency,
    laplace=laplace, prior_structure='layerwise', backend=backend, n_epochs_burnin=n_epochs_burnin,
    scheduler='cos', optimizer=optimizer, prior_prec_init=prior_prec_init, use_wandb=use_wandb
)

plt.plot(margliksh, label='heteroscedastic')
plt.ylabel('log marginal likelihood')
plt.legend()
plt.show()
# f_mu, f_var, y_var = la(x)
# f_mu, f_var, y_var = f_mu.squeeze(), f_var.squeeze(), y_var.squeeze()
# mh_map, sh_map = f_mu.numpy(), 2 * np.sqrt(y_var.numpy())
# mh_bayes, sh_bayes = f_mu.numpy(), 2 * np.sqrt(f_var.numpy() + y_var.numpy())
# sh_emp = 2 * np.sqrt(f_var.numpy())