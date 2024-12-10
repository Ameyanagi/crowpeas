from larch.xafs import xftf, xftr, feffpath, path2chi, ftwindow
from larch.io import read_ascii
from larch.fitting import param, guess, param_group
from larch import Group
from larch.xafs.sigma2_models import sigma2_eins
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from scipy.optimize import root
import numpy as np
import scipy.constants as const






# read data from dir

feff_path_file_rh = "/home/nick/Projects/crowpeas/feff/Rhfeff0001.dat"
feff_path_rh = feffpath(feff_path_file_rh)

feff_path_file_au = "/home/nick/Projects/crowpeas/feff/Aufeff0001.dat"
feff_path_au = feffpath(feff_path_file_au)

feff_path_file_pt = "/home/nick/Projects/crowpeas/feff/Ptfeff0001.dat"
feff_path_pt = feffpath(feff_path_file_pt)

feff_path_file_pd = "/home/nick/Projects/crowpeas/feff/Pdfeff0001.dat"
feff_path_pd = feffpath(feff_path_file_pd)


#target_ein = 350
sample_temp = 298.15

#pred_ss2 = 0.00255

#testss2 = sigma2_eins(298.15, 350, feff_path)

#print(testss2)

EINS_FACTOR = 1.e20*const.hbar**2/(2*const.k*const.atomic_mass)

def solve_for_theta(sigma2, t, path):
    """Compute theta given sigma2, t, and path using numerical root finding."""
    import numpy as np
    from scipy.optimize import brentq

    feffpath = path._feffdat
    if feffpath is None:
        return 0.
    t = max(float(t), 1.e-5)
    sigma2 = max(float(sigma2), 1.e-12)
    rmass = 0.
    for sym, iz, ipot, amass, x, y, z in feffpath.geom:
        rmass += 1.0 / max(0.1, amass)
    rmass = 1.0 / max(1.e-12, rmass)
    A = EINS_FACTOR / (sigma2 * rmass)

    # Define the function whose root we want to find
    def func(theta):
        return theta * np.tanh(theta / (2.0 * t)) - A

    # Find the root of the function
    theta_min = 0
    theta_max = 500
    theta = brentq(func, theta_min, theta_max)
    return theta


# Test the function
#predicted_theta = solve_for_theta(pred_ss2, sample_temp, feff_path)
#print(f"Predicted temperature: {predicted_theta:.2f} K")

# MSE between predicted theta and target theta
#mse = (predicted_theta - target_ein)**2
#print(f"MSE: {mse:.2f}")

# Plot the function
theta = np.linspace(100, 500, 1000)
sigma2_array_rh = np.array([sigma2_eins(sample_temp, t, feff_path_rh) for t in theta])
sigma2_array_au = np.array([sigma2_eins(sample_temp, t, feff_path_au) for t in theta])
sigma2_array_pt = np.array([sigma2_eins(sample_temp, t, feff_path_pt) for t in theta])
sigma2_array_pd = np.array([sigma2_eins(sample_temp, t, feff_path_pd) for t in theta])

plt.plot(theta, sigma2_array_rh, label="Rh", color='red')
plt.plot(theta, sigma2_array_au, label="Au", color='green')
plt.plot(theta, sigma2_array_pt, label="Pt", color='blue')
plt.plot(theta, sigma2_array_pd, label="Pd", color='black')
# add predicted theta to plot as red dot
ss2_nn_rh = 0.0026
ss2_art_rh = 0.0037
theta_rh = 350 * (1/1.27)
plt.scatter(theta_rh, ss2_nn_rh, color='red', label = "Rh NN")

ss2_nn_au = 0.0080
ss2_art_au = 0.0080
theta_au = 178 * (1/1.27)
plt.scatter(theta_au, ss2_nn_au, color='green', label = "Au NN")

ss2_nn_pt = 0.0041
ss2_art_pt = 0.0050
theta_pt = 225 * (1/1.27)
plt.scatter(theta_pt, ss2_nn_pt, color='blue', label = "Pt NN")

ss2_nn_pd = 0.0052
ss2_art_pd = 0.006
theta_pd = 275 * (1/1.27)
plt.scatter(theta_pd, ss2_nn_pd, color='black', label = "Pd NN")



#add artemis result
#plt.scatter(theta_rh, ss2_art_rh, color='blue', label = "Rh Art")
#plt.scatter(theta_au, ss2_art_au, color='blue', label = "Au Art")
#plt.scatter(theta_pt, ss2_art_pt, color='blue', label = "Pt Art")
#plt.scatter(theta_pd, ss2_art_pd, color='blue', label = "Pd Art")
# add art result to plot with open diamond marker
plt.scatter(theta_rh, ss2_art_rh, color='red', marker='D', label = "Rh Art")
plt.scatter(theta_au, ss2_art_au, color='green', marker='D', label = "Au Art")
plt.scatter(theta_pt, ss2_art_pt, color='blue', marker='D', label = "Pt Art")
plt.scatter(theta_pd, ss2_art_pd, color='black', marker='D', label = "Pd Art")



plt.xlabel("Theta")
plt.ylabel("Sigma^2")
plt.title("Sigma^2 vs Theta for Room temp sample")


plt.savefig("sigma2_vs_theta.png")

#print(sigma2_array_au[:10], sigma2_array_pt[:10])



