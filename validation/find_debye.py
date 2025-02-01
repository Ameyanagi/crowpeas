import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve

def debye_integrand(x):
    # Integrand for the Debye heat capacity integral
    return (x**4 * np.exp(x)) / (np.exp(x) - 1)**2

def heat_capacity(T, Theta_D):
    # R: gas constant in J/(mol·K)
    R = 8.31446261815324
    # Debye heat capacity equation:
    # C_V(T) = 9 * R * (T/Theta_D)^3 * ∫_0^{Theta_D/T} [x^4 * exp(x)/(exp(x)-1)^2] dx
    upper_limit = Theta_D / T
    integral_val, _ = quad(debye_integrand, 0, upper_limit)
    return 9 * R * (T / Theta_D)**3 * integral_val

def find_debye_temp(T, C_target, guess=150.0):
    # Use fsolve to find the Theta_D that gives the measured heat capacity
    def func(Theta_D):
        return heat_capacity(T, Theta_D) - C_target

    Theta_D_solution = fsolve(func, guess)[0]
    return Theta_D_solution

if __name__ == "__main__":
    # Example usage:
    T = 298.0
    C_measured = 25.0  # J/(mol·K), example value
    Theta_D_est = find_debye_temp(T, C_measured)
    print(f"Estimated Debye Temperature at {T} K: {Theta_D_est:.2f} K")
