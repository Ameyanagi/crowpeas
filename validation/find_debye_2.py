import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

def debye_integrand(x):
    # Integrand for the Debye heat capacity integral
    # Well-defined as x->0, but let’s rely on quad to handle that.
    return (x**4 * np.exp(x)) / (np.exp(x) - 1)**2

def heat_capacity(T, Theta_D):
    # Ensure Theta_D > 0
    if Theta_D <= 0:
        # Return a large positive/negative number to indicate no solution here
        return np.inf
    
    R = 8.31446261815324
    upper_limit = Theta_D / T
    # If upper_limit is too small (close to 0), integral will be small -> near 0 capacity.
    integral_val, _ = quad(debye_integrand, 0, upper_limit)
    return 9 * R * (T / Theta_D)**3 * integral_val

def bracket_root(func, start=1.0, step=10.0, max_tries=1000, lower_bound=1.0):
    """
    Attempts to find a sign change starting at `start` and moving upward.
    Also prevents going below `lower_bound` to avoid zero or negative Theta_D.
    """
    if start < lower_bound:
        start = lower_bound

    # Move upward
    low = start
    f_low = func(low)
    high = start
    for _ in range(max_tries):
        high += step
        f_high = func(high)
        if f_low * f_high < 0:
            return low, high

    # Move downward, but not below lower_bound
    low = start
    f_low = func(low)
    high = start
    for _ in range(max_tries):
        new_low = low - step
        if new_low < lower_bound:
            break  # Don't go below lower_bound
        low = new_low
        f_low = func(low)
        if f_low * f_high < 0:
            return low, high

    raise ValueError("No sign change found for the function within given steps.")

def find_debye_temp(T, C_target, start_guess=300.0):
    def func(Theta_D):
        return heat_capacity(T, Theta_D) - C_target

    a, b = bracket_root(func, start=start_guess)
    Theta_D_solution = brentq(func, a, b)
    return Theta_D_solution

if __name__ == "__main__":
    # Example usage:
    T = 298.0
    C_measured = 25.0  # J/(mol·K), example input
    Theta_D_est = find_debye_temp(T, C_measured, start_guess=300.0)
    print(f"Estimated Debye Temperature at {T} K: {Theta_D_est:.2f} K")
