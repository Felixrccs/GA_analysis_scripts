from ga_analysis.echem import target_diagram
from sage_lib.partition.Partition import Partition
from scipy.special import logsumexp
import numpy as np

def calculate_boltzmann_probs(energies, T, sigma_ml=0.0):
    """
    Robust Boltzmann calculation with Bayesian Uncertainty Correction.
    
    Mathematics:
    If the predicted energy E_m has a Gaussian error epsilon ~ N(0, sigma^2), 
    the expected statistical weight <exp(-E/kT)> is calculated by integrating 
    over the error distribution:
    
    <exp(-E/kT)> = integral[ exp(-(E_m - eps)/kT) * P(eps) ] deps
                 = exp(-E_m/kT) * integral[ exp(eps/kT) * P(eps) ] deps
    
    The integral is the Moment Generating Function of a Gaussian, resulting in:
    Weight = exp(-E_m/kT) * exp(sigma^2 / (2 * (kT)^2))
    
    This correction factor ensures that structures with high uncertainty don't 
    biasing the ensemble averages.
    
    Args:
        energies (np.array): Formation energies in eV.
        T (float): Temperature in Kelvin.
        sigma_ml (float): Prediction uncertainty (MAE/RMSE) in eV.
    
    Returns:
        probabilities, log_partition_function (ln Z), log_probs
    """
    kB = 8.617333262e-5
    beta = 1.0 / (kB * T)
    
    # 1. BAYESIAN CORRECTION TERM
    # This accounts for the 'broadening' of the energy states due to ML error.
    correction = ( (beta * sigma_ml)**2 ) / 2.0
    
    # 2. LOG-DOMAIN WEIGHTS
    # We stay in log-space to prevent overflow in the Partition Function.
    x = (-energies * beta) + correction
    
    # 3. LOG-SUM-EXP (ln Z)
    # Using scipy.special.logsumexp for high-performance numerical stability.
    log_z = logsumexp(x, axis=0)
    print(log_z.shape)
    
    # 4. LOG-PROBABILITIES
    # ln(P_i) = ln(w_i) - ln(Z)
    log_probs = x - log_z[np.newaxis, :]
    probs = np.exp(log_probs)

    
    return probs, log_z, log_probs


p = Partition(storage='hybrid',
              local_root='/home/hero/data/re_data_base/32x32',
              #local_root='./low_E',t
              access ='ro')



print('start')
comp = target_diagram(
    reference_potentials= {"Cu": -3.727123268440009, "H2O": -14.253282664300396,  "H": -6.81835453297334/2},
    H_range= np.linspace(-1.0,0.5,200))
U, E_form, gamma, Ox = comp(p)


P, _, log_P = calculate_boltzmann_probs(E_form, 300)
print(P.shape)

ids = np.unique(np.argmax(P, axis=0))
print(ids)


import matplotlib.pyplot as plt

for i in ids:
    plt.plot(U,P[i])

plt.show()