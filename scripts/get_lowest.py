# gpu_lines_instanced_clip_fixed.py
# Millions of lines y = a*x + b with one-time GPU upload,
# instanced GL_LINES, shader-side clip planes (no GS), transparency,
# zoom-aware on-GPU LOD, and robust early Y-culling (clamped to [xmin,xmax]).

import sys
import ctypes as C
import numpy as np
import glfw
from OpenGL.GL import *
from scipy.constants import k,e
from sage_lib.partition.Partition import Partition

def get_delta_mu_H(U, pH = 0, T=298, p_H2 =1):
    """
    Transforms a applied_potential into delta mu_H (H+ + e-) via CHE

    Parameters
    ----------
    U : float / array,
        applied_potential (V)
    pH : float
        pH Value
    T : float, 298
        Temperature in K
    p_H2 : float, 1
        hydrogen gas_phase pressure in K

    Returns
    -------
    mu_H : float / array,
        (H+ + e-) chemical potential in eV
    """
    return (
        - (2.303 * k * T / e) * pH
        - (k * T / (2 * e)) * np.log(p_H2 / 1.)
        - U  # main dependence
    )

def get_delta_U(d_mu, pH = 0, T=298, p_H2 = 1):
    """
    Transforms a delta mu_H (H+ + e-) applied_potential via CHE

    Parameters
    ----------
    m_H : float / array,
        (H+ + e-) chemical potential in eV
    pH : float
        pH Value
    T : float, 298
        Temperature in K
    p_H2 : float, 1
        hydrogen gas_phase pressure in K

    Returns
    -------
    U : float / array,
        applied_potential (V)
    """
    return (
        - (2.303 * k * T / e) * pH
        - (k * T / (2 * e)) * np.log(p_H2 / 1.)
        - d_mu  # main dependence
    )


def get_bulk_classic(mu_H2O, mu_H):
    """
    calculation of the bulk transition

    Parameters
    ----------
    mu_H2O : float
        H2O chemical potential in eV
    m_H_0 : float
        standard (H+ + e-) chemical potential in eV

    Returns
    -------
    delta mu_H : float,
        (H+ + e-) chemical potential in eV
    """
    ref_Cu_int = -14.913209 / 4
    ref_Cu2O_int = (-27.27211939) / 4 # two oxygen

    delta_O = 2 * (ref_Cu2O_int - ref_Cu_int)
    return (mu_H2O-delta_O)/2 - mu_H

def objective_min_distance_to_electrochemicalhull(
    reference_potentials: dict,
    H_range: tuple = (-1.0, 0.5),
    steps: int = 100,
    unique_labels: list = None,
):
    """
    Objective function for GA: minimal distance of each structure to the convex hull
    across a range of applied electrochemical potentials (U).

    The electron chemical potential is varied via the CHE formalism:
        mu_e(U) = - e * U + pH- and p_H2-dependent terms.

    Parameters
    ----------
    reference_potentials : dict
        Dictionary of fixed chemical potentials, e.g. {'Cu': -3.5, 'O': -4.2, 'H2O': -14.25}.
        These are constants and serve as the baseline for non-variable species.
    H_range : tuple
        (H_min, H_max) range of applied potential (in eV).
    steps : int
        Number of discrete U values to sample between H_min and H_max.

    Returns
    -------
    compute : callable
        Function that, when called with a list of structures, returns
        min_distances: np.ndarray of shape (N_structs,)
            Minimum energy distance to convex hull for each structure across U_range.
    """
    unique_labels = {lbl for lbl in reference_potentials.keys()}.union({'O','H'}) - {'H2O'}
    unique_labels_dict = { u:i for i, u in enumerate(unique_labels) }
    M = len(unique_labels)

    def compute(dataset):
        """
        Compute min distance to convex hull for each structure across sampled U values.

        Structures are expected to provide:
            - structure.AtomPositionManager.E : total energy (eV)
            - structure.AtomPositionManager.latticeVectors : (3,3) array for cell vectors
        """

        # 1) Unique labels: hard coded for application
        #unique_labels = ['H','O','Cu']

        # 2) Build composition matrix X and energy array y
        N = len(dataset)

        # Fill composition counts and energies
        y = dataset.get_all_energies()

        species, species_order = dataset.get_all_compositions(return_species=True)
        mapping = dataset.get_species_mapping(order="stored")
        idx = np.fromiter((mapping.get(lbl, -1) for lbl in unique_labels), dtype=int, count=len(unique_labels))
        valid = (idx >= 0)
        X = np.zeros((species.shape[0], len(unique_labels)), dtype=species.dtype)
        if np.any(valid):
            X[:, valid] = species[:, idx[valid]]

        # 3) CHE adjustment Adjust for mu_O = mu_H2O - 2mu_H
        X[:,unique_labels_dict['H']] -= 2*X[:,unique_labels_dict['O']]

        # Reference chemical potentials for fixed species
        base_mu = np.array([reference_potentials.get(lbl, 0.0) for lbl in unique_labels])
        base_mu[ unique_labels_dict['O'] ] = reference_potentials.get('H2O', 0.0)

        # Formation energy reference + # Add vibrational correction
        y = y - 1.231 + 0.314*X[:,unique_labels_dict['H']] - 0.026*X[:,unique_labels_dict['Cu']] + 0.156*X[:,unique_labels_dict['O']]
        fE_ref = y - X.dot(base_mu)
        nH = X[:, unique_labels_dict['H']]

        # Sample H potentials
        H_values = np.linspace(H_range[0], H_range[1], steps)

        # Vectorized formation energies
        fE_array = fE_ref[:, None] - nH[:, None]*H_values[None, :]
        fE_hull = fE_array.min(axis=0)
        min_distances = (fE_array - fE_hull).min(axis=1)

        return fE_array.astype(np.float32), min_distances

    return compute




if __name__ == "__main__":

    func = objective_min_distance_to_electrochemicalhull(
        reference_potentials={
            "Cu":  -14.916443703626898 / 4,
            "H2O": -14.25,
            "H":   -6.81835453297334 / 2,
        },
        H_range=(-1.0, 0.5),
        steps=200,
    )
    
    p = Partition(storage="hybrid", local_root="/home/hero/data/cleaned_data_base/end_08_08_1", access = 'ro')
    ab, min_distances = func(p)
    sub_ids = np.argsort(min_distances)[0:100]
    print(min_distances[sub_ids])

    p1 = Partition(storage="hybrid", local_root="./vib_test", access = 'rw')
    for i in sub_ids:
        print(i)
        p1.add(p[int(i)])


   # p1.export_files('tmp.xyz')


