import numpy as np
from scipy.spatial import ConvexHull



def objective_min_distance_to_hull_electrochemical(
    reference_potentials: dict,
    variable_species: str = "H",
    A: float = None,
    H_range: tuple = (-1.2, 0.5),
    steps: int = 100
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
    variable_species : str
        Label of the species whose potential is varied. Typically 'e-' or 'H'.
    A : float
        Surface area in Å² (used to normalize formation energies). If None, area
        will be computed per structure from lattice vectors.
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

    A_cte = isinstance(A, float)

    def compute(structures):
        """
        Compute min distance to convex hull for each structure across sampled U values.

        Structures are expected to provide:
            - structure.AtomPositionManager.atomLabelsList : array of atomic labels
            - structure.AtomPositionManager.E : total energy (eV)
            - structure.AtomPositionManager.latticeVectors : (3,3) array for cell vectors
        """

        # 1) Collect unique labels
        unique_labels = ['H','O','Cu']

        # 2) Build composition matrix X and energy array y
        N = len(structures)
        M = len(unique_labels)
        X = np.zeros((N, M), dtype=float)
        y = np.zeros(N, dtype=float)

        A_array = np.zeros(N) if not A_cte else A
        for i, struct in enumerate(structures):
            y[i] = getattr(struct.AtomPositionManager, 'E', 0.0)
            labels_array = struct.AtomPositionManager.atomLabelsList
            for j, lbl in enumerate(unique_labels):
                X[i, j] = np.count_nonzero(labels_array == lbl)

            if not A_cte:
                # 2D surface area from first two lattice vectors
                a_vec = struct.AtomPositionManager.latticeVectors[0, :2]
                b_vec = struct.AtomPositionManager.latticeVectors[1, :2]
                A_array[i] = abs(np.linalg.det(np.array([a_vec, b_vec])))



        X[:,0]-=2*X[:,1]
        
        # Reference chemical potentials for fixed species
        base_mu = np.array([reference_potentials.get(lbl, reference_potentials.get('H2O',0.0)) for lbl in unique_labels])

        # 4) Precompute reference part of formation energy
        mu_zeroed = base_mu.copy()
        fE_ref = y - X.dot(mu_zeroed)

        # 5) Potential sampling
        H_values = np.linspace(H_range[0], H_range[1], steps)


        min_distances = np.full(N, np.inf)
        gamma = [] 

        for H in H_values:
            # 6) Formation energies including variable mu
            fE = fE_ref - X[:, 0] * H
            fE = fE / A_array  # normalize by area
            
            gamma.append(fE)
        return H_values, np.array(gamma).T

    return compute

from ase.io import read
import matplotlib.pyplot as plt
from sage_lib.partition.Partition import Partition

atoms = read('/home/friccius/data/Production_run/MACE_run/MACE_diagram/absorbates_out.xyz',':')
new_at = []
for at in atoms:
    if at.info['config_type']=='optimize_last_converged':
        at.info = {'E': at.info['optimize_energy']}
        new_at.append(at)


#atoms = read('/home/friccius/data/Production_run/MACE_run/MACE_diagram/a',':')
#new_at = []

partition=Partition()
partition.read_files(f'/home/friccius/Documents/tmp/supercell_2_2_1/generation/out.xyz')

comp = objective_min_distance_to_hull_electrochemical(reference_potentials={'Cu': -14.916443703626898/4, 'H': -6.81835453297334/2, 'H2O': -14.25})
U,data = comp(partition)
for i,val in enumerate(data):
    plt.plot(U,val, label =i)
    plt.xlabel('mu_H - mu_H_0 (eV)')

print(np.argmin(data.T,axis=1))

plt.show()

