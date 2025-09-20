


def objective_min_distance_to_hull_electrochemical(
    reference_potentials: dict,
    variable_species: str = "H",
    A: float = None,
    U_range: tuple = (-1.0, 1.0),
    steps: int = 20,
    R: float = 8.3145e-3,   # eV/K (gas constant in eV)
    T: float = 300.0,       # temperature in K
    F: float = 96_485.0,    # C/mol (Faraday constant)
    pH: float = 0.0,
    p_H2: float = 1.0,
    p0: float = 1.0
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
    U_range : tuple
        (U_min, U_max) range of applied potential (in Volts).
    steps : int
        Number of discrete U values to sample between U_min and U_max.
    R, T, F : float
        Thermodynamic constants (gas constant [eV/K], temperature [K], Faraday constant).
    pH : float
        Electrolyte pH, entering the CHE relation.
    p_H2 : float
        Partial pressure of H2 (bar).
    p0 : float
        Standard pressure (bar).

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
        import numpy as np
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
            fE = fE_ref - (X[:, 0] - 2* X[:, 1]) * H
            fE = fE / A_array  # normalize by area
            fE -= np.min(fE)

            min_distances = np.minimum(min_distances, dist)

        return min_distances

    return compute

from ase.io import read

atoms = read('/home/friccius/data/Production_run/MACE_run/MACE_diagram/absorbates_out.xyz',':')
new_at = []
for at in atoms:
    if at.info['config_type']=='optimize_last_converged':
        at.info = {'E': at.info['optimize_energy']}
        new_at.append(at)



comp = objective_min_distance_to_hull_electrochemical(reference_potentials={'Cu': -3.5, 'H': -4.2, 'H2O': -14.25})
comp(new_at)
