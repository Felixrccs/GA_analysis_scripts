import numpy as np
from scipy.constants import k,e
from ase.calculators.singlepoint import SinglePointCalculator

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


def target_diagram(
    reference_potentials: dict,
    H_range: tuple = (-1.0, 0.5),
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

    def compute(structures):
        """
        Compute min distance to convex hull for each structure across sampled U values.

        Structures are expected to provide:
            - structure.AtomPositionManager.E : total energy (eV)
            - structure.AtomPositionManager.latticeVectors : (3,3) array for cell vectors
        """

        for at in structures:
            try:
               at.info['E'] = at.get_potential_energy()
            except:
              at.info['E'] = 0.0

        # 1) Unique labels: hard coded for application
        unique_labels = ['H','O','Cu']

        # 2) Build composition matrix X and energy array y
        N = len(structures)
        M = len(unique_labels)
        X = np.zeros((N, M), dtype=float)
        y = np.zeros(N, dtype=float)

        for i, struct in enumerate(structures):
            y[i] = struct.info['E']
            labels_array = np.array(struct.get_chemical_symbols())
            for j, lbl in enumerate(unique_labels):
                X[i, j] = np.count_nonzero(labels_array == lbl)


        # 3) Adjust for mu_O = mu_H2O - 2mu_H
        X[:,0]-=2*X[:,1]
        
        # Reference chemical potentials for fixed species
        base_mu = np.array([reference_potentials.get(lbl, reference_potentials.get('H2O',0.0)) for lbl in unique_labels])

        # 4) Precompute reference part of formation energy
        fE_ref = y - X.dot(base_mu)

        # 5) mu_H sampling
        H_values = np.linspace(H_range[0], H_range[1], steps)


        allfE = [] 

        for H in H_values:
            # 6) Formation energies including change in mu_H from standard conditions (pH=0, U=0V vs SHE)
            fE = fE_ref - X[:, 0] * H
            
            allfE.append(fE)

        allfE = np.array(allfE)

        pareto = np.argmin(allfE,axis=1)
        _, ids = np.unique(pareto, return_index=True)
        new_atoms = [structures[ind] for ind in pareto[sorted(ids)]]
        return new_atoms

    return compute



from ase.io import read,write

#atoms = read(f'/home/friccius/data/MACE/MACE_hero_run/ga_dir/generation_5/supercell_4_4_1/config_all.xyz',':')
atoms = read(f'./out.xyz',":")
# calculate surface free energies
comp = target_diagram(reference_potentials= {"Cu": -3.727123268440009, "H2O": -14.253282664300396,  "H": -6.81835453297334/2})
out = comp(atoms)
write('./pareto.xyz',out)

from ase.visualize import view
view(out)



