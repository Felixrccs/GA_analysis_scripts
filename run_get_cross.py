import numpy as np
from scipy.constants import k,e

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

        a_vec = dataset[2].AtomPositionManager.latticeVectors[0, :2]
        b_vec = dataset[2].AtomPositionManager.latticeVectors[1, :2]
        A = abs(np.linalg.det(np.array([a_vec, b_vec])))


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
        print(unique_labels_dict)
        print(X[0])
        X[:,unique_labels_dict['H']] -= 2*X[:,unique_labels_dict['O']]
        print(X[0])
        # Reference chemical potentials for fixed species
        base_mu = np.array([reference_potentials.get(lbl, 0.0) for lbl in unique_labels])
        base_mu[ unique_labels_dict['O'] ] = reference_potentials.get('H2O', 0.0)
        print(base_mu)
        # Formation energy reference
        fE_ref = y - X.dot(base_mu)
        nH = X[:, unique_labels_dict['H']]
        print(y[0], fE_ref[0])
        # Sample H potentials
        H_values = np.linspace(H_range[0], H_range[1], steps)

        # Vectorized formation energies
        fE_array = fE_ref[:, None] - nH[:, None]*H_values[None, :]
        print(fE_array[0])
        fE_array /= A
        print(A)

        
        return H_values, np.array(fE_array)

    return compute


import matplotlib.pyplot as plt
from sage_lib.partition.Partition import Partition





fig,ax = plt.subplots(figsize = (6,4))



#p = Partition(
#    storage='composite',
#    base_root='../data_base/end_32_32_1_inclusive_GA_11',
#    local_root='./'
#)


p = Partition(
    storage='hybrid',
    local_root='../data_base/end_32_32_1_inclusive_GA_11',
    access = 'ro')

p1 = Partition(storage='hybrid',
              local_root='./')

for i,n in enumerate(p):
    if n.AtomPositionManager.metadata['operation']== 'crossover':
        print(i)
        p1.add(n)
    if i ==1000:
        break



comp = target_diagram(reference_potentials= {"Cu": -3.727123268440009, "H2O": -14.253282664300396,  "H": -6.81835453297334/2})
U,data = comp(p1)

minima , index = np.unique(np.argmin(np.abs(data),axis=0),return_index=True)

print(minima, index)
s_min = [x for _, x in sorted(zip(index, minima), key=lambda pair: pair[0])]

p2 = Partition()
for i in s_min:
    print(i)
    p2.add(p1[i])

p1.export_files('tmp.xyz')
#p2.export_files('low.xyz')

