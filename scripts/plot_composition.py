from ga_analysis.utils import calculate_boltzmann_probs
from ga_analysis.echem import target_diagram, get_bulk_classic, get_delta_mu_H, get_delta_U
from ga_analysis.plot_defaults import set_default_params
from sage_lib.partition.Partition import Partition

import numpy as np
import matplotlib.pyplot as plt


set_default_params()

p = Partition(storage='hybrid',
              #local_root='/home/hero/data/final_data_base/end_04_04_1',
              local_root='/home/hero/data/cleaned_data_base/end_08_08_1',
              access ='ro')

species, species_order = p.get_all_compositions(return_species=True)



print('start')
comp = target_diagram(
    reference_potentials= {"Cu": -3.727123268440009, "H2O": -14.253282664300396,  "H": -6.81835453297334/2},
    H_range= np.linspace(-1.0,0.5,200))
U, E_form, gamma, Ox = comp(p)


P, log_Z, log_P = calculate_boltzmann_probs(E_form, 300)



coverage = np.matmul(P.T,species).T/np.square(8)
fig, ax = plt.subplots(figsize = (7,4))

for i in range(coverage.shape[0]):
    cov = coverage[i]
    if species_order[i]=='Cu':
        cov-=5.
    ax.plot(U,coverage[i], label= species_order[i],lw=2)
ax.legend()

ax.set_xlabel(r"$\mu_{\rm H} = \frac{1}{2}(\mu_{\rm H_2O} - \mu_{O})$ (eV)")
ax.set_ylabel(r"$\Theta$ (ML)")
ax.set_xlim([0.5,-1.0])
ax.set_ylim([-0.03,1.03])
ax.vlines(get_bulk_classic(-14.253282664300396, -6.81835453297334/2),-0.1,100.,colors='b',lw=1)
ax.fill_between([-2.,get_bulk_classic(-14.253282664300396, -6.81835453297334/2)],[400,400], [-100,-100],fc='b',alpha=0.15, zorder=1)
ax.text(-0.8,0.2,'Cu$_2$O-bulk',va='center',ha='center', bbox=dict(facecolor=(1,1,1), edgecolor=(0,0,0)))

secax = ax.secondary_xaxis('top', functions=(get_delta_mu_H, get_delta_U))
secax.set_xlabel(r'$U_{\rm RHE} / U_{\rm SHE}$ (pH=0) (V)')

        # SHE pH = 13 (Literature, compare to slides)
third = ax.secondary_xaxis(1.2, functions=(lambda x: get_delta_mu_H(x,pH=13), lambda x: get_delta_U(x,pH=13)))
third.set_xlabel(r'$U_{\rm SHE}$ (pH=13) (V) literature')

plt.savefig('../plots/composition.png',dpi=300, bbox_inches="tight")
plt.show()


