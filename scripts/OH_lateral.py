
from sage_lib.partition.Partition import Partition
import matplotlib.pyplot as plt
import numpy as np

# Open database in read-only mode for analysis
db = Partition(
    storage="hybrid",
    local_root="/home/hero/data/cleaned_data_base/end_08_08_1",
    access="ro"
)

print(f"Loaded {len(db)} structures.")


# 1. Bulk extraction for performance
formulas = db.get_formulas()
energies = db.get_energies()
print(db.get_species_mapping(order="stored"))



E_CuOH = -1144.5845947265625
E_Cu = -1133.7833251953125
mu_Cu = -14.916443703626898 / 4
mu_H = -6.81835453297334/2


fig, ax = plt.subplots(figsize=(5, 4), sharex=True, sharey=True)


mu_H_eff = mu_H + 0.

plot_minima = False  # True: plot only minimum E per n_O; False: plot all E

seen = {}
for i, f in enumerate(formulas):
    if abs(f[1]-f[2]) < f[2]/8 and f[2] > 0:
        n_O = f[2]
        n_Cu = f[0] - 320
        n_H = f[1] - f[2]
        E = (energies[i] - n_O*E_CuOH + (n_O-1)*E_Cu - n_Cu*mu_Cu - n_H*mu_H_eff) / n_O
        if n_O not in seen:
            seen[n_O] = []
        seen[n_O].append(E)

if plot_minima:
    x = np.array(list(seen.keys()))
    y = np.array([min(Es) for Es in seen.values()]) * 1000
else:
    x = np.array([n_O for n_O, Es in seen.items() for _ in Es])
    y = np.array([E for Es in seen.values() for E in Es]) * 1000

E_min = y.min()
print('scatter')
if plot_minima:
    ax.scatter(x, y, s=1, c='tab:blue')
else:
    ax.plot(x, y, '.', ms=1, color='tab:blue', rasterized=True)
ax.set_xlabel(r'$N_{\rm O}$')
ax.set_ylabel(r'$E_{\rm Lateral}/N_{\rm O}$ (meV)')
ax.set_ylim([E_min-10, E_min+400])

plt.tight_layout()
plt.savefig('../plots/OH_lateral.png', dpi=300, bbox_inches='tight')
plt.show()


