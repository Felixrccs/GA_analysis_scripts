from sage_lib.partition.Partition import Partition
import matplotlib.pyplot as plt
from ga_analysis.plot_defaults import set_default_params
import numpy as np

set_default_params()

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


E_CuH = -1137.3636474609375
E_Cu = -1133.7833251953125
mu_Cu = -14.916443703626898 / 4


only_minima = False

# 2. Group by chemical formula
groups = {}
for i, f in enumerate(formulas):
    if f[2] == 0 and f[1]>0 and f[0]==320:
        n_Cu = f[0] - 320
        E = (energies[i] - f[1]*E_CuH + (f[1]-1)*E_Cu - mu_Cu*n_Cu)/f[1]
        n = f[1]
        if n not in groups:
            groups[n] = []
        groups[n].append((E, i))

n_H, E_arr, indices = [], [], []
for n in sorted(groups.keys()):
    entries = np.array(groups[n])
    indices.append(entries[np.argmin(entries[:,0]),1])
    if only_minima:
        entries = [min(entries, key=lambda x: x[0])]
    for E, idx in entries:
        n_H.append(n)
        E_arr.append(E)

E_arr = np.array(E_arr)

E_arr *= 1000

fix, ax = plt.subplots(figsize = (5,3))
ax.scatter(n_H, E_arr, marker='o', label='E',c='tab:blue',s=1)
E_min = np.min(E_arr)
ax.set_ylim([E_min-10,E_min+120])

ax.set_ylabel(r'$E_{\rm Lateral}/N_{\rm H}$ (meV)')
ax.set_xlabel(r'$N_{\rm H}$')
plt.savefig('../plots/H_lateral.png', dpi=300, bbox_inches='tight')
plt.show()

print(indices)
new_db = Partition(
    storage="hybrid",
    local_root="../output/H_db",
    access="rw"
)


for i in indices:
    new_db.add(db[int(i)])
    