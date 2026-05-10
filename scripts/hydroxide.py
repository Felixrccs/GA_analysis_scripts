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


fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

for ax, dmu in zip(axes.flat, [-1., -0.5, 0., 0.5]):
    mu_H_eff = mu_H + dmu

    x, y, c = [], [], []
    seen = {}
    for i, f in enumerate(formulas):
        if abs(f[1]-f[2]) < f[2]/8 and f[2] > 0:
            n_O = f[2]
            n_Cu = f[0] - 320
            n_H = f[1] - f[2]
            E = (energies[i] - n_O*E_CuOH + (n_O-1)*E_Cu - n_Cu*mu_Cu - n_H*mu_H_eff) / n_O
            key = (n_O, n_Cu)
            if key not in seen or E < seen[key]:
                seen[key] = E
    for (n_O, n_Cu), E in seen.items():
        x.append(n_O)
        y.append(n_Cu)
        c.append(E)

    x_vals = sorted(set(x))
    y_vals = sorted(set(y))
    x_idx = {v: j for j, v in enumerate(x_vals)}
    y_idx = {v: j for j, v in enumerate(y_vals)}

    grid = np.full((len(y_vals), len(x_vals)), np.nan)
    for xi, yi, ci in zip(x, y, c):
        grid[y_idx[yi], x_idx[xi]] = ci

    vmin = np.nanmin(c)
    im = ax.imshow(grid, origin='lower', aspect='auto', cmap='viridis',
                   vmin=vmin, vmax=vmin+0.4,
                   extent=[min(x_vals)-0.5, max(x_vals)+0.5,
                           min(y_vals)-0.5, max(y_vals)+0.5])
    plt.colorbar(im, ax=ax, label='E (eV)')
    ax.set_title(rf'$\mu_H = {dmu}$')
    ax.set_xlabel(r'$N_{\rm O}$')
    ax.set_ylabel(r'$N_{\rm Cu}$')

plt.tight_layout()
plt.savefig('../plots/CuO.png', dpi=300, bbox_inches='tight')
plt.show()


