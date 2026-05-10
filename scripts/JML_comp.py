import numpy as np
from sage_lib.partition.Partition import Partition


db = Partition(storage="hybrid", local_root="/home/hero/data/cleaned_data_base/end_08_08_1", access = 'ro')


from collections import defaultdict

# 1. Bulk extraction for performance
formulas = db.get_formulas()
energies = db.get_energies()
mapping = db.get_species_mapping(order="stored")


# 2. Group by chemical formula
groups = defaultdict(list)
for i, c in enumerate(formulas):
    formula = f"Cu{int(c[mapping['Cu']])}O{int(c[mapping['O']])}H{int(c[mapping['H']])}"

    if formula in ['Cu320O1H1','Cu320O0H0']:
        groups[formula].append(i)

# 3. Select top-1 per composition using vectorized lookup
final_selection = []
for formula, indices in groups.items():
    # Sort local group by pre-extracted energies
    local_energies = energies[indices]
    best_idx = indices[np.argmin(local_energies)]
    print(formula, energies[best_idx])
    final_selection.append(best_idx)



db.export_subset(final_selection, "1OH.xyz")
