import numpy as np
from sage_lib.partition.Partition import Partition


p = Partition(storage="hybrid", local_root="/home/hero/data/cleaned_data_base/end_08_08_1", access = 'ro')

species, species_order = p.get_all_compositions(return_species=True)
mapping = p.get_species_mapping(order="stored")

ids = np.where(np.logical_and(species[:,mapping['H']]==1, species[:,mapping['O']]==0))[0]
y = p.get_all_energies()

for i in ids:
    print(p[int(i)].atoms.atomCountDict, y[i])


ids = np.where(np.logical_and(species[:,mapping['H']]==0, species[:,mapping['O']]==0), )[0]
y = p.get_all_energies()

for i in ids:
    print(p[int(i)].atoms.atomCountDict, y[i])


