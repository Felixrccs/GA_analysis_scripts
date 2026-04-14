
import numpy as np
from sage_lib.partition.Partition import Partition
from ga_analysis.echem import target_diagram


func = target_diagram(
    reference_potentials={
        "Cu":  -14.916443703626898 / 4,
        "H2O": -14.25,
        "H":   -6.81835453297334 / 2,
    },
    H_range=np.linspace(-1.0, 0.5,400),
)

p = Partition(storage="hybrid", local_root="/home/hero/data/re_data_base/32x32", access = 'ro')
_, E ,_ , Ox  = func(p)
sub_ids = np.unique(np.argmin(E,axis=0))
print(sub_ids)
print(Ox[sub_ids])
print(sub_ids[np.argsort(Ox[sub_ids])])


p1 = Partition()
for i in sub_ids[np.argsort(Ox[sub_ids])]:
    print(i)
    p1.add(p[int(i)])
p1.export_files('low_E_front.xyz')


