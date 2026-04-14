from ase.io import read, write
from ase.geometry.analysis import Analysis
import matplotlib.pyplot as plt
import numpy as np
from ase.neighborlist import neighbor_list
import copy

atoms = read('../concat_5.xyz','2339:2340')

species = [('H','H'),('Cu','H'),('Cu','O'),('Cu','Cu'),('O','H'),('O','O')]
#species = [('Cu','Cu'), ('Cu','O')]

ds = {}
for i in species:
    ds[i]=[]


d_max = {('Cu','Cu'):2.0, ('Cu','O'):1.6}
for at in atoms:
    print(at.info)
    for i in ds.keys():
        tmp_d = neighbor_list('d', at,{(i[0],i[1]):3}, self_interaction=False)
        ds[i] = [*ds[i],*tmp_d]

for i in ds.keys():
    hist, bins = np.histogram(ds[i], np.linspace(0,3,100),density=True)
    plt.stairs(hist,bins,label=f'{i[0]}-{i[1]}')
plt.xlabel(r'd ($\rm \AA$)')
plt.ylabel('Count (%)')

plt.legend()
plt.tight_layout()

#ana = Analysis(atoms)

#rdf = ana.get_rdf(3, 40,elements='H')
#plt.plot(np.linspace(0,3,100),np.mean(ana.get_rdf(3, 100,elements='H'),axis=0),label = 'H-H')
#CuO = np.mean(ana.get_rdf(3, 100,elements=('Cu','O')),axis=0)
#CuCu = np.mean(ana.get_rdf(3, 100,elements=('Cu')),axis=0)
#CuO /= max(CuO)
#CuCu /= max(CuCu)
#delta = CuO-CuCu
#delta /= np.sum(delta)/100

#plt.plot(np.linspace(0,3,100),delta,label = 'Cu-O')

plt.show()

