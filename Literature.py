import os
from ase.io import read, write
import numpy as np
from mace.calculators import MACECalculator
import matplotlib.pyplot as plt
from itertools import chain
from ase.visualize import view

files = ['/home/friccius/data/Production_run/strategical/benchmark/DFT/out_zhu_44+7O.xyz', '/home/friccius/data/Production_run/strategical/benchmark/DFT/out_zhu_44.xyz','/home/friccius/data/Production_run/strategical/benchmark/DFT/zhu_29+1O.xyz','/home/friccius/data/Production_run/strategical/benchmark/DFT/Cu196.xyz']

new_atoms = []
for f in files:
    atoms = read(f)
    atoms.info = {}
    atoms.set_cell(atoms.cell*2.569/2.568,scale_atoms=True)
    atoms.arrays["fixed"] = np.zeros(len(atoms),dtype=int)
    new_atoms.append(atoms)


calculator = MACECalculator(model_paths='/home/friccius/data/MACE/MACE_hero_run/MACE/mace_7_stagetwo.model', device='cpu')

gamma = []
for at in new_atoms:
    at.calc = calculator
    at.info['energy'] = at.get_potential_energy()
    gamma.append(at)

view(gamma)

write(f'./Lit.xyz',gamma)

