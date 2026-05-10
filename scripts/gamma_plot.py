from ga_analysis.echem import target_diagram
from sage_lib.partition.Partition import Partition
import numpy as np
import glplot.pyplot as gplt
import matplotlib.pyplot as plt



p = Partition(storage='hybrid',
              local_root='/home/hero/data/re_data_base/32x32',
              #local_root='./low_E',t
              access ='ro')



comp = target_diagram(
    reference_potentials= {"Cu": -3.727123268440009, "H2O": -14.253282664300396,  "H": -6.81835453297334/2},
    #H_range= np.linspace(-1.0,0.5,200))
    H_range=np.array([0,1]))
U, E_form, gamma, Ox = comp(p)

gamma *= 1000
a = gamma.T[1]-gamma.T[0]
b = gamma.T[0]



gplt.figure("Density Gain Test", density=True, hud=True, lod=False)
gplt.lines(a, b, x_range=(-1, 0.5), color='yellow')
gplt.xlim(-1.,0.5)
gplt.ylim(0.,400.)
    
print("Check if the 'Density Factor' slider appears in the HUD and affects the heatmap.")
gplt.show()

