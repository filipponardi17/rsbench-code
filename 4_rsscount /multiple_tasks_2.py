#FILE CHE FUNZIONA CON UN SINGOLO TASK / PATTERN

import itertools
import numpy as np
from functools import reduce
from pyeda.inter import exprvars, And, Or, OneHot, Equal
from countrss_module import _pp_solution, _booldot, ConfigurableXOR
from countrss_module import count_rss


n_variables = 3


#task1 solo con il pattern [0,0,0]

gs1 = np.array([[0,0,0]])  # singolo esempio
ys1 = np.array([reduce(lambda a,b: a^b, g, 0) for g in gs1])

gvecs1 = []
for g in gs1:
    enc = []
    for val in g:
        enc.extend([1,0] if val == 0 else [0,1])
    gvecs1.append(enc)
gvecs1 = np.array(gvecs1)

task1 = ConfigurableXOR(n_variables=3)
task1.load_data(gvecs1, ys1)

print("task1 gvecs", task1.gvecs)
print("task1 ys", task1.ys)

print("### Counting RS for Task1 ###")
count_rss(task1)