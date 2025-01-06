#FILE CHE FUNZIONA CON META TASK/PATTERN O TUTTI TASK/PATTERN 

import itertools
import numpy as np
from functools import reduce
from pyeda.inter import exprvars, And, Or, OneHot, Equal
from countrss_module import _pp_solution, _booldot, ConfigurableXOR, count_rss


n_variables = 4

# Task1: tutti i task
gs1 = np.array(list(itertools.product([0,1], repeat=n_variables)))
ys1 = np.array([reduce(lambda a,b: a^b, g, 0) for g in gs1])
# One-hot encoding per gs1
gvecs1 = []
for g in gs1:
    enc = []
    for val in g:
        enc.extend([1,0] if val == 0 else [0,1])
    gvecs1.append(enc)
gvecs1 = np.array(gvecs1)

# # Task2: solo la metà dei task
# gs2 = gs1[:4]
# ys2 = ys1[:4]
# gvecs2 = gvecs1[:4]

# Ora creiamo i dataset per ogni task
task1 = ConfigurableXOR(n_variables)
task1.load_data(gvecs1, ys1)
print("task1 gvecs", task1.gvecs)
print("task1 ys", task1.ys)


# task2 = ConfigurableXOR(n_variables)
# task2.load_data(gvecs2, ys2)



# Ora applichiamo la funzione a ciascun task
print("### Counting RS for Task1 ###")
count_rss(task1)

# print("### Counting RS for Task2 ###")
# count_rss(task2)
