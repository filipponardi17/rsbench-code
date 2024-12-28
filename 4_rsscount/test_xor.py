from countrss_module import XorDataset
import itertools
import numpy as np
from pyeda.inter import exprvars, And, Or, Xor, Implies, OneHot, Equal
from functools import reduce
from countrss_module import _pp_solution, _booldot, count_rss

# Genera tutti i possibili input per 3 variabili
gs = np.array(list(itertools.product([0,1], repeat=3)))

# Calcola ys applicando XOR su ogni riga di gs
ys = np.array([reduce(lambda a,b: a^b, g, 0) for g in gs])

# Ora crei l'istanza di XorDataset
class Args:
    n_variables = 3
    from_cnf = None
    concept_sup = 0.0
    subsample = 1.0
    print_data = False
    store_litmap = False
    enumerate = True
    seed = 1
    
args = Args()

dataset = XorDataset(args)

# Ora assegni manualmente gvecs e ys
dataset.gvecs = np.zeros((len(gs), dataset.n_bits))
for i, g in enumerate(gs):
    # One-hot encoding manuale, ad es. per due valori per variabile
    
    encoding = []
    for val in g:
        if val == 0:
            encoding.extend([1,0])
        else:
            encoding.extend([0,1])
    dataset.gvecs[i] = encoding

dataset.ys = ys

print("### Counting RS ###")
count_rss(dataset)

