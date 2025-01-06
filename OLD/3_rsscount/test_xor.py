from countrss_module import XorDataset
import itertools
import numpy as np
from pyeda.inter import exprvars, And, Or, Xor, Implies, OneHot, Equal
from functools import reduce
from countrss_module import _pp_solution, _booldot

# Genera  input per 3 variabili
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
# Ma ricorda che XorDataset di suo genera gs e ys se chiami dataset.make_data().
# Puoi semplicemente assegnare i tuoi manualmente, se la classe non lo impedisce.
dataset.gvecs = np.zeros((len(gs), dataset.n_bits))
for i, g in enumerate(gs):
    # One-hot encoding manuale, ad es. per due valori per variabile
    # Ogni variabile ha due bit: la variabile j viene codificata come [1,0] se g[j] = 0 altrimenti [0,1]
    encoding = []
    for val in g:
        if val == 0:
            encoding.extend([1,0])
        else:
            encoding.extend([0,1])
    dataset.gvecs[i] = encoding

dataset.ys = ys

# Ora puoi utilizzare la stessa logica del main di gen-rss-count per costruire la formula ed enumerare i modelli.
# ...
# (Copia la parte relativa alla costruzione del CNF e al conteggio delle soluzioni)


print(f"Building formula: {len(dataset.gvecs)} gvecs, {dataset.n_bits} bits")

# generating the formula encoding the RSs
A = exprvars("A", dataset.n_bits, dataset.n_bits)
# print("A")
# print(A)
O = exprvars("O", dataset.n_variables, dataset.n_variables)
# print("A")
# print(O)

# A encodes a function C* -> C
# each C* index is mapped into exactly one C index
# although multiple C* indices can be mapped to the same C index
# (i.e. no OneHot on O's rows)
formula = And(*[OneHot(*O[:, k])
                for k in range(dataset.n_variables)])

# nzb(k1, k2) = the (k1,k2)-block in A is NON ZERO
nzb = lambda k1,k2 : Or(A[k1*2, k2*2], A[k1*2, k2*2 + 1],
                        A[k1*2 + 1, k2*2], A[k1*2 + 1, k2*2 + 1])

# zero-blocks A are zero and viceversa
formula &= And(*[Equal(O[k1, k2], nzb(k1, k2))
                    for k1 in range(dataset.n_variables)
                    for k2 in range(dataset.n_variables)])

formula &= And(*[OneHot(*A[:, i])
                for i in range(dataset.n_bits)])

# encode extra symbolic background
formula &= dataset.encode_background(A)



# force RSs to achieve perfect performance on data
for gvec, y in zip(dataset.gvecs, dataset.ys):
    cvec = [_booldot(A[i, :], gvec).simplify()
            for i in range(len(gvec))]
    "PRINTING CVEC TO CHECK"
    # print("gvec")
    # print(gvec)
    # print("cvec")
    # print(cvec)
    # cvec2 = [A[i, :] for i in range(len(gvec))]
    # print("cvec2")
    # print(cvec2)


    offset = 0
    for vsize in dataset.domain_sizes:
        formula &= OneHot(*cvec[offset:offset+vsize])
        offset += vsize
                    
    formula &= dataset.k(cvec, y)

n_sol = 0
for sol in formula.satisfy_all():
    _pp_solution(sol, dataset.n_variables, dataset.n_bits)
    n_sol += 1
print(f"{n_sol} solutions")