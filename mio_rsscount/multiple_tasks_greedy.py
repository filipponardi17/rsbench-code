#PROVA DI GREEDY ALGORITHM

import itertools
import numpy as np
from functools import reduce
from countrss_module import _pp_solution, _booldot, ConfigurableXOR, count_rss

def generate_xor_patterns(n_variables):
    gs_all = np.array(list(itertools.product([0,1], repeat=n_variables)))
    ys_all = np.array([reduce(lambda a,b: a^b, g, 0) for g in gs_all])

    gvecs_all = []
    for g in gs_all:
        enc = []
        for val in g:
            enc.extend([1,0] if val == 0 else [0,1])
        gvecs_all.append(enc)
    gvecs_all = np.array(gvecs_all)

    return gs_all, ys_all, gvecs_all

def rss_count_for_subset(gvecs_subset, ys_subset, n_variables):
    """Ritorna il numero di RS per il subset di gvec specificati."""
    dataset = ConfigurableXOR(n_variables=n_variables)
    dataset.load_data(gvecs_subset, ys_subset)
    return count_rss(dataset)

def greedy_selection(n_variables, final_size=4):
    # Genera tutti i pattern
    gs_all, ys_all, gvecs_all = generate_xor_patterns(n_variables)

    # Inizialmente:
    # T contiene tutti i pattern
    # S è vuoto
    T = list(range(len(gvecs_all)))  # Indici dei pattern
    #print(f"{len(T)} solutions for this task")
    #print("T", T)
    S = []

    # 1: troviamo il singolo gvec che minimizza i RS da solo
    best_g = None
    best_rs = None
    for i in T:
        gsubset = gvecs_all[[i]]  # subset con un solo task
        ysubset = ys_all[[i]]
        rs_val = rss_count_for_subset(gsubset, ysubset, n_variables)
        if best_rs is None or rs_val < best_rs:
            best_rs = rs_val
            best_g = i

    # 2: Aggiungiamo questo task a S e rimuovilo da T
    S.append(best_g)
    T.remove(best_g)

    # 3: Ora len(S)=1. Dobbiamo aggiungerne altri finché non ne abbiamo final_size.
    while len(S) < final_size:
        # 4: Proviamo ciascun task in T, aggiungendolo temporaneamente a S e vediamo quante RS otteniamo
        candidate = None
        candidate_rs = None

        for task in T:
            new_subset = S + [task]
            gsubset = gvecs_all[new_subset]
            ysubset = ys_all[new_subset]
            rs_val = rss_count_for_subset(gsubset, ysubset, n_variables)
            if candidate_rs is None or rs_val < candidate_rs:
                candidate_rs = rs_val
                candidate = task

        # 5 :Aggiungiamo il task trovato permanentemente a S
        S.append(candidate)
        T.remove(candidate)
    #6: Ritorna S
    return S

if __name__ == "__main__":
    n_variables = 4
    S = greedy_selection(n_variables, final_size=4)
    print("I pattern selezionati sono (indici):", S)


    """
    final_size=4
24 solutions for this task
I pattern selezionati sono (indici): [0, 1, 2, 4]
"""