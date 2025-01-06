#FUNZIONA CON I TASK INCREMENTALI QUINDI SE INDICO NUM_TASKS = 4 PRENDE I PRIMI 4 TASK


import itertools
import numpy as np
from functools import reduce
from pyeda.inter import exprvars, And, Or, OneHot, Equal
from countrss_module import _pp_solution, _booldot, ConfigurableXOR, count_rss

def generate_xor_patterns(n_variables):
    """
    Genera tutti i pattern XOR su n_variables variabili.
    Ritorna (gs_list, ys_list, gvecs_list) dove:
    - gs_list è la lista di pattern binari (una lista di array, ciascuno shape (n_variables,))
    - ys_list è la lista delle corrispondenti etichette XOR
    - gvecs_list è la lista dei pattern in one-hot encoding.
    """
    # Tutte le combinazioni possibili
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


def run_incremental_tasks(n_variables, num_tasks):
    """
    Crea num_tasks "task", ciascuno costituito da un singolo pattern XOR di n_variables.
    Al passo i (1 <= i <= num_tasks):
    - Usa i primi i pattern (task) combinati assieme
    - Carica questi i pattern nel ConfigurableXOR
    - Conta i RS.

    Assumiamo che il totale dei pattern possibili per n_variables sia >= num_tasks.
    Per n=3 variabili, abbiamo 8 pattern massimi. Quindi num_tasks dovrebbe essere <=8.
    """

    # Genera tutti i pattern
    gs_all, ys_all, gvecs_all = generate_xor_patterns(n_variables)

    # Controllo di sicurezza
    if num_tasks > len(gs_all):
        raise ValueError(f"num_tasks = {num_tasks} è maggiore del numero di pattern disponibili ({len(gs_all)}).")

    # Itera da 1 a num_tasks
    for i in range(1, num_tasks+1):
        # Combiniamo i primi i pattern
        gvecs_subset = gvecs_all[:i]
        ys_subset = ys_all[:i]
        print(f"Task {i}: {len(gvecs_subset)} pattern")
        print("gvecs_subset", gvecs_subset)

        # Crea un dataset configurabile e carica questi dati
        dataset = ConfigurableXOR(n_variables=n_variables)
        dataset.load_data(gvecs_subset, ys_subset)

        print(f"### Counting RS for first {i} task(s) ###")
        count_rss(dataset)


if __name__ == "__main__":
    n_variables = 4
    num_tasks = 4  # vogliamo considerare da 1 a 8 task incrementali

    run_incremental_tasks(n_variables, num_tasks)