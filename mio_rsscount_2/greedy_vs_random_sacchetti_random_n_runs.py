import time
import itertools
import numpy as np
from functools import reduce
import csv

from countrss_module_approxmc_2 import ConfigurableXOR, count_rss
# Se necessario, importate anche le altre funzioni come _pp_solution, _booldot

def generate_xor_patterns(n_variables):
    """
    Genera tutti i pattern dell'XOR a n variabili.
    Restituisce:
    - gs_all: shape (2^n, n)
    - ys_all: shape (2^n,) con il bit di output XOR
    - gvecs_all: shape (2^n, 2*n) con la codifica 1,0 / 0,1
    """
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

def define_random_containers(n_variables, seed=0, min_size=1, max_size=3):
    """
    Genera una partizione casuale degli indici [0..(2^n - 1)] in blocchi
    di grandezza compresa tra min_size e max_size.
    Esempio: n_variables=4 -> 16 pattern totali -> Contenitori random.
    """
    rng = np.random.RandomState(seed)
    n_patterns = 2 ** n_variables  # Es: se n_variables=4, n_patterns=16
    all_indices = list(range(n_patterns))  # [0, 1, ..., 15]

    containers = []
    while all_indices:
        # Scegliamo casualmente la dimensione del contenitore
        size = rng.randint(min_size, max_size+1)
        size = min(size, len(all_indices))  # Non eccedere il residuo

        # Pesco 'size' indici a caso
        chosen = rng.choice(all_indices, size=size, replace=False)
        chosen = sorted(chosen)  # facoltativo
        containers.append(chosen)

        # Rimuovo questi indici da all_indices
        for c in chosen:
            all_indices.remove(c)

    return containers

def rss_count_for_subset(gvecs_subset, ys_subset, n_variables):
    """
    Ritorna il numero di RS (soluzioni) per il subset di gvec specificati.
    """
    dataset = ConfigurableXOR(n_variables=n_variables)
    dataset.load_data(gvecs_subset, ys_subset)
    return count_rss(dataset)

def greedy_selection_containers(containers, gs_all, ys_all, gvecs_all, n_variables):
    """
    Algoritmo greedy a livello di 'contenitori':
    - A ogni iterazione, individua il contenitore che minimizza la RS (se aggiunto).
    - Lo aggiunge e rimuove da T finché non li ha presi tutti.
    
    Ritorna (S, rs_values):
    - S: lista degli indici dei contenitori nell'ordine di selezione
    - rs_values: lista dei valori di RS dopo ogni aggiunta
    """
    T = list(range(len(containers)))  # Indici dei contenitori
    S = []
    S_indices = []
    rs_values = []

    while len(T) > 0:
        best_container = None
        best_rs = None
        
        for c_idx in T:
            new_indices = S_indices + containers[c_idx]
            gsubset = gvecs_all[new_indices]
            ysubset = ys_all[new_indices]
            rs_val = rss_count_for_subset(gsubset, ysubset, n_variables)
            if best_rs is None or rs_val < best_rs:
                best_rs = rs_val
                best_container = c_idx

        # Aggiungo il container migliore
        S.append(best_container)
        S_indices.extend(containers[best_container])
        T.remove(best_container)

        # Calcolo la RS aggiornata
        gsubset = gvecs_all[S_indices]
        ysubset = ys_all[S_indices]
        rs_val = rss_count_for_subset(gsubset, ysubset, n_variables)
        rs_values.append(rs_val)

    return S, rs_values

def random_selection_containers(containers, gs_all, ys_all, gvecs_all, n_variables, seed=0):
    """
    Algoritmo random a livello di 'contenitori':
    - A ogni iterazione, scelgo a caso uno dei contenitori rimanenti,
      lo aggiungo e lo rimuovo da T.
    - Calcolo la RS. Ripeto finchè non li ho presi tutti.
    """
    rng = np.random.RandomState(seed)
    T = list(range(len(containers)))
    S = []
    S_indices = []
    rs_values = []

    while len(T) > 0:
        c_random_index = rng.randint(0, len(T))
        chosen_container = T[c_random_index]

        S.append(chosen_container)
        S_indices.extend(containers[chosen_container])
        T.remove(chosen_container)

        # RS su S_indices
        gsubset = gvecs_all[S_indices]
        ysubset = ys_all[S_indices]
        rs_val = rss_count_for_subset(gsubset, ysubset, n_variables)
        rs_values.append(rs_val)

    return S, rs_values


if __name__ == "__main__":
    start_global = time.time()

    n_variables = 5
    # Generiamo una volta tutti i pattern (gs_all, ys_all, gvecs_all).
    gs_all, ys_all, gvecs_all = generate_xor_patterns(n_variables)

    # Faremo 5 run. In ognuna:
    #  - generiamo contenitori random con un seed diverso
    #  - eseguiamo Greedy
    #  - eseguiamo Random
    #  - salviamo i risultati su due CSV: output_rs_values{run}.csv e output_selection_order{run}.csv

    for run_idx in range(1, 21):
        print(f"\n=== RUN {run_idx} ===")

        # Generiamo i contenitori casuali (seed diverso a ogni run)
        # Se vuoi puoi cambiare la formula, es: seed=1000 + run_idx
        containers = define_random_containers(n_variables, seed=run_idx, min_size=1, max_size=3)
        # Per bellezza stampiamo i contenitori con int "puro" (senza np.int64)
        pretty_containers = [list(map(int, c)) for c in containers]
        print("Contenitori generati (random):", pretty_containers)


    #     # ESEGUI GREEDY
        S_greedy, rs_vals_greedy = greedy_selection_containers(
            containers, gs_all, ys_all, gvecs_all, n_variables
        )

        # ESEGUI RANDOM (usiamo un seme differente dal solito, 
        # altrimenti rifà la stessa sequenza)
        # se vuoi “fisso” per ogni run, puoi passare seed=run_idx, 
        # oppure un seme diverso.
        S_random, rs_vals_random = random_selection_containers(
            containers, gs_all, ys_all, gvecs_all, n_variables, seed=42+run_idx
        )

        # Scriviamo i CSV di questa run
        n_steps = len(containers)

        # 1) CSV con i valori RS
        rs_filename = f"output_rs_values{run_idx}.csv"
        with open(rs_filename, 'w', newline='') as csvfile:
            fieldnames = ['step', 'rs_random', 'rs_greedy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(n_steps):
                writer.writerow({
                    'step': i + 1,
                    'rs_random': rs_vals_random[i],
                    'rs_greedy': rs_vals_greedy[i]
                })

        # 2) CSV con l'ordine di selezione dei contenitori
        sel_filename = f"output_selection_order{run_idx}.csv"
        with open(sel_filename, 'w', newline='') as csvfile:
            fieldnames = ['step', 'selection_random', 'selection_greedy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(n_steps):
                writer.writerow({
                    'step': i + 1,
                    'selection_random': S_random[i],
                    'selection_greedy': S_greedy[i]
                })

        print(f"  -> Salvati: {rs_filename} e {sel_filename}")

    end_global = time.time()
    print(f"\nTempo di esecuzione globale (s): {end_global - start_global:.3f}")