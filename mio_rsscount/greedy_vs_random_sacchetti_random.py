import time
import itertools
import numpy as np
from functools import reduce
import csv

from countrss_module import ConfigurableXOR, count_rss
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
        size = min(size, len(all_indices))  # Non eccedere gli indici rimanenti

        # Pesco 'size' indici a caso da all_indices
        chosen = rng.choice(all_indices, size=size, replace=False)
        chosen = sorted(chosen)  # facoltativo, per ordine interno
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
    - Alla prima iterazione, trova il contenitore che minimizza la RS.
    - Aggiungilo alla soluzione. Ripeti finchè non li hai presi tutti.
    
    Ritorna (S, rs_values):
    - S: lista degli indici dei contenitori nell'ordine di selezione
    - rs_values: lista dei valori RS dopo ogni aggiunta
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

        # Aggiungiamo il container migliore
        S.append(best_container)
        S_indices.extend(containers[best_container])
        T.remove(best_container)

        # Calcolo RS aggiornato
        gsubset = gvecs_all[S_indices]
        ysubset = ys_all[S_indices]
        rs_val = rss_count_for_subset(gsubset, ysubset, n_variables)
        rs_values.append(rs_val)

    return S, rs_values

def random_selection_containers(containers, gs_all, ys_all, gvecs_all, n_variables, seed=0):
    """
    Algoritmo random a livello di 'contenitori':
    - A ogni iterazione, scelgo a caso uno dei contenitori rimanenti,
      lo aggiungo e rimuovo dal set T.
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
    start = time.time()

    # Scegli il numero di variabili che vuoi
    n_variables = 4

    # Generiamo tutti i pattern
    gs_all, ys_all, gvecs_all = generate_xor_patterns(n_variables)

    # Generiamo i contenitori (in modo RANDOM) con dimensioni tra 1 e 3
    containers = define_random_containers(n_variables, seed=123, min_size=1, max_size=3)
    pretty_containers = [list(map(int, c)) for c in containers]
    print("Contenitori generati (random):", pretty_containers)


    #GREEDY
    S_greedy, rs_vals_greedy = greedy_selection_containers(
        containers, gs_all, ys_all, gvecs_all, n_variables
    )

    # RANDOM
    S_random, rs_vals_random = random_selection_containers(
        containers, gs_all, ys_all, gvecs_all, n_variables, seed=123
    )

    # Salvataggio CSV dei valori RS
    with open('output_rs_values.csv', 'w', newline='') as csvfile:
        fieldnames = ['step', 'rs_random', 'rs_greedy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # In genere avremo len(containers) step per entrambi
        n_steps = len(containers)
        for i in range(n_steps):
            writer.writerow({
                'step': i + 1,
                'rs_random': rs_vals_random[i],
                'rs_greedy': rs_vals_greedy[i]
            })
    
    # Salvataggio CSV dell'ordine di selezione dei contenitori
    with open('output_selection_order.csv', 'w', newline='') as csvfile:
        fieldnames = ['step', 'selection_random', 'selection_greedy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_steps):
            writer.writerow({
                'step': i + 1,
                'selection_random': S_random[i],
                'selection_greedy': S_greedy[i]
            })

    end = time.time()
    print("\nTempo di esecuzione (s):", end - start)