import time
import itertools
import numpy as np
from functools import reduce
import csv

from countrss_module_approxmc_2 import ConfigurableXOR, count_rss
# Se necessario, importate anche le altre funzioni come _pp_solution, _booldot

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

def define_random_containers(n_variables, seed=0, min_size=1, max_size=3):
    rng = np.random.RandomState(seed)
    n_patterns = 2 ** n_variables
    all_indices = list(range(n_patterns))

    containers = []
    while all_indices:
        size = rng.randint(min_size, max_size+1)
        size = min(size, len(all_indices))

        chosen = rng.choice(all_indices, size=size, replace=False)
        chosen = sorted(chosen)
        containers.append(chosen)

        for c in chosen:
            all_indices.remove(c)

    return containers

def rss_count_for_subset(gvecs_subset, ys_subset, n_variables):
    dataset = ConfigurableXOR(n_variables=n_variables)
    dataset.load_data(gvecs_subset, ys_subset)
    return count_rss(dataset)

def greedy_selection_containers(containers, gs_all, ys_all, gvecs_all, n_variables):
    MAX_SELECTIONS = 10

    T = list(range(len(containers)))
    S = []
    S_indices = []
    rs_values = []
    step_count = 0
    
    while len(T) > 0 and step_count < MAX_SELECTIONS:
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

        S.append(best_container)
        S_indices.extend(containers[best_container])
        T.remove(best_container)

        rs_val = rss_count_for_subset(gvecs_all[S_indices],
                                      ys_all[S_indices],
                                      n_variables)
        rs_values.append(rs_val)

        step_count += 1

    return S, rs_values

def random_selection_containers(containers, gs_all, ys_all, gvecs_all, n_variables, seed=0):
    MAX_SELECTIONS = 10

    rng = np.random.RandomState(seed)
    T = list(range(len(containers)))
    S = []
    S_indices = []
    rs_values = []
    step_count = 0
    
    while len(T) > 0 and step_count < MAX_SELECTIONS:
        c_random_index = rng.randint(0, len(T))
        chosen_container = T[c_random_index]

        S.append(chosen_container)
        S_indices.extend(containers[chosen_container])
        T.remove(chosen_container)

        rs_val = rss_count_for_subset(gvecs_all[S_indices],
                                      ys_all[S_indices],
                                      n_variables)
        rs_values.append(rs_val)

        step_count += 1

    return S, rs_values


if __name__ == "__main__":
    start_global = time.time()

    n_variables = 5
    gs_all, ys_all, gvecs_all = generate_xor_patterns(n_variables)

    for run_idx in range(1, 21):
        print(f"\n=== RUN {run_idx} ===")

        containers = define_random_containers(n_variables, seed=run_idx, min_size=1, max_size=3)
        pretty_containers = [list(map(int, c)) for c in containers]
        print("Contenitori generati (random):", pretty_containers)

        # Replichiamo ciascun contenitore 4 volte
        extended_containers = []
        for c in containers:
            for _ in range(10):
                extended_containers.append(c)
        containers = extended_containers
        print(f"Numero di contenitori totali dopo la replica: {len(containers)}")

        # ESEGUI GREEDY (max 15 step)
        S_greedy, rs_vals_greedy = greedy_selection_containers(
            containers, gs_all, ys_all, gvecs_all, n_variables
        )

        # ESEGUI RANDOM (max 15 step)
        S_random, rs_vals_random = random_selection_containers(
            containers, gs_all, ys_all, gvecs_all, n_variables, seed=42 + run_idx
        )

        # -----------------------------------------------------
        # Per salvare i CSV, usiamo la lunghezza effettiva dei risultati
        # -----------------------------------------------------
        n_steps_rs = min(len(rs_vals_random), len(rs_vals_greedy))
        n_steps_sel = min(len(S_random), len(S_greedy))

        # 1) CSV con i valori RS
        rs_filename = f"output_rs_values{run_idx}.csv"
        with open(rs_filename, 'w', newline='') as csvfile:
            fieldnames = ['step', 'rs_random', 'rs_greedy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(n_steps_rs):
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
            for i in range(n_steps_sel):
                writer.writerow({
                    'step': i + 1,
                    'selection_random': S_random[i],
                    'selection_greedy': S_greedy[i]
                })

        print(f"  -> Salvati: {rs_filename} e {sel_filename}")

    end_global = time.time()
    print(f"\nTempo di esecuzione globale (s): {end_global - start_global:.3f}")