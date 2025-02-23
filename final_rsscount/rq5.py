import time
import itertools
import numpy as np
from functools import reduce
import csv
from countrss_module_approxmc_2 import ConfigurableXOR, count_rss

# =============================================================================
# Generazione dei pattern XOR
# =============================================================================
def generate_xor_patterns(n_variables, duplication_count=0, dup_seed=None):
    """
    Genera i 2^n pattern per n_variables.
    Se duplication_count > 0, duplica duplication_count pattern scelti casualmente.
    
    Restituisce:
      - gs_all: array con i pattern (con duplicazioni se previste)
      - ys_all: array con lo XOR dei bit di ogni pattern
      - gvecs_all: array in codifica one-hot (0 -> [1,0], 1 -> [0,1])
    """
    # Genera i pattern base (2^n)
    gs_all = np.array(list(itertools.product([0, 1], repeat=n_variables)))
    
    # Duplica duplication_count pattern scelti casualmente se richiesto
    if duplication_count > 0 and dup_seed is not None:
        rng = np.random.RandomState(dup_seed)
        indices_to_duplicate = rng.choice(len(gs_all), size=duplication_count, replace=False)
        duplicate_gs = gs_all[indices_to_duplicate]
        gs_all = np.concatenate([gs_all, duplicate_gs], axis=0)
    
    # Calcola lo XOR per ogni pattern
    ys_all = np.array([reduce(lambda a, b: a ^ b, g, 0) for g in gs_all])
    
    # Codifica in one-hot: 0 -> [1,0] e 1 -> [0,1]
    gvecs_all = []
    for g in gs_all:
        enc = []
        for val in g:
            enc.extend([1, 0] if val == 0 else [0, 1])
        gvecs_all.append(enc)
    gvecs_all = np.array(gvecs_all)
    
    return gs_all, ys_all, gvecs_all

# =============================================================================
# Definizione dei contenitori (senza ripetizioni) in maniera randomica
# =============================================================================
def define_random_containers(n_variables, seed=0, min_size=1, max_size=3):
    """
    Suddivide i pattern possibili (2^n_variables) in una serie di 'contenitori' random.
    Ogni contenitore è una lista di indici dei pattern (senza ripetizioni).
    """
    rng = np.random.RandomState(seed)
    n_patterns = 2 ** n_variables
    all_indices = list(range(n_patterns))
    
    containers = []
    while all_indices:
        size = rng.randint(min_size, max_size + 1)
        size = min(size, len(all_indices))
        
        chosen = rng.choice(all_indices, size=size, replace=False)
        chosen = sorted(chosen)
        containers.append(chosen)
        
        for c in chosen:
            all_indices.remove(c)
    
    return containers

# =============================================================================
# Clonazione dei contenitori
# =============================================================================
def clone_containers(containers, clone_factor=10):
    """
    Replica ciascun contenitore clone_factor volte.
    """
    extended_containers = []
    for c in containers:
        for _ in range(clone_factor):
            extended_containers.append(c)
    return extended_containers

# =============================================================================
# Calcolo della RSS per un sottoinsieme di pattern
# =============================================================================
def rss_count_for_subset(gvecs_subset, ys_subset, n_variables):
    dataset = ConfigurableXOR(n_variables=n_variables)
    dataset.load_data(gvecs_subset, ys_subset)
    return count_rss(dataset)

# =============================================================================
# Strategie di selezione: Greedy e Random
# =============================================================================
def greedy_selection_containers(containers, gs_all, ys_all, gvecs_all, n_variables, max_selections=50):
    T = list(range(len(containers)))  # Indici dei contenitori disponibili
    S = []                            # Indici dei contenitori selezionati
    S_indices = []                    # Indici dei pattern raccolti
    rs_values = []
    rs_values.append(rss_count_for_subset(gvecs_all[S_indices], ys_all[S_indices], n_variables))
    step_count = 0

    while T and step_count < max_selections:
        best_container = None
        best_rs = None
        
        # Valuta l'aggiunta di ciascun contenitore
        for c_idx in T:
            new_indices = S_indices + containers[c_idx]
            rs_val = rss_count_for_subset(gvecs_all[new_indices], ys_all[new_indices], n_variables)
            if best_rs is None or rs_val < best_rs:
                best_rs = rs_val
                best_container = c_idx
        
        S.append(best_container)
        S_indices.extend(containers[best_container])
        T.remove(best_container)
        rs_values.append(rss_count_for_subset(gvecs_all[S_indices], ys_all[S_indices], n_variables))
        step_count += 1

    return S, rs_values

def random_selection_containers(containers, gs_all, ys_all, gvecs_all, n_variables, seed=0, max_selections=50):
    rng = np.random.RandomState(seed)
    T = list(range(len(containers)))
    S = []
    S_indices = []
    rs_values = []
    rs_values.append(rss_count_for_subset(gvecs_all[S_indices], ys_all[S_indices], n_variables))
    step_count = 0

    while T and step_count < max_selections:
        chosen_idx = int(rng.choice(T))
        S.append(chosen_idx)
        S_indices.extend(containers[chosen_idx])
        T.remove(chosen_idx)
        rs_values.append(rss_count_for_subset(gvecs_all[S_indices], ys_all[S_indices], n_variables))
        step_count += 1

    return S, rs_values

# =============================================================================
# MAIN: combinazione delle casistiche secondo le specifiche
# =============================================================================
if __name__ == "__main__":
    start_global = time.time()

    n_variables = 4
    # Parametri per la duplicazione:
    duplication_count = 8      # Numero di pattern da duplicare (es. 8)
    base_dup_seed = 1234       # Seme per la duplicazione

    # Parametri per i contenitori:
    min_size = 1
    max_size = 3
    clone_factor = 10          # Quante volte clonare ciascun contenitore

    # Parametro base per i run:
    base_seed = 1234
    n_runs = 10

    # Genera i pattern: se duplication_count > 0, verranno duplicati duplication_count pattern
    gs_all, ys_all, gvecs_all = generate_xor_patterns(n_variables,
                                                       duplication_count=duplication_count,
                                                       dup_seed=base_dup_seed)
    print(f"Generati {len(gs_all)} pattern (inclusi {duplication_count} duplicati)")

    # Esecuzione di più run
    for run_idx in range(1, n_runs + 1):
        current_seed = base_seed + run_idx
        print(f"\n=== RUN {run_idx} (seed={current_seed}) ===")
        
        # Costruzione dei contenitori (senza specificare il numero, viene calcolato in base ai pattern disponibili)
        containers = define_random_containers(n_variables, seed=current_seed, min_size=min_size, max_size=max_size)
        print("Contenitori generati:", containers)
        
        # Clonazione dei contenitori
        containers = clone_containers(containers, clone_factor=clone_factor)
        print(f"Numero di contenitori totali dopo la clonazione: {len(containers)}")
        
        # Selezione Greedy e Random
        S_greedy, rs_vals_greedy = greedy_selection_containers(containers, gs_all, ys_all, gvecs_all, n_variables)
        S_random, rs_vals_random = random_selection_containers(containers, gs_all, ys_all, gvecs_all, n_variables, seed=current_seed)
        
        # Salvataggio dei risultati in CSV
        n_steps_rs = min(len(rs_vals_greedy), len(rs_vals_random))
        n_steps_sel = min(len(S_greedy), len(S_random))
        
        rs_filename = f"output_rs_values_run{run_idx}.csv"
        with open(rs_filename, 'w', newline='') as csvfile:
            fieldnames = ['step', 'rs_random', 'rs_greedy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(n_steps_rs):
                writer.writerow({
                    'step': i,
                    'rs_random': rs_vals_random[i],
                    'rs_greedy': rs_vals_greedy[i]
                })
        
        sel_filename = f"output_selection_order_run{run_idx}.csv"
        with open(sel_filename, 'w', newline='') as csvfile:
            fieldnames = ['step', 'selection_random_patterns_expanded', 'selection_greedy_patterns_expanded']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(n_steps_sel):
                # Ottiene gli indici dei pattern nel contenitore scelto ad ogni step
                random_patterns_indices = containers[S_random[i]]
                greedy_patterns_indices = containers[S_greedy[i]]
                # Conversione in stringa (es: "0,1,0,1;1,0,1,0;...")
                random_patterns_expanded = ";".join([",".join(str(x) for x in gs_all[idx]) for idx in random_patterns_indices])
                greedy_patterns_expanded = ";".join([",".join(str(x) for x in gs_all[idx]) for idx in greedy_patterns_indices])
                writer.writerow({
                    'step': i + 1,
                    'selection_random_patterns_expanded': random_patterns_expanded,
                    'selection_greedy_patterns_expanded': greedy_patterns_expanded
                })
        print(f"  -> Salvati: {rs_filename} e {sel_filename}")
    
    print(f"\nTempo di esecuzione globale (s): {time.time()-start_global:.3f}")