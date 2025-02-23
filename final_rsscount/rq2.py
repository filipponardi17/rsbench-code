import time
import itertools
import numpy as np
from functools import reduce
import csv
from countrss_module_approxmc_2 import ConfigurableXOR, count_rss
# Se necessario, importate anche le altre funzioni come _pp_solution, _booldot

def generate_xor_patterns(n_variables, dup_seed=None):
    """
    Genera tutti i pattern possibili per n_variables bit.
    Per n_variables=4, genera 16 pattern, poi ne seleziona casualmente 8 e li duplico,
    ottenendo un totale di 24 pattern.
    
    Restituisce:
      - gs_all: array di shape (total_patterns, n_variables) con i bit (0/1)
      - ys_all: array di shape (total_patterns,) con lo XOR dei bit di ogni pattern
      - gvecs_all: array codificato in forma one-hot (0 -> [1,0], 1 -> [0,1])
    """
    # Genera i 16 pattern originali
    original_gs = np.array(list(itertools.product([0,1], repeat=n_variables)))
    original_ys = np.array([reduce(lambda a, b: a ^ b, g, 0) for g in original_gs])
    
    # Codifica in one-hot
    original_gvecs = []
    for g in original_gs:
        enc = []
        for val in g:
            enc.extend([1, 0] if val == 0 else [0, 1])
        original_gvecs.append(enc)
    original_gvecs = np.array(original_gvecs)
    
    # Duplica 8 pattern scelti casualmente
    rng = np.random.RandomState(dup_seed)
    indices_to_duplicate = rng.choice(len(original_gs), size=len(original_gs)//2, replace=False)
    
    duplicate_gs = original_gs[indices_to_duplicate]
    duplicate_ys = original_ys[indices_to_duplicate]
    duplicate_gvecs = original_gvecs[indices_to_duplicate]
    
    # Combina gli array originali con quelli duplicati
    gs_all = np.concatenate([original_gs, duplicate_gs], axis=0)
    ys_all = np.concatenate([original_ys, duplicate_ys], axis=0)
    gvecs_all = np.concatenate([original_gvecs, duplicate_gvecs], axis=0)
    
    return gs_all, ys_all, gvecs_all


def define_random_containers(gs_all, seed=0):
    """
    Suddivide i pattern presenti in gs_all in contenitori da 2 task ciascuno, 
    scegliendoli in ordine randomico e senza ripetizioni.
    """
    rng = np.random.RandomState(seed)
    n_patterns = len(gs_all)  # ora potrebbe essere, ad esempio, 24
    all_indices = list(range(n_patterns))
    rng.shuffle(all_indices)  # mescola per avere ordine randomico

    containers = []
    # Assumiamo che n_patterns sia pari
    while all_indices:
        container = [all_indices.pop(), all_indices.pop()]
        containers.append(container)
    return containers


def rss_count_for_subset(gvecs_subset, ys_subset, n_variables):
    """
    Crea un oggetto ConfigurableXOR, carica i dati (pattern + label)
    e calcola la RSS (numero di soluzioni) con la funzione count_rss.
    """
    dataset = ConfigurableXOR(n_variables=n_variables)
    dataset.load_data(gvecs_subset, ys_subset)
    return count_rss(dataset)


def greedy_selection_containers(containers, gs_all, ys_all, gvecs_all, n_variables):
    """
    Selezione 'greedy': ad ogni step aggiunge il contenitore che minimizza la RSS risultante.
    Parte con un dataset vuoto (RSS calcolata subito come step 0).
    """
    MAX_SELECTIONS = 50

    T = list(range(len(containers)))  # Indici di tutti i contenitori disponibili
    S = []                            # Lista di contenitori scelti (indici in T)
    S_indices = []                    # Indici reali dei pattern raccolti in S
    rs_values = []                    # Valori RSS calcolati step-by-step
    step_count = 0

    # 1) Conta iniziale da dataset vuoto:
    rs_val_initial = rss_count_for_subset(gvecs_all[S_indices],
                                          ys_all[S_indices],
                                          n_variables)
    rs_values.append(rs_val_initial)

    # 2) Inizio selezione 'greedy'
    while len(T) > 0 and step_count < MAX_SELECTIONS:
        best_container = None
        best_rs = None
        
        # Trova il contenitore che, se aggiunto, dà l'RSS minore
        for c_idx in T:
            new_indices = S_indices + containers[c_idx]
            gsubset = gvecs_all[new_indices]
            ysubset = ys_all[new_indices]
            rs_val = rss_count_for_subset(gsubset, ysubset, n_variables)
            if best_rs is None or rs_val < best_rs:
                best_rs = rs_val
                best_container = c_idx

        # Aggiungi al set di selezionati
        S.append(best_container)
        S_indices.extend(containers[best_container])
        T.remove(best_container)

        # Calcola la RSS corrente
        rs_val = rss_count_for_subset(gvecs_all[S_indices],
                                      ys_all[S_indices],
                                      n_variables)
        rs_values.append(rs_val)

        step_count += 1

    return S, rs_values


def random_selection_containers(containers, gs_all, ys_all, gvecs_all, n_variables, seed=0):
    """
    Selezione 'random': ad ogni step sceglie un contenitore a caso.
    Parte con un dataset vuoto (RSS calcolata subito come step 0).
    """
    MAX_SELECTIONS = 50

    rng = np.random.RandomState(seed)
    T = list(range(len(containers)))  # Indici di tutti i contenitori disponibili
    S = []                            # Lista di contenitori scelti (indici in T)
    S_indices = []                    # Indici reali dei pattern raccolti in S
    rs_values = []                    # Valori RSS calcolati step-by-step
    step_count = 0

    # 1) Conta iniziale da dataset vuoto:
    rs_val_initial = rss_count_for_subset(gvecs_all[S_indices],
                                          ys_all[S_indices],
                                          n_variables)
    rs_values.append(rs_val_initial)

    # 2) Inizio selezione 'random'
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

    n_variables = 4
    # Passa un seed per la duplicazione se necessario (oppure None per randomizzare ogni volta)
    gs_all, ys_all, gvecs_all = generate_xor_patterns(n_variables, dup_seed=42)
    print(f"Generati {len(gs_all)} pattern (originali + duplicati)")

    # Eseguo 10 run
    for run_idx in range(1, 11):
        print(f"\n=== RUN {run_idx} ===")

        # Creazione dei contenitori: ogni contenitore contiene 2 task scelti casualmente dai {len(gs_all)} pattern
        containers = define_random_containers(gs_all, seed=run_idx)
        print("Contenitori generati (2 task ciascuno):", containers)
        print(f"Numero di contenitori totali: {len(containers)}")

        # ESEGUI GREEDY
        S_greedy, rs_vals_greedy = greedy_selection_containers(
            containers, gs_all, ys_all, gvecs_all, n_variables
        )

        # ESEGUI RANDOM
        S_random, rs_vals_random = random_selection_containers(
            containers, gs_all, ys_all, gvecs_all, n_variables, seed=42 + run_idx
        )

        # -----------------------------------------------------
        # Salvataggio dei CSV basato sulla lunghezza effettiva dei risultati
        # -----------------------------------------------------
        n_steps_rs = min(len(rs_vals_random), len(rs_vals_greedy))
        n_steps_sel = min(len(S_random), len(S_greedy))

        # 1) CSV con i valori RSS
        rs_filename = f"rq2_output_rs_values{run_idx}.csv"
        with open(rs_filename, 'w', newline='') as csvfile:
            fieldnames = ['step', 'rs_random', 'rs_greedy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            # La riga i=0 corrisponde allo step "dataset vuoto"
            for i in range(n_steps_rs):
                writer.writerow({
                    'step': i,
                    'rs_random': rs_vals_random[i],
                    'rs_greedy': rs_vals_greedy[i]
                })

        # 2) CSV con l'ordine di selezione dei contenitori e i pattern contenuti
        sel_filename = f"rq2_output_selection_order{run_idx}.csv"
        with open(sel_filename, 'w', newline='') as csvfile:
            fieldnames = [
                'step', 
                'selection_random_patterns_expanded',
                'selection_greedy_patterns_expanded'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(n_steps_sel):
                # Indice del contenitore scelto a questo step (random e greedy)
                chosen_random_idx = S_random[i]
                chosen_greedy_idx = S_greedy[i]
                
                # Pattern reali contenuti nei rispettivi contenitori (come indici)
                random_patterns_indices = containers[chosen_random_idx]
                greedy_patterns_indices = containers[chosen_greedy_idx]

                # Rappresentazione "espansa" in binario, es: "0,1,0,1;1,0,1,0;..."
                random_patterns_expanded_list = []
                for idx_pattern in random_patterns_indices:
                    pattern = gs_all[idx_pattern]  # es: array([0, 1, 0, 1])
                    pattern_str = ",".join(str(x) for x in pattern)
                    random_patterns_expanded_list.append(pattern_str)
                random_patterns_expanded_str = ";".join(random_patterns_expanded_list)

                greedy_patterns_expanded_list = []
                for idx_pattern in greedy_patterns_indices:
                    pattern = gs_all[idx_pattern]
                    pattern_str = ",".join(str(x) for x in pattern)
                    greedy_patterns_expanded_list.append(pattern_str)
                greedy_patterns_expanded_str = ";".join(greedy_patterns_expanded_list)

                # Scrittura CSV riga per riga
                writer.writerow({
                    'step': i+1,  # step conteggiati da 1 a n_steps_sel
                    'selection_random_patterns_expanded': random_patterns_expanded_str,
                    'selection_greedy_patterns_expanded': greedy_patterns_expanded_str
                })

        print(f"  -> Salvati: {rs_filename} e {sel_filename}")

    end_global = time.time()
    print(f"\nTempo di esecuzione globale (s): {end_global - start_global:.3f}")