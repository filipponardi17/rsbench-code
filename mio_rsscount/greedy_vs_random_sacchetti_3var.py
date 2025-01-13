#HARD CODED PER 4 VAR IN MODO CHE ABBIANO QUESTE GRANDEZZE : 
# 1
# 2
# 3
# 4
# 1
# 2
# 3

import time
import itertools
import numpy as np
from functools import reduce
import csv  # Per salvare i risultati

from countrss_module import ConfigurableXOR, count_rss
# Se necessario, importate anche le altre funzioni come _pp_solution, _booldot

def generate_xor_patterns_3vars():
    """
    Genera TUTTI i 16 pattern dell'XOR a 4 variabili, 
    restituendo (gs_all, ys_all, gvecs_all).
    """
    n_variables = 3
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

def define_containers_for_3_variables():
    """
    Definisce i 'contenitori' (hard-coded) per i 8 pattern delle 4 variabili.
    Ogni contenitore è una lista di indici (pattern) da selezionare IN BLOCCO.
    
    Esempio con 5 contenitori di grandezza 1,2,2,1,2:
       - C0 -> [0]
       - C1 -> [1,2]
       - C2 -> [3,4]
       - C3 -> [5]      
       - C4 -> [6,7]
    """
    containers = [
        [0],            # 1 pattern
        [1, 2],         # 2 pattern
        [3, 4],         # 2 pattern
        [5],             # 1 pattern
        [6, 7]         # 2 pattern
    ]
    return containers

def rss_count_for_subset(gvecs_subset, ys_subset, n_variables=3):
    """Ritorna il numero di RS per il subset di gvec specificati."""
    dataset = ConfigurableXOR(n_variables=n_variables)
    dataset.load_data(gvecs_subset, ys_subset)
    return count_rss(dataset)

def greedy_selection_containers(seed=0):
    """
    Algoritmo greedy a livello di 'contenitori':
    - Abbiamo 7 contenitori (hard-coded).
    - A ogni iterazione selezioniamo il contenitore che,
      aggiunto a S_indices, minimizza il conteggio RS.
    - Proseguiamo finché non abbiamo selezionato tutti i contenitori.

    Restituisce:
    - S: indice dei contenitori nell'ordine in cui sono stati scelti
    - rs_values: lista dei valori di RS dopo ogni aggiunta di contenitore
    """
    np.random.seed(seed)

    # Carichiamo tutti i pattern XOR a 4 variabili
    gs_all, ys_all, gvecs_all = generate_xor_patterns_3vars()
    # Definiamo i contenitori
    containers = define_containers_for_3_variables()

    # Questi sono gli indici dei contenitori ancora selezionabili
    T = list(range(len(containers)))  # [0, 1, 2, 3, 4, 5, 6]

    # S tiene traccia di quali contenitori abbiamo scelto (in ordine)
    S = []
    # S_indices tiene traccia dell’insieme di *pattern* effettivi selezionati finora
    S_indices = []
    
    # Lista per memorizzare i valori di RS a ogni aggiunta
    rs_values = []

    while len(T) > 0:
        best_container = None
        best_rs = None
        
        for c_idx in T:
            # Se aggiungiamo il contenitore c_idx, i pattern totali sarebbero:
            new_indices = S_indices + containers[c_idx]
            # Creiamo i subset corrispondenti
            gsubset = gvecs_all[new_indices]
            ysubset = ys_all[new_indices]
            rs_val = rss_count_for_subset(gsubset, ysubset, n_variables=3)
            
            # Verifichiamo se è il migliore (cioè se la RS si abbassa di più)
            if best_rs is None or rs_val < best_rs:
                best_rs = rs_val
                best_container = c_idx

        # Una volta trovato il container migliore, lo aggiungiamo
        S.append(best_container)
        S_indices.extend(containers[best_container])
        # Rimuoviamo il container scelto da T
        T.remove(best_container)
        
        # Calcoliamo l'RS con l’insieme S_indices aggiornato
        gsubset = gvecs_all[S_indices]
        ysubset = ys_all[S_indices]
        rs_val = rss_count_for_subset(gsubset, ysubset, n_variables=3)
        rs_values.append(rs_val)

    return S, rs_values

def random_selection_containers(seed=0):
    """
    Algoritmo random a livello di 'contenitori':
    - A ogni iterazione, seleziona un contenitore a caso da T,
      lo aggiunge a S_indices, calcola la RS e lo rimuove da T.
    - Prosegue finché non abbiamo scelto tutti i contenitori.

    Restituisce:
    - S: indice dei contenitori nell'ordine in cui sono stati scelti
    - rs_values: lista dei valori di RS dopo ogni aggiunta di contenitore
    """
    np.random.seed(seed)

    # Carichiamo i 16 pattern dell'XOR a 4 variabili
    gs_all, ys_all, gvecs_all = generate_xor_patterns_3vars()
    # Definiamo i contenitori
    containers = define_containers_for_3_variables()

    # Indici dei contenitori ancora selezionabili
    T = list(range(len(containers)))  # [0, 1, 2, 3, 4, 5]
    
    S = []          # contiene l'ordine in cui scegliamo i contenitori
    S_indices = []  # tutti i pattern selezionati finora
    rs_values = []

    while len(T) > 0:
        # Scegliamo un container a caso
        c_random_index = np.random.randint(0, len(T))
        chosen_container = T[c_random_index]
        
        # Aggiungiamo i pattern del contenitore scelto
        S.append(chosen_container)
        S_indices.extend(containers[chosen_container])
        # Rimuoviamo quel contenitore da T
        T.remove(chosen_container)

        # Calcoliamo l'RS con la nuova S_indices
        gsubset = gvecs_all[S_indices]
        ysubset = ys_all[S_indices]
        rs_val = rss_count_for_subset(gsubset, ysubset, n_variables=3)
        rs_values.append(rs_val)

    return S, rs_values

if __name__ == "__main__":
    start = time.time()

    # ESEMPIO: eseguiamo su 4 variabili (16 pattern) in 7 'contenitori'
    # GREEDY
    S_greedy, rs_vals_greedy = greedy_selection_containers(seed=42)

    # RANDOM
    S_random, rs_vals_random = random_selection_containers(seed=42)

    # Salvataggio CSV (unico file per i valori di RS)
    with open('output_rs_values.csv', 'w', newline='') as csvfile:
        fieldnames = ['step', 'rs_random', 'rs_greedy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Qui assumiamo che random e greedy abbiano la stessa lunghezza (7 step).
        # Se i contenitori sono 7, ci saranno 7 step in entrambi i casi.
        for i in range(len(rs_vals_random)):
            writer.writerow({
                'step': i + 1,
                'rs_random': rs_vals_random[i],
                'rs_greedy': rs_vals_greedy[i]
            })
    
    # (Facoltativo) salvare anche l'ordine di selezione di ciascun container
    with open('output_selection_order.csv', 'w', newline='') as csvfile:
        fieldnames = ['step', 'selection_random', 'selection_greedy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(S_random)):
            writer.writerow({
                'step': i + 1,
                'selection_random': S_random[i],
                'selection_greedy': S_greedy[i]
            })

    end = time.time()
    print("\nTempo di esecuzione (s):", end - start)