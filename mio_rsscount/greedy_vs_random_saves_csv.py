import time
import itertools
import numpy as np
from functools import reduce
import csv  # <-- Importiamo il modulo csv (o puoi usare Pandas se preferisci)

from countrss_module import ConfigurableXOR, count_rss
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

def rss_count_for_subset(gvecs_subset, ys_subset, n_variables):
    """Ritorna il numero di RS per il subset di gvec specificati."""
    dataset = ConfigurableXOR(n_variables=n_variables)
    dataset.load_data(gvecs_subset, ys_subset)
    return count_rss(dataset)

def greedy_selection(n_variables, seed=0):
    """
    Algoritmo greedy:
    - Finché T non è vuoto, cerca l'elemento di T che, aggiunto a S, minimizza il conteggio RS.
    - Aggiunge quell'elemento a S, lo rimuove da T.
    - Salva il conteggio RS con la S aggiornata.
    
    Restituisce:
    - S: la lista degli indici selezionati (in ordine di selezione)
    - rs_values: la lista dei valori di RS calcolati a ogni passo
    """
    np.random.seed(seed)  # In un classico greedy non serve, ma lo mettiamo per coerenza

    # Genera tutti i pattern
    gs_all, ys_all, gvecs_all = generate_xor_patterns(n_variables)
    T = list(range(len(gvecs_all)))  # Indici dei pattern
    S = []
    
    # Lista per memorizzare i valori di RS a ogni aggiunta
    rs_values = []

    while len(T) > 0:
        candidate = None
        candidate_rs = None
        
        # Trova il task che, aggiunto a S, minimizza RS
        for task in T:
            new_subset = S + [task]
            gsubset = gvecs_all[new_subset]
            ysubset = ys_all[new_subset]
            rs_val = rss_count_for_subset(gsubset, ysubset, n_variables)
            
            if candidate_rs is None or rs_val < candidate_rs:
                candidate_rs = rs_val
                candidate = task

        # Aggiungiamo il 'candidate' a S
        S.append(candidate)
        T.remove(candidate)
        
        # Calcoliamo l'RS di S aggiornata
        gsubset = gvecs_all[S]
        ysubset = ys_all[S]
        rs_val = rss_count_for_subset(gsubset, ysubset, n_variables)
        rs_values.append(rs_val)

    return S, rs_values

def random_selection(n_variables, seed=0):
    """
    Algoritmo random:
    - Finché T non è vuoto, prende casualmente un elemento da T, lo aggiunge a S e lo rimuove da T.
    - Calcola e salva il conteggio RS di volta in volta per la nuova S.

    Restituisce:
    - S: la lista degli indici selezionati (in ordine di selezione)
    - rs_values: la lista dei valori di RS calcolati a ogni passo
    """
    np.random.seed(seed)
    
    # Genera tutti i pattern
    gs_all, ys_all, gvecs_all = generate_xor_patterns(n_variables)
    T = list(range(len(gvecs_all)))  # Indici dei pattern
    S = []
    
    rs_values = []

    while len(T) > 0:
        # Scegliamo un indice a caso
        random_index = np.random.randint(0, len(T))
        candidate = T[random_index]

        # Aggiungiamo il candidato a S e lo rimuoviamo da T
        S.append(candidate)
        T.remove(candidate)

        # Calcoliamo l'RS con la nuova S
        gsubset = gvecs_all[S]
        ysubset = ys_all[S]
        rs_val = rss_count_for_subset(gsubset, ysubset, n_variables)
        rs_values.append(rs_val)

    return S, rs_values

if __name__ == "__main__":
    start = time.time()  # Avvio timer

    n_variables = 3
    
    # ESEMPIO ESECUZIONE GREEDY
    S_greedy, rs_vals_greedy = greedy_selection(n_variables, seed=42)

    # ESEMPIO ESECUZIONE RANDOM
    S_random, rs_vals_random = random_selection(n_variables, seed=42)

    # *** SALVATAGGIO IN CSV ***
    # Supponendo che tu voglia un unico CSV che contiene,
    # per ogni passo, i valori RS di Random e di Greedy.
    
    with open('output_rs_values.csv', 'w', newline='') as csvfile:
        fieldnames = ['step', 'rs_random', 'rs_greedy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Qui assumiamo che random e greedy abbiamo la stessa lunghezza (16 pattern per XOR a 4 variabili).
        for i in range(len(rs_vals_random)):
            writer.writerow({
                'step': i + 1,
                'rs_random': rs_vals_random[i],
                'rs_greedy': rs_vals_greedy[i]
            })

    # (Facoltativo) se vuoi salvare anche l'ordine di selezione, puoi farlo in un secondo file
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

    end = time.time()  # Fine timer
    print("\nTempo di esecuzione (s):", end - start)