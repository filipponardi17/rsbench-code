import itertools
import numpy as np
from functools import reduce
from pyeda.inter import exprvars, And, Or, OneHot, Equal
from countrss_module_approxmc_2 import _pp_solution, _booldot, ConfigurableCustomFormula, count_rss

def generate_xor_patterns(n_variables):
    """
    Genera tutti i pattern XOR su n_variables variabili.
    Ritorna (gs_all, ys_all, gvecs_all) dove:
      - gs_all è un array di pattern binari (ogni pattern è una tupla, es. [1, 0, 1, 0])
      - ys_all è un array con le etichette ottenute originariamente con XOR (non usato poi)
      - gvecs_all è un array dei pattern in one-hot encoding.
    """
    # Genera tutte le combinazioni possibili
    gs_all = np.array(list(itertools.product([0, 1], repeat=n_variables)))
    ys_all = np.array([reduce(lambda a, b: a ^ b, g, 0) for g in gs_all])
    
    # One-hot encoding: per ogni bit, 0 -> [1,0] e 1 -> [0,1]
    gvecs_all = []
    for g in gs_all:
        enc = []
        for val in g:
            enc.extend([1, 0] if val == 0 else [0, 1])
        gvecs_all.append(enc)
    gvecs_all = np.array(gvecs_all)
    
    return gs_all, ys_all, gvecs_all

def run_incremental_tasks(n_variables, num_tasks):
    """
    Crea num_tasks "task", ciascuno costituito dai primi i pattern (con 1 <= i <= num_tasks).
    
    Per ogni task:
      - Stampa ogni pattern (in formato 1,0,1,0) e il label calcolato con la formula:
            ((c1 == c2) and (c3 == c4)) or ((c1 != c2) and (c3 != c4))
      - Carica i pattern (in one-hot) e i label nel dataset ConfigurableCustomFormula
      - Conta le soluzioni (RS) tramite count_rss.
    
    Si assume che il totale dei pattern possibili sia >= num_tasks.
    """
    gs_all, ys_all, gvecs_all = generate_xor_patterns(n_variables)
    
    if num_tasks > len(gs_all):
        raise ValueError(f"num_tasks = {num_tasks} è maggiore del numero di pattern disponibili ({len(gs_all)}).")
    
    for i in range(1, num_tasks + 1):
        print(f"\nTask {i}:")
        # Stampa i pattern (non in one-hot) e il label calcolato con la nuova formula.
        for pattern in gs_all[:i]:
            pattern_str = ",".join(str(bit) for bit in pattern)
            # Calcola il label secondo la formula:
            # ((c1 == c2) and (c3 == c4)) or ((c1 != c2) and (c3 != c4))
            # Assumiamo che pattern sia [c1, c2, c3, c4]
            label = ((pattern[0] == pattern[1] and pattern[2] == pattern[3]) or
                     (pattern[0] != pattern[1] and pattern[2] != pattern[3]))
            print(f"   Pattern: {pattern_str}  => label: {label}")
        
        # Ricalcola i label per il sottoinsieme usando la nuova formula
        new_labels = []
        for pattern in gs_all[:i]:
            label = ((pattern[0] == pattern[1] and pattern[2] == pattern[3]) or
                     (pattern[0] != pattern[1] and pattern[2] != pattern[3]))
            new_labels.append(int(label))
        new_labels = np.array(new_labels)
        
        # Carica i dati (pattern in one-hot e nuovi label) nel dataset
        dataset = ConfigurableCustomFormula(n_variables=n_variables)
        dataset.load_data(gvecs_all[:i], new_labels)
        
        print(f"### Counting RS for first {i} task(s) ###")
        count_rss(dataset)

if __name__ == "__main__":
    n_variables = 4
    num_tasks = 16  # Si considerano da 1 a 4 task incrementali
    run_incremental_tasks(n_variables, num_tasks)