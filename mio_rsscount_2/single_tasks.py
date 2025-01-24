#FILE CHE FUNZIONA CON UN SINGOLO TASK / PATTERN

import itertools
import numpy as np
from functools import reduce
from pyeda.inter import exprvars, And, Or, OneHot, Equal
from countrss_module_approxmc_2 import _pp_solution, _booldot, ConfigurableXOR
from countrss_module_approxmc_2 import count_rss


n_variables = 4

# Caso con una lista vuota (nessun esempio)
gs1 = np.array([])  # lista vuota
ys1 = np.array([])  # output corrispondente, anch'esso vuoto

# Creazione di gvecs1 vuoto (necessario per `ConfigurableXOR`)
gvecs1 = np.empty((0, n_variables * 2))  # Lista vuota con dimensione compatibile

# Inizializzazione del task
task1 = ConfigurableXOR(n_variables=4)
task1.load_data(gvecs1, ys1)  # Caricamento dei dati vuoti

# Debug per verificare i dati caricati
print("task1 gvecs:", task1.gvecs)  # Dovrebbe essere vuoto
print("task1 ys:", task1.ys)        # Dovrebbe essere vuoto

# Conteggio delle soluzioni RSS
print("### Counting RS for Task1 (with empty data) ###")
count_rss(task1)  # Esegue il conteggio con i dati vuoti