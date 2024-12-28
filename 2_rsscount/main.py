# project/main.py
import numpy as np
import subprocess
import pickle
import re

# Funzione per generare tutti i dati XOR (gs, ys)
def generate_xor_data(n_variables=3):
    gs = []
    ys = []
    for i in range(8):  # 2^(3) = 8 combinazioni
        bits = [(i >> b) & 1 for b in reversed(range(n_variables))]
        # codifica one-hot: 0 -> [1,0], 1 -> [0,1]
        g_enc = []
        for b in bits:
            if b == 0:
                g_enc += [1,0]
            else:
                g_enc += [0,1]
        y = bits[0]^bits[1]^bits[2]
        gs.append(g_enc)
        ys.append(y)
    return np.array(gs), np.array(ys)

def count_RS_for_task(gs, ys):
    with open("temp_data.pkl", "wb") as f:
        pickle.dump((gs, ys), f)

    cmd = ["python", "gen-rss-count.py", "configxor", "--data-path", "temp_data.pkl", "-n", "3", "-E"]
    result = subprocess.run(cmd, capture_output=True, text=True)  # tolto check=True

    # Controlliamo se il processo è andato in errore
    if result.returncode != 0:
        print("Errore nell'esecuzione di gen-rss-count.py:")
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        raise RuntimeError(f"gen-rss-count.py ha fallito con codice di errore {result.returncode}")

    match = re.search(r'(\d+)\s+solutions', result.stdout)
    if match:
        return int(match.group(1))
    else:
        return 0

def main():
    # Generiamo tutti i dati XOR
    full_gs, full_ys = generate_xor_data(3)

    # Definiamo i 4 task:
    # Task 1: [0,1]
    # Task 2: [2,3]
    # Task 3: [4,5]
    # Task 4: [6,7]
    tasks = {
        "T1": (0,2),  # slice [0:2]
        "T2": (2,4),  # slice [2:4]
        "T3": (4,6),  # slice [4:6]
        "T4": (6,8)   # slice [6:8]
    }

    # S insieme dei task selezionati
    S = set()

    # Condizione di terminazione: |S| < 3
    while len(S) < 3:
        # Consideriamo i task non in S
        candidates = [t for t in tasks.keys() if t not in S]

        # Per ognuno, contiamo le RS
        rs_counts = {}
        for t in candidates:
            start, end = tasks[t]
            gs = full_gs[start:end]
            ys = full_ys[start:end]
            rs = count_RS_for_task(gs, ys)
            rs_counts[t] = rs
            print(f"Task {t}: {rs} RS")

        # Selezioniamo il task con più RS
        best_task = max(rs_counts, key=rs_counts.get)
        print(f"Seleziono il task {best_task} con {rs_counts[best_task]} RS\n")

        # Aggiungiamo best_task a S
        S.add(best_task)

    print(f"Terminato! S = {S}, dimensione {len(S)}")

    # Stampiamo i gs dei task in S
    full_gs, full_ys = generate_xor_data(3)  # Ricrea i dati, o usali se salvati globalmente
    tasks = {
        "T1": (0,2),
        "T2": (2,4),
        "T3": (4,6),
        "T4": (6,8)
    }

    print("I gs contenuti in S sono:")
    for t in S:
        start, end = tasks[t]
        print(f"Task {t} -> gs:")
        print(full_gs[start:end])

if __name__ == "__main__":
    main()