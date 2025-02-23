# DOPPIA LINEA SENZA PLOTLY

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Trova tutti i file CSV nella cartella corrente che matchano il pattern
    csv_files = sorted(glob.glob('rq3_output_rs_values*.csv'))

    if not csv_files:
        print("Nessun file CSV trovato con il pattern 'output_rs_values*.csv'.")
        return

    # Lista di DataFrame, uno per ciascun file
    list_dfs = []

    for file in csv_files:
        df = pd.read_csv(file)
        list_dfs.append(df)

    # Concatena tutti i DataFrame uno sotto l'altro
    combined_df = pd.concat(list_dfs, ignore_index=True)

    # Raggruppa per "step" e calcola media e std
    grouped = combined_df.groupby('step')
    means = grouped.mean()
    stds = grouped.std()

    # Estrai valori di step, mean e std
    steps = means.index
    random_mean = means['rs_random']
    random_std = stds['rs_random']

    greedy_mean = means['rs_greedy']
    greedy_std = stds['rs_greedy']

    # Configura il grafico
    plt.figure(figsize=(10, 6))

    # Grafico per rs_random
    plt.fill_between(
        steps,
        random_mean - random_std,
        random_mean + random_std,
        color='blue',
        alpha=0.2,
        label='rs_random (std)'
    )
    plt.plot(steps, random_mean, '-o', color='blue', label='rs_random (mean)')

    # Grafico per rs_greedy
    plt.fill_between(
        steps,
        greedy_mean - greedy_std,
        greedy_mean + greedy_std,
        color='red',
        alpha=0.2,
        label='rs_greedy (std)'
    )
    plt.plot(steps, greedy_mean, '-o', color='red', label='rs_greedy (mean)')

    # Aggiungi titoli e legende
    plt.title('Media e Deviazione Standard di rs_random e rs_greedy per Step')
    plt.xlabel('Step')
    plt.ylabel('Valore')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Salva il grafico su file
    plt.savefig('rq3_values_plot.png', dpi=300)

    # Mostra il grafico
    plt.show()

if __name__ == "__main__":
    main()