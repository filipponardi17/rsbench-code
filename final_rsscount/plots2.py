#DOPPIA LINEA CON PLOTLY

import glob
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo

def main():
    # Trova tutti i file CSV nella cartella corrente che matchano il pattern
    csv_files = sorted(glob.glob('rq3_output_rs_values*.csv'))

    if not csv_files:
        print("Nessun file CSV trovato con il pattern 'rq1_output_rs_values*.csv'.")
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

    # Costruiamo i "fill" per rappresentare la std (area)
    # Per rs_random
    random_upper = random_mean + random_std
    random_lower = random_mean - random_std

    fill_random = go.Scatter(
        x = list(steps) + list(steps[::-1]),
        y = list(random_upper) + list(random_lower[::-1]),
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.2)',  # blu chiaro
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
    )

    trace_random = go.Scatter(
        x=steps,
        y=random_mean,
        mode='lines+markers',
        name='rs_random (mean)',
        line=dict(color='blue')
    )

    # Per rs_greedy
    greedy_upper = greedy_mean + greedy_std
    greedy_lower = greedy_mean - greedy_std

    fill_greedy = go.Scatter(
        x = list(steps) + list(steps[::-1]),
        y = list(greedy_upper) + list(greedy_lower[::-1]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',  # rosso chiaro
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
    )

    trace_greedy = go.Scatter(
        x=steps,
        y=greedy_mean,
        mode='lines+markers',
        name='rs_greedy (mean)',
        line=dict(color='red')
    )

    # Layout del grafico
    layout = go.Layout(
        title='Media e Deviazione Standard di rs_random e rs_greedy per Step',
        xaxis=dict(title='Step'),
        yaxis=dict(title='Valore'),
        hovermode='x'
    )

    # Creiamo la figura finale con i 4 oggetti
    fig = go.Figure(data=[fill_random, trace_random, fill_greedy, trace_greedy], layout=layout)

    # Mostra il grafico in locale (aprirà una pagina web con il grafico)
    pyo.plot(fig, filename='rq3_values_plot.html')

if __name__ == "__main__":
    main()