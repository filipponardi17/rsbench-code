import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def preprocess_df(df, step_col='Step'):
    # Estrai il numero dallo step per poter ordinare correttamente (es. "step1" -> 1)
    df = df.copy()
    df['step_num'] = df[step_col].str.extract('(\d+)').astype(int)
    # Ordina per step_num
    df = df.sort_values('step_num')
    return df

def compute_stats(df, metric):
    # Raggruppa per step (usando il numero per mantenere l'ordinamento corretto)
    grouped = df.groupby(['step_num', 'Step'])
    means = grouped[metric].mean().reset_index()
    stds = grouped[metric].std().reset_index()
    # Ordina per step_num
    means = means.sort_values('step_num')
    stds = stds.sort_values('step_num')
    # Estrai le liste: usiamo il valore originale "Step" come etichetta
    steps = means['Step'].tolist()
    mean_vals = means[metric]
    std_vals = stds[metric]
    upper = mean_vals + std_vals
    lower = mean_vals - std_vals
    return steps, mean_vals.tolist(), std_vals.tolist(), upper.tolist(), lower.tolist()

if __name__ == "__main__":
    # Legge i CSV per random e greedy
    df_random = pd.read_csv('rq2_log_concept_random_score.csv')
    df_greedy = pd.read_csv('rq2_log_concept_greedy_score.csv')

    # Preprocessa (ordina per step)
    df_random = preprocess_df(df_random)
    df_greedy = preprocess_df(df_greedy)

    # Calcola media e std per "Labels_F1"
    steps_random_la, random_la_mean, random_la_std, random_la_upper, random_la_lower = compute_stats(df_random, 'Labels_F1')
    steps_greedy_la, greedy_la_mean, greedy_la_std, greedy_la_upper, greedy_la_lower = compute_stats(df_greedy, 'Labels_F1')

    # Assumiamo che i passi siano gli stessi per entrambi (altrimenti si potrebbe allineare in altro modo)
    steps_la = steps_random_la

    # Calcola media e std per "Concept_F1"
    steps_random_cf, random_cf_mean, random_cf_std, random_cf_upper, random_cf_lower = compute_stats(df_random, 'Concept_F1')
    steps_greedy_cf, greedy_cf_mean, greedy_cf_std, greedy_cf_upper, greedy_cf_lower = compute_stats(df_greedy, 'Concept_F1')

    steps_cf = steps_random_cf

    # Crea una figura con 2 subplot (righe) con asse x condiviso
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Labels F1", "Concept F1"),
                        shared_xaxes=True)

    # SUBPLOT 1: Labels Accuracy
    # Aree di fill per Random
    fig.add_trace(go.Scatter(
        x = steps_la + steps_la[::-1],
        y = random_la_upper + random_la_lower[::-1],
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.2)',  # blu chiaro
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ), row=1, col=1)

    # Linea per Random
    fig.add_trace(go.Scatter(
        x=steps_la,
        y=random_la_mean,
        mode='lines+markers',
        name='Random - Labels F1',
        line=dict(color='blue')
    ), row=1, col=1)

    # Aree di fill per Greedy
    fig.add_trace(go.Scatter(
        x = steps_la + steps_la[::-1],
        y = greedy_la_upper + greedy_la_lower[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',  # rosso chiaro
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ), row=1, col=1)

    # Linea per Greedy
    fig.add_trace(go.Scatter(
        x=steps_la,
        y=greedy_la_mean,
        mode='lines+markers',
        name='Greedy - Labels F1',
        line=dict(color='red')
    ), row=1, col=1)

    # SUBPLOT 2: Concept F1
    # Aree di fill per Random
    fig.add_trace(go.Scatter(
        x = steps_cf + steps_cf[::-1],
        y = random_cf_upper + random_cf_lower[::-1],
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ), row=2, col=1)

    # Linea per Random
    fig.add_trace(go.Scatter(
        x=steps_cf,
        y=random_cf_mean,
        mode='lines+markers',
        name='Random - Concept F1',
        line=dict(color='blue')
    ), row=2, col=1)

    # Aree di fill per Greedy
    fig.add_trace(go.Scatter(
        x = steps_cf + steps_cf[::-1],
        y = greedy_cf_upper + greedy_cf_lower[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ), row=2, col=1)

    # Linea per Greedy
    fig.add_trace(go.Scatter(
        x=steps_cf,
        y=greedy_cf_mean,
        mode='lines+markers',
        name='Greedy - Concept F1',
        line=dict(color='red')
    ), row=2, col=1)

    # Aggiornamento layout
    fig.update_layout(
        title="Confronto di Random vs Greedy (Media e Deviazione Standard)",
        xaxis_title="Step",
        yaxis_title="Valore della Metriaca",
        height=700
    )

    fig.show()