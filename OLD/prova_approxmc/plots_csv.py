import pandas as pd
import plotly.graph_objects as go

if __name__ == "__main__":
    # Leggiamo il CSV
    df = pd.read_csv('output_rs_values.csv')
    # df ora contiene le colonne: step, rs_random, rs_greedy

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['step'], 
        y=df['rs_random'], 
        mode='lines+markers', 
        name='RS - Random'
    ))
    fig.add_trace(go.Scatter(
        x=df['step'], 
        y=df['rs_greedy'], 
        mode='lines+markers', 
        name='RS - Greedy'
    ))
    fig.update_layout(
        title="Confronto RS Random vs Greedy",
        xaxis_title="Step",
        yaxis_title="Valore RS"
    )
    fig.show()