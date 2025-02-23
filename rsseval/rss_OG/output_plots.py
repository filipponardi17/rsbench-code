import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if __name__ == "__main__":
    # Read the CSV files for random and greedy data
    df_random = pd.read_csv('random_for_plot.csv')
    df_greedy = pd.read_csv('greedy_for_plot.csv')

    # Create a figure with 2 rows and 1 column of subplots, sharing the x-axis
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Labels Accuracy", "Concept F1"),
                        shared_xaxes=True)

    # First subplot: Plot "Labels_Accuracy"
    fig.add_trace(go.Scatter(
        x=df_random['Step'],
        y=df_random['Labels_Accuracy'],
        mode='lines+markers',
        name='Random - Labels Accuracy'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_greedy['Step'],
        y=df_greedy['Labels_Accuracy'],
        mode='lines+markers',
        name='Greedy - Labels Accuracy'
    ), row=1, col=1)

    # Second subplot: Plot "Concept_F1"
    fig.add_trace(go.Scatter(
        x=df_random['Step'],
        y=df_random['Concept_F1'],
        mode='lines+markers',
        name='Random - Concept F1'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df_greedy['Step'],
        y=df_greedy['Concept_F1'],
        mode='lines+markers',
        name='Greedy - Concept F1'
    ), row=2, col=1)

    # Update the overall layout
    fig.update_layout(
        title="Comparison of Random vs Greedy Metrics",
        xaxis_title="Step",
        yaxis_title="Metric Value",
        height=700
    )

    fig.show()