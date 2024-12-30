import dash
from dash import dcc, html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

orig_df = pd.read_csv(r'E:\Mars_Dust_Storm\MDAD\data\MDAD.csv')
proc_df = pd.read_csv(r'E:\Mars_Dust_Storm\MDAD\data\MDAD_refined_cleaned.csv')

orig_df = orig_df.apply(pd.to_numeric, errors='ignore')
proc_df = proc_df.apply(pd.to_numeric, errors='ignore')

app.layout = html.Div(
    style={
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': '#f0f4f8',
        'padding': '20px'
    },
    children=[
        html.H1(
            'Mars Dust Storm Analysis Visualization',
            style={
                'textAlign': 'center',
                'color': '#4A90E2',
                'marginBottom': '30px',
                'fontSize': '32px'
            }
        ),
        html.Div(
            style={
                'display': 'flex',
                'justifyContent': 'center',
                'marginBottom': '30px'
            },
            children=[
                dcc.Dropdown(
                    id='metric-dd',
                    options=[{'label': col, 'value': col} for col in orig_df.columns],
                    value='Area (km²)',
                    style={
                        'width': '50%',
                        'padding': '10px',
                        'fontSize': '18px',
                        'borderRadius': '8px',
                        'margin': 'auto',
                        'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)',
                        'transition': '0.3s'
                    }
                ),
            ]
        ),
        html.Div(
            style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(400px, 1fr))',
                'gap': '20px'
            },
            children=[
                dcc.Graph(id='box-plot', style={'height': '400px'}),
                dcc.Graph(id='histogram', style={'height': '400px'})
            ]
        ),
        html.Div(
            style={
                'display': 'flex',
                'justifyContent': 'center',
                'marginTop': '40px'
            },
            children=[
                dcc.Graph(id='heatmap', style={'width': '80%', 'height': '500px'})
            ]
        ),
    ]
)

@app.callback(
    Output('box-plot', 'figure'),
    [Input('metric-dd', 'value')]
)
def update_box_plot(selected_metric):
    if selected_metric in orig_df.columns:
        box_fig = go.Figure()
        box_fig.add_trace(go.Box(
            y=orig_df[selected_metric],
            name="Original",
            boxmean=True,
            marker_color='rgba(255, 99, 132, 0.6)',
            line_color='rgba(255, 99, 132, 1)'
        ))
        box_fig.add_trace(go.Box(
            y=proc_df[selected_metric],
            name="Processed",
            boxmean=True,
            marker_color='rgba(54, 162, 235, 0.6)',
            line_color='rgba(54, 162, 235, 1)'
        ))
        box_fig.update_layout(
            title=f'Box Plot for {selected_metric}',
            xaxis_title="Dataset",
            yaxis_title=selected_metric,
            template='plotly_dark',
            plot_bgcolor='#2d3e50',
            font=dict(color='white')
        )
        return box_fig
    return {}

@app.callback(
    Output('histogram', 'figure'),
    [Input('metric-dd', 'value')]
)
def update_histogram(selected_metric):
    if selected_metric in orig_df.columns:
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=orig_df[selected_metric],
            name="Original",
            marker_color='rgba(255, 99, 132, 0.7)',
            opacity=0.75
        ))
        hist_fig.add_trace(go.Histogram(
            x=proc_df[selected_metric],
            name="Processed",
            marker_color='rgba(54, 162, 235, 0.7)',
            opacity=0.75
        ))
        hist_fig.update_layout(
            title=f'Histogram for {selected_metric}',
            xaxis_title=selected_metric,
            yaxis_title="Count",
            barmode='overlay',
            template='plotly_dark',
            plot_bgcolor='#2d3e50',
            font=dict(color='white')
        )
        return hist_fig
    return {}

@app.callback(
    Output('heatmap', 'figure'),
    [Input('metric-dd', 'value')]
)
def update_heatmap(selected_metric):
    if selected_metric in orig_df.columns:
        corr_matrix = orig_df.select_dtypes(include='number').corr()
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Viridis',
            colorbar=dict(title="Correlation")
        ))
        heatmap_fig.update_layout(
            title='Correlation Heatmap',
            xaxis_title="Features",
            yaxis_title="Features",
            template='plotly_dark',
            plot_bgcolor='#2d3e50',
            font=dict(color='white')
        )
        return heatmap_fig
    return {}

if __name__ == '__main__':
    app.run_server(debug=True)
