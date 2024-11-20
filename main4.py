import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Define feature names
FEATURE_NAMES = ['age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol',
                 'fastingbloodsugar', 'restingrelectro', 'maxheartrate',
                 'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels']

# Create synthetic dataset
np.random.seed(42)
n_samples = 1000
synthetic_data = {
    'patientid': range(1, n_samples + 1),
    'age': np.random.normal(55, 10, n_samples).clip(30, 80),
    'gender': np.random.binomial(1, 0.5, n_samples),
    'chestpain': np.random.randint(0, 4, n_samples),
    'restingBP': np.random.normal(130, 20, n_samples).clip(90, 200),
    'serumcholestrol': np.random.normal(220, 40, n_samples).clip(150, 400),
    'fastingbloodsugar': np.random.binomial(1, 0.2, n_samples),
    'restingrelectro': np.random.randint(0, 3, n_samples),
    'maxheartrate': np.random.normal(150, 20, n_samples).clip(80, 220),
    'exerciseangia': np.random.binomial(1, 0.3, n_samples),
    'oldpeak': np.random.normal(1.5, 1.2, n_samples).clip(0, 6),
    'slope': np.random.randint(1, 4, n_samples),
    'noofmajorvessels': np.random.randint(0, 4, n_samples)
}

def create_target(row):
    """Create a binary target variable based on risk factors."""
    risk_score = 0
    risk_score += (row['age'] - 50) / 30
    risk_score += (row['restingBP'] - 120) / 80
    risk_score += (row['maxheartrate'] - 150) / 70
    risk_score += row['oldpeak'] / 6
    risk_score += row['noofmajorvessels'] / 3
    risk_score += 0.5 if row['exerciseangia'] == 1 else 0
    risk_score += 0.3 * row['chestpain']
    probability = min(1, risk_score / 2)
    return int(probability > 0.5)

# Create DataFrame and target
df_original = pd.DataFrame(synthetic_data)
df_original['target'] = df_original.apply(create_target, axis=1)

# List of models
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Cardiovascular Disease Prediction Bias Analysis", style={'textAlign': 'center'}),
    html.Div([
        html.Label("Induce Bias On:", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='bias-feature-dropdown',
            options=[{'label': feature, 'value': feature} for feature in FEATURE_NAMES],
            multi=True,
            placeholder="Select features to induce bias",
            style={'marginBottom': '20px'}
        ),
        html.Div(id='bias-sliders'),
        html.Button('Analyze Predictions', id='predict-button', n_clicks=0, style={
            'backgroundColor': '#007bff', 'color': 'white', 'border': 'none',
            'borderRadius': '5px', 'cursor': 'pointer', 'padding': '10px 20px'
        })
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
    html.Div(id='model-results-table', style={'padding': '20px', 'marginTop': '20px'}),
    html.Div(id='feature-importance', style={'padding': '20px', 'marginTop': '20px'})
])

def induce_bias(df, bias_features, slider_values):
    biased_df = df.copy()
    for feature, threshold in zip(bias_features, slider_values):
        if feature == 'age':
            biased_df.loc[df['age'] > threshold, 'target'] = 1
        elif feature == 'maxheartrate':
            biased_df.loc[df['maxheartrate'] > threshold, 'target'] = 1
        elif feature == 'restingBP':
            biased_df.loc[df['restingBP'] > threshold, 'target'] = 1
        elif feature == 'oldpeak':
            biased_df.loc[df['oldpeak'] > threshold, 'target'] = 1
    return biased_df

@app.callback(
    Output('bias-sliders', 'children'),
    Input('bias-feature-dropdown', 'value')
)
def update_bias_sliders(selected_features):
    if not selected_features:
        return "No features selected for bias induction."
    sliders = []
    for feature in selected_features:
        sliders.append(html.Label(f"Set bias for the {feature}: "))
        sliders.append(
            dcc.Slider(
                id=f'bias-{feature}-slider',
                min=df_original[feature].min(),
                max=df_original[feature].max(),
                step=(df_original[feature].max() - df_original[feature].min()) / 100,
                value=df_original[feature].mean(),
                marks={int(i): str(int(i)) for i in np.linspace(df_original[feature].min(), df_original[feature].max(), 5)}
            )
        )
    return sliders

@app.callback(
    Output('model-results-table', 'children'),
    Output('feature-importance', 'children'),
    Input('predict-button', 'n_clicks'),
    State('bias-feature-dropdown', 'value'),
    **{f'bias-{feature}-slider.value': State(f'bias-{feature}-slider', 'value') for feature in FEATURE_NAMES}
)
def update_model_results(n_clicks, bias_features, **slider_values):
    if n_clicks == 0:
        return "Click 'Analyze Predictions' to see results.", "Feature importance will be displayed here."

    # Convert slider values to a list for bias induction
    slider_values = [slider_values[f'bias-{feature}-slider.value'] for feature in bias_features]

    biased_df = induce_bias(df_original, bias_features, slider_values) if bias_features else df_original

    X = df_original[FEATURE_NAMES]
    y = df_original['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_biased_train, _, y_biased_train, _ = train_test_split(
        biased_df[FEATURE_NAMES], biased_df['target'], test_size=0.2, random_state=42
    )

    results = []
    feature_importance_content = []
    for model_name, model in MODELS.items():
        model.fit(X_train, y_train)
        orig_acc = accuracy_score(y_test, model.predict(X_test))

        model.fit(X_biased_train, y_biased_train)
        biased_acc = accuracy_score(y_test, model.predict(X_test))

        results.append({'Model': model_name, 'Original Accuracy': orig_acc, 'Biased Accuracy': biased_acc})

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_content.append(
                html.Div([
                    html.H4(f"Feature Importance for {model_name}"),
                    html.Ul([html.Li(f"{feature}: {importance:.4f}") for feature, importance in zip(FEATURE_NAMES, importances)])
                ])
            )

    results_df = pd.DataFrame(results)
    table = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in results_df.columns])),
        html.Tbody([
            html.Tr([html.Td(row[col]) for col in results_df.columns])
            for _, row in results_df.iterrows()
        ])
    ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '20px'})

    return table, feature_importance_content

if __name__ == '__main__':
    app.run_server(debug=True)
