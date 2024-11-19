import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
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
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier()
}

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Cardiovascular Disease Prediction Bias Analysis", style={'textAlign': 'center'}),

    html.Div([
        # Bias Feature Dropdown
        html.Label("Induce Bias On:", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='bias-feature-dropdown',
            options=[{'label': feature, 'value': feature} for feature in FEATURE_NAMES],
            multi=True,
            placeholder="Select features to induce bias",
            style={'marginBottom': '20px'}
        ),

        # Sliders for Bias Thresholds
        html.Div(id='bias-sliders'),

        # Predict Button
        html.Button('Analyze Predictions', id='predict-button', n_clicks=0, style={
            'backgroundColor': '#007bff', 'color': 'white', 'border': 'none',
            'borderRadius': '5px', 'cursor': 'pointer', 'padding': '10px 20px'
        })
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),

    # Results Table
    html.Div(id='model-results-table', style={'padding': '20px', 'marginTop': '20px'}),
])

def induce_bias(df, bias_features, slider_values):
    """Induce bias in selected features based on slider thresholds."""
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
    # Dynamically create sliders based on selected features
    if not selected_features:
        return "No features selected for bias induction."
    
    sliders = []
    for feature in selected_features:
        sliders.append(html.Label(f"Set bias for the {feature}: "))
        sliders.append(
            dcc.Slider(
                id = f'bias-{feature}-slider',
                min = df_original[feature].min(),
                max = df_original[feature].max(),
                step = (df_original[feature].max() - df_original[feature].min()) / 100,
                value = df_original[feature].mean(),
                marks= {i: str(i) for i in range(int(df_original[feature].min()), int(df_original[feature].max()), 10)}

            )
        )
    return sliders

@app.callback(
    Output('model-results-table', 'children'),
    Input('predict-button', 'n_clicks'),
    State('bias-feature-dropdown', 'value')
)
def update_model_results(n_clicks, bias_features):
    if n_clicks == 0:
        return "Click 'Analyze Predictions' to see results."

    # Collect slider values for selected features
    slider_values = []
    if bias_features:
        for feature in bias_features:
            slider_id = f'bias-{feature}-slider'
            slider_value = dash.callback_context.inputs.get(f'{slider_id}.value', None)
            slider_values.append(slider_value)

    # Train-test split
    X = df_original[FEATURE_NAMES]
    y = df_original['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Induce bias if features are selected
    if bias_features and slider_values:
        biased_df = induce_bias(df_original, bias_features, slider_values)
        X_biased_train, _, y_biased_train, _ = train_test_split(
            biased_df[FEATURE_NAMES], biased_df['target'], test_size=0.2, random_state=42
        )
    else:
        biased_df = df_original
        X_biased_train, y_biased_train = X_train, y_train

    # Train and evaluate models
    results = []
    for model_name, model in MODELS.items():
        # Train on original data
        model.fit(X_train, y_train)
        orig_acc = accuracy_score(y_test, model.predict(X_test))

        # Train on biased data
        model.fit(X_biased_train, y_biased_train)
        biased_acc = accuracy_score(y_test, model.predict(X_test))

        results.append({'Model': model_name, 'Original Accuracy': orig_acc, 'Biased Accuracy': biased_acc})

    # Create results table
    results_df = pd.DataFrame(results)
    table = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in results_df.columns])),
        html.Tbody([
            html.Tr([html.Td(row[col]) for col in results_df.columns])
            for _, row in results_df.iterrows()
        ])
    ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '20px'})

    return table

if __name__ == '__main__':
    app.run_server(debug=True)