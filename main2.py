import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

# Create target variable based on risk factors
def create_target(row):
    risk_score = 0
    risk_score += (row['age'] - 50) / 30
    risk_score += (row['restingBP'] - 120) / 80
    risk_score += (row['maxheartrate'] - 150) / 70
    risk_score += row['oldpeak'] / 6
    risk_score += row['noofmajorvessels'] / 3
    risk_score += 0.5 if row['exerciseangia'] == 1 else 0
    risk_score += 0.3 * row['chestpain']
    
    # Cap the risk score to a maximum threshold for realism
    probability = min(1, risk_score / 2)  # Adjust the divisor to control the cap
    return int(probability > 0.5)

# Create DataFrame and target
df_original = pd.DataFrame(synthetic_data)
df_original['target'] = df_original.apply(create_target, axis=1)

# Define models to test
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC(probability=True)
}

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Cardiovascular Disease Prediction Bias Analysis", style={'textAlign': 'center'}),
    
    html.Div("Compare the performance of multiple models under biased conditions.", style={'textAlign': 'center'}),
    
    html.Button(
        'Analyze Models',
        id='analyze-button',
        n_clicks=0,
        style={
            'margin': '20px',
            'padding': '10px 20px',
            'backgroundColor': '#007bff',
            'color': 'white',
            'border': 'none',
            'borderRadius': '5px',
            'cursor': 'pointer',
            'fontSize': '16px'
        }
    ),
    
    # Table for results
    html.Div(id='results-table', style={
        'marginTop': '20px',
        'padding': '20px',
        'backgroundColor': '#f8f9fa',
        'borderRadius': '10px'
    }),
])

def create_biased_dataset(df, age, maxheartrate, bp):
    """Create biased dataset with more pronounced effects."""
    df_biased = df.copy()
    age_mask = df_biased['age'] > age
    df_biased.loc[age_mask, 'target'] = 1
    hr_mask = df_biased['maxheartrate'] > maxheartrate
    df_biased.loc[hr_mask, 'target'] = 1
    bp_mask = df_biased['restingBP'] > bp
    df_biased.loc[bp_mask, 'target'] = 1
    return df_biased

@app.callback(
    Output('results-table', 'children'),
    Input('analyze-button', 'n_clicks')
)
def analyze_models(n_clicks):
    if n_clicks == 0:
        return "Click 'Analyze Models' to see results."

    # Split data into features and target
    X = df_original[FEATURE_NAMES]
    y = df_original['target']
    
    # Create biased dataset
    df_biased = create_biased_dataset(df_original, age=50, maxheartrate=140, bp=130)
    X_biased = df_biased[FEATURE_NAMES]
    y_biased = df_biased['target']
    
    results = []
    
    for model_name, model in models.items():
        # Train on original dataset
        model.fit(X, y)
        original_acc = accuracy_score(y, model.predict(X))
        
        # Train on biased dataset
        model.fit(X_biased, y_biased)
        biased_acc = accuracy_score(y_biased, model.predict(X_biased))
        
        # Append results
        results.append({
            "Model": model_name,
            "Accuracy (Original)": f"{original_acc:.2%}",
            "Accuracy (Biased)": f"{biased_acc:.2%}",
            "Robustness (Accuracy Drop)": f"{(original_acc - biased_acc):.2%}"
        })
    
    # Create table
    table_header = [html.Tr([html.Th(col) for col in results[0].keys()])]
    table_body = [html.Tr([html.Td(result[col]) for col in result.keys()]) for result in results]
    
    return html.Table(
        table_header + table_body,
        style={
            'width': '100%',
            'border': '1px solid #ddd',
            'borderCollapse': 'collapse',
            'textAlign': 'left',
            'marginTop': '10px'
        }
    )

if __name__ == '__main__':
    app.run_server(debug=True)
