import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

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
df_biased = df_original.copy()

# Train the model
features = df_original[FEATURE_NAMES]
target = df_original['target']
model = LogisticRegression(max_iter=1000)
model.fit(features, target)

# Rest of the code remains the same until create_input_dataframe function
app = dash.Dash(__name__)

# Layout remains the same as in previous version
app.layout = html.Div([
    # Title and description remain the same
    html.H1("Cardiovascular Disease Prediction Bias Analysis", style={'textAlign': 'center'}),
    
    html.Div([
        html.P("Analyze how different patient characteristics affect the prediction of cardiovascular disease.",
               style={'textAlign': 'center', 'fontSize': '16px', 'margin': '20px'})
    ]),
    
    # Input Controls remain the same
    html.Div([
        html.Div([
            html.Div([
                html.Label('Age:', style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='age-slider',
                    min=30,
                    max=80,
                    step=1,
                    value=50,
                    marks={i: str(i) for i in range(30, 81, 10)},
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label('Resting BP:', style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='bp-slider',
                    min=90,
                    max=200,
                    step=1,
                    value=130,
                    marks={i: str(i) for i in range(90, 201, 20)},
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
        ], style={'margin': '20px 0'}),
        
        html.Div([
            html.Div([
                html.Label('Max Heart Rate:', style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='heartrate-slider',
                    min=80,
                    max=220,
                    step=1,
                    value=150,
                    marks={i: str(i) for i in range(80, 221, 20)},
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label('Old Peak:', style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='oldpeak-slider',
                    min=0,
                    max=6,
                    step=0.1,
                    value=2.0,
                    marks={i: str(i) for i in range(0, 7)},
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
        ], style={'margin': '20px 0'}),
        
        html.Div([
            html.Div([
                html.Label('Number of Major Vessels:', style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='vessels-slider',
                    min=0,
                    max=3,
                    step=1,
                    value=1,
                    marks={i: str(i) for i in range(4)},
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label('Chest Pain Type:', style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='chestpain-slider',
                    min=0,
                    max=3,
                    step=1,
                    value=1,
                    marks={i: str(i) for i in range(4)},
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
        ], style={'margin': '20px 0'}),
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
    
    # Predict button
    html.Button(
        'Analyze Predictions',
        id='predict-button',
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
    
    # Results container
    html.Div([
        html.Div(id='original-prediction', style={
            'padding': '20px',
            'margin': '10px',
            'backgroundColor': '#e9ecef',
            'borderRadius': '5px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }),
        html.Div(id='biased-prediction', style={
            'padding': '20px',
            'margin': '10px',
            'backgroundColor': '#e9ecef',
            'borderRadius': '5px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }),
    ], style={'padding': '20px'}),
])

def create_input_dataframe(age, bp, maxheartrate, oldpeak, vessels, chestpain):
    """Create a pandas DataFrame with proper feature names for prediction"""
    default_values = {
        'age': age,
        'gender': 1,
        'chestpain': chestpain,
        'restingBP': bp,
        'serumcholestrol': 220,
        'fastingbloodsugar': 0,
        'restingrelectro': 1,
        'maxheartrate': maxheartrate,
        'exerciseangia': 0,
        'oldpeak': oldpeak,
        'slope': 2,
        'noofmajorvessels': vessels
    }
    return pd.DataFrame([default_values])[FEATURE_NAMES]

def create_biased_dataset(df, age, maxheartrate, bp):
    """Create biased dataset with more pronounced effects"""
    df_biased = df.copy()
    
    # Apply bias based on age
    age_mask = df_biased['age'] > age
    df_biased.loc[age_mask, 'target'] = 1
    
    # Apply bias based on heart rate
    hr_mask = df_biased['maxheartrate'] > maxheartrate
    df_biased.loc[hr_mask, 'target'] = 1
    
    # Apply bias based on blood pressure
    bp_mask = df_biased['restingBP'] > bp
    df_biased.loc[bp_mask, 'target'] = 1
    
    return df_biased

@app.callback(
    [Output('original-prediction', 'children'),
     Output('biased-prediction', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('age-slider', 'value'),
     State('bp-slider', 'value'),
     State('heartrate-slider', 'value'),
     State('oldpeak-slider', 'value'),
     State('vessels-slider', 'value'),
     State('chestpain-slider', 'value')]
)
def update_predictions(n_clicks, age, bp, maxheartrate, oldpeak, vessels, chestpain):
    if n_clicks == 0:
        return "Click 'Analyze Predictions' to see results", "Click 'Analyze Predictions' to see results"
    
    # Create input DataFrame
    input_data = create_input_dataframe(age, bp, maxheartrate, oldpeak, vessels, chestpain)
    print("Input Data for Prediction:", input_data)  # Debugging line

    # Original prediction
    original_pred = model.predict(input_data)[0]
    original_prob = model.predict_proba(input_data)[0][1]
    
    # Create biased dataset and retrain model
    df_biased_temp = create_biased_dataset(df_original, age, maxheartrate, bp)
    print("Biased Dataset:\n", df_biased_temp['target'].value_counts())  # Debugging line
    
    biased_model = LogisticRegression(max_iter=1000)
    biased_model.fit(df_biased_temp[FEATURE_NAMES], df_biased_temp['target'])
    
    # Biased prediction
    biased_pred = biased_model.predict(input_data)[0]
    biased_prob = biased_model.predict_proba(input_data)[0][1]

    print("Original Prediction:", original_pred, original_prob)  # Debugging line
    print("Biased Prediction:", biased_pred, biased_prob)  # Debugging line

    # Format results
    original_result = html.Div([
        html.H4("Original Model Prediction:", style={'color': '#007bff'}),
        html.P(f"Diagnosis: {'High Risk of CVD' if original_pred == 1 else 'Low Risk of CVD'}", 
               style={'fontSize': '16px', 'fontWeight': 'bold'}),
        html.P(f"Confidence: {original_prob:.1%}", 
               style={'fontSize': '14px'})
    ])
    
    biased_result = html.Div([
        html.H4("Biased Model Prediction:", style={'color': '#dc3545'}),
        html.P(f"Diagnosis: {'High Risk of CVD' if biased_pred == 1 else 'Low Risk of CVD'}", 
               style={'fontSize': '16px', 'fontWeight': 'bold'}),
        html.P(f"Confidence: {biased_prob:.1%}", 
               style={'fontSize': '14px'}),
        html.P("Note: This model has been artificially biased based on age, heart rate, and blood pressure.", 
               style={'fontSize': '12px', 'fontStyle': 'italic'})
    ])
    
    return original_result, biased_result

if __name__ == '__main__':
    app.run_server(debug=True)