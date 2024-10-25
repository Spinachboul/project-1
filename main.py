# import dash
# from dash import dcc, html

# import numpy as np
# import pickle
# from dash.dependencies import Input, Output

# # Load pre-trained models
# with open('models/original_logistic_regression_model.pkl', 'rb') as f:
#     original_model = pickle.load(f)

# with open('models/biased_logistic_regression_model.pkl', 'rb') as f:
#     biased_model = pickle.load(f)


# # Initialize the dash app
# app = dash.Dash(__name__)

# # Define default values for the other 9 features (these can be actual means or representative values)
# other_features_defaults = [0.5] * 9  # Replace with actual default values for consistency with training

# # Define the layout of the app
# app.layout = html.Div([
#     html.H1("Mitigating Bias in Healthcare AI"),

#     html.Div([
#         html.Label('Age'),
#         dcc.Slider(
#             id='age-slider', min=20, max=80, value=50,
#             marks={i: str(i) for i in range(20, 81, 10)}, step=1
#         )
#     ], style={'margin': '20px'}),

#     html.Div([
#         html.Label('Cholesterol'),
#         dcc.Slider(
#             id='cholesterol-slider', min=100, max=400, value=200,
#             marks={i: str(i) for i in range(100, 401, 50)}, step=1
#         )
#     ], style={'margin': '20px'}),

#     html.Div([
#         html.Label('Blood Pressure'),
#         dcc.Slider(
#             id='bp-slider', min=80, max=200, value=120,
#             marks={i: str(i) for i in range(80, 201, 20)}, step=1
#         )
#     ], style={'margin': '20px'}),

#     html.Div(id='output-predictions', style={'marginTop': 30})
# ])

# @app.callback(
#     Output('output-predictions', 'children'),
#     [Input('age-slider', 'value'),
#      Input('cholesterol-slider', 'value'),
#      Input('bp-slider', 'value')]
# )
# def update_predictions(age, cholesterol, blood_pressure):
#     # Constructing input data with extreme values
#     # Testing with minimum and maximum values to see model sensitivity
#     input_data_min = np.array([[20, 100, 80] + other_features_defaults])
#     input_data_max = np.array([[80, 400, 200] + other_features_defaults])
    
#     # Predictions with extreme low values
#     min_pred_original = original_model.predict(input_data_min)[0]
#     min_pred_biased = biased_model.predict(input_data_min)[0]

#     # Predictions with extreme high values
#     max_pred_original = original_model.predict(input_data_max)[0]
#     max_pred_biased = biased_model.predict(input_data_max)[0]
    
#     # Print outputs for debugging
#     print("Extreme Low - Original Prediction:", min_pred_original)
#     print("Extreme Low - Biased Prediction:", min_pred_biased)
#     print("Extreme High - Original Prediction:", max_pred_original)
#     print("Extreme High - Biased Prediction:", max_pred_biased)
    
#     # Display results for both extreme cases
#     return [
#         html.Div(f"Extreme Low Values - Original Prediction: {'Disease' if min_pred_original == 1 else 'No Disease'}"),
#         html.Div(f"Extreme Low Values - Biased Prediction: {'Disease' if min_pred_biased == 1 else 'No Disease'}"),
#         html.Br(),
#         html.Div(f"Extreme High Values - Original Prediction: {'Disease' if max_pred_original == 1 else 'No Disease'}"),
#         html.Div(f"Extreme High Values - Biased Prediction: {'Disease' if max_pred_biased == 1 else 'No Disease'}")
#     ]


# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Sample dataset
df_original = pd.read_csv('Cardiovascular_Disease_Dataset\Cardiovascular_Disease_Dataset.csv')  # Use your original dataset
df_biased = df_original.copy()

# Logistic regression model (initial training)
features = df_original.drop(columns=['target'])  # Assuming 'target' is the label
target = df_original['target']

model = LogisticRegression(max_iter=1000)
model.fit(features, target)

# Dash app initialization
app = dash.Dash(__name__)

# Layout for the app
app.layout = html.Div([
    # Title
    html.H1("Bias in AI Healthcare Predictions", style={'textAlign': 'center'}),
    
    # Sliders to induce bias (age, maxheartrate, oldpeak as examples)
    html.Div([
        html.Label('Age:'),
        dcc.Slider(id='age-slider', min=30, max=80, step=1, value=50),
        html.Label('Max Heart Rate:'),
        dcc.Slider(id='heartrate-slider', min=80, max=220, step=1, value=120),
        html.Label('Old Peak:'),
        dcc.Slider(id='oldpeak-slider', min=0, max=6, step=0.1, value=2.0),
    ], style={'padding': '20px'}),
    
    # Predict button
    html.Button('Predict', id='predict-button', n_clicks=0),
    
    # Display the original and biased predictions
    html.Div(id='original-prediction', style={'padding': '20px'}),
    html.Div(id='biased-prediction', style={'padding': '20px'}),
])

# Function to retrain the model with biased data
def retrain_model_with_bias(df_biased):
    features_biased = df_biased.drop(columns=['target'])
    target_biased = df_biased['target']
    biased_model = LogisticRegression(max_iter=1000)
    biased_model.fit(features_biased, target_biased)
    return biased_model

# Callback to handle retraining and predictions
@app.callback(
    [Output('original-prediction', 'children'),
     Output('biased-prediction', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('age-slider', 'value'),
     State('heartrate-slider', 'value'),
     State('oldpeak-slider', 'value')]
)
def update_predictions(n_clicks, age, maxheartrate, oldpeak):
    # Prepare the input data by ensuring it has the same number of features as the model
    # Let's assume the feature list for the original dataset is like this: ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    # Default values for the other features not controlled by sliders (these should match your actual dataset)
    # Update the values accordingly based on your dataset
    default_values = {
        'sex': 1,
        'cp': 2,
        'trestbps': 120,
        'chol': 200,
        'fbs': 0,
        'restecg': 1,
        'thalach': maxheartrate,  # This is controlled by the slider
        'exang': 0,
        'oldpeak': oldpeak,  # This is controlled by the slider
        'slope': 2,
        'ca': 0,
        'thal': 2
    }

    # Input data based on slider values and default values for the rest
    input_data = np.array([[age, default_values['sex'], default_values['cp'], default_values['trestbps'], default_values['chol'], 
                            default_values['fbs'], default_values['restecg'], maxheartrate, default_values['exang'], oldpeak, 
                            default_values['slope'], default_values['ca'], default_values['thal']]])
    
    # Original prediction
    original_pred = model.predict(input_data)[0]
    
    # Induce bias in the dataset (example for maxheartrate)
    df_biased['maxheartrate'] = np.where(df_biased['maxheartrate'] > maxheartrate, maxheartrate, df_biased['maxheartrate'])
    
    # Retrain the model with biased dataset
    biased_model = retrain_model_with_bias(df_biased)
    
    # Biased prediction
    biased_pred = biased_model.predict(input_data)[0]
    
    # Display results
    return (f"Original Model Prediction: {'Disease' if original_pred == 1 else 'No Disease'}",
            f"Biased Model Prediction: {'Disease' if biased_pred == 1 else 'No Disease'}")

if __name__ == '__main__':
    app.run_server(debug=True)
