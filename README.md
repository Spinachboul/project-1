# Cardiovascular Disease Prediction Bias Analysis

## Overview

This project is a web-based application built using Dash that allows users to analyze how different patient characteristics affect the prediction of cardiovascular disease (CVD). It utilizes a synthetic dataset and machine learning techniques to explore the impact of bias in predictive modeling. Users can manipulate input parameters through sliders and see how these changes influence the predictions from both an original and a biased model.

## Features

- **Synthetic Dataset Generation**: The application generates a synthetic dataset representing patient characteristics relevant to cardiovascular disease prediction.
  
- **User Input Controls**: Interactive sliders allow users to adjust key parameters:
  - Age
  - Resting Blood Pressure (BP)
  - Maximum Heart Rate
  - Old Peak (a measure of exercise-induced angina)
  - Number of Major Vessels
  - Chest Pain Type

- **Model Training**: 
  - **Original Model**: Trained on the synthetic dataset to provide baseline predictions.
  - **Biased Model**: Trained on a modified dataset that introduces bias based on user-defined criteria, enabling the analysis of how bias can affect predictions.

- **Prediction Results**: After adjusting the sliders and clicking the "Analyze Predictions" button, users receive:
  - Predictions and confidence levels from the original model.
  - Predictions and confidence levels from the biased model.
  - A note highlighting that the biased model has been influenced by certain parameters.

## Installation

To run this application, you will need to have Python installed along with the required libraries. You can install the required libraries using pip:

```bash
pip install dash pandas numpy scikit-learn
```

## Running the Application

1. Save the provided code in a Python file, e.g., `app.py`.
2. Open a terminal and navigate to the directory where `app.py` is located.
3. Run the application with the following command:

   ```bash
   python app.py
   ```

4. Open a web browser and go to `http://127.0.0.1:8050/` to access the application.

## User Instructions

1. **Adjust Sliders**: Use the sliders to set the desired values for each parameter. Experiment with extreme values to observe how predictions change.
   
2. **Analyze Predictions**: After adjusting the sliders, click the **"Analyze Predictions"** button. The application will display:
   - **Original Model Prediction**: Diagnosis and confidence from the original unaltered model.
   - **Biased Model Prediction**: Diagnosis and confidence from the model trained on the biased dataset.

3. **Interpret Results**: Review the output to understand how different characteristics impact the predictions for cardiovascular disease risk.

## Code Structure

- **Imports**: Essential libraries such as Dash, Pandas, Numpy, and Scikit-learn are imported for data manipulation and model building.
  
- **Data Generation**: A synthetic dataset is created using a predefined set of features that include patient age, gender, blood pressure, heart rate, and other clinical indicators.
  
- **Target Variable Creation**: A function generates a binary target variable indicating high or low risk of CVD based on a risk score derived from patient characteristics.

- **Dash Application Layout**: The layout includes titles, descriptions, input controls (sliders), a predict button, and areas for displaying results.

- **Callbacks**: The app uses callbacks to link user interactions with the prediction logic, updating the results based on slider adjustments.