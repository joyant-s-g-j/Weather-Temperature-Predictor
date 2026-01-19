import gradio as gr
import pandas as pd
import numpy as np
import pickle

with open('diabetes_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)


def predict_diabetes(gender, age, hypertension, heart_disease,  health_risk,
                     smoking_history, bmi, HbA1c_level, blood_glucose_level):
    input_df = pd.DataFrame([[
        gender, age, hypertension, heart_disease, health_risk,
        smoking_history, bmi, HbA1c_level, blood_glucose_level
    ]], columns=[
        'gender', 'age', 'hypertension', 'heart_disease', 'health_risk',
        'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'
    ])

    pred = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][1]

    if pred == 1:
        return f"Prediction: Diabetic (Probability: {pred_proba*100:.2f}%)"
    else:
        return f"Prediction: Non-Diabetic (Probability: {(1 - pred_proba)*100:.2f}%)"
    

with gr.Blocks() as app:
    gr.Markdown("<h1 align='center'>Diabetes Prediction App</h1>")
    
    with gr.Row():
        with gr.Column():
            gender = gr.Dropdown(choices=['Male', 'Female'], label="Gender")
            age = gr.Number(label="Age")
            hypertension = gr.Dropdown(choices=[0, 1], label="Hypertension (0: No, 1: Yes)")
            heart_disease = gr.Dropdown(choices=[0, 1], label="Heart Disease (0: No, 1: Yes)")
            health_risk = gr.Number(label="Health Risk (Sum of Hypertension and Heart Disease)")
        with gr.Column():
            smoking_history = gr.Dropdown(choices=['never', 'No Info', 'current', 'former', 'ever', 'not current'], label="Smoking History")
            bmi = gr.Number(label="BMI")
            HbA1c_level = gr.Number(label="HbA1c Level")
            blood_glucose_level = gr.Number(label="Blood Glucose Level")
        
    predict_btn = gr.Button("Predict")
    output = gr.Textbox(label="Prediction Result")

    predict_btn.click(
        fn=predict_diabetes,
        inputs=[gender, age, hypertension, heart_disease, health_risk,
                smoking_history, bmi, HbA1c_level, blood_glucose_level],
        outputs=output
    )

app.launch()