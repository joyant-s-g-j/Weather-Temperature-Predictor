# Diabetes Predictor 🩺💉

🚀 **Diabetes Predictor** is a machine learning-powered web application that predicts the likelihood of diabetes based on various health parameters. Built as a **Phitron AI/ML Course Final Project**, this application uses a **Random Forest Classifier** trained on health data to provide accurate predictions. The interactive web interface is powered by **Gradio**, making it easy for users to input their health information and get instant results.

### 🔗 Live Demo  
Check out the live version here: **[Diabetes Predictor Live](https://huggingface.co/spaces/joyant-s-g-j/Diabetes-Predictor)**  

## ✨ Features  
🔬 **ML-Powered Predictions** – Uses a trained Random Forest model for accurate diabetes prediction.  
📊 **Probability Scores** – Shows prediction confidence with probability percentages.  
🧹 **Data Preprocessing** – Implements outlier detection, feature scaling, and one-hot encoding.  
🎯 **Feature Engineering** – Custom health risk feature combining hypertension and heart disease.  
🖥️ **Interactive UI** – Clean and intuitive Gradio-based web interface.  
📱 **Responsive Design** – Works seamlessly across desktop and mobile devices.  

## 🛠️ Tech Stack  
- **Machine Learning**: Scikit-learn, Random Forest Classifier  
- **Data Processing**: Pandas, NumPy  
- **Web Framework**: Gradio  
- **Model Serialization**: Pickle  
- **Preprocessing**: StandardScaler, OneHotEncoder, ColumnTransformer  

## 📋 Input Parameters  
| Parameter | Description |
|-----------|-------------|
| **Gender** | Male / Female |
| **Age** | Patient's age in years |
| **Hypertension** | 0 (No) / 1 (Yes) |
| **Heart Disease** | 0 (No) / 1 (Yes) |
| **Health Risk** | Sum of Hypertension and Heart Disease |
| **Smoking History** | never, No Info, current, former, ever, not current |
| **BMI** | Body Mass Index |
| **HbA1c Level** | Hemoglobin A1c level |
| **Blood Glucose Level** | Blood glucose level (mg/dL) |

##  Project Structure  
```
diabetes-predictor/
├── app.py                          # Gradio web application
├── rf_train.py                     # Model training script
├── requirements.txt                # Python dependencies
├── diabetes_prediction_model.pkl   # Trained model (generated after training)
├── dataset/
│   └── diabetes_prediction_dataset.csv  # Training dataset
└── README.md                       # Project documentation
```

## 📈 Model Performance  
The Random Forest Classifier is trained with the following configuration:  
- **n_estimators**: 50  
- **max_depth**: 10  
- **min_samples_split**: 2  
- **random_state**: 42  

The model includes preprocessing steps:  
- **Numerical Features**: StandardScaler normalization  
- **Categorical Features**: OneHotEncoder (drop first)  
- **Outlier Removal**: IQR-based outlier detection  

## 🔧 Model Pipeline  
```
Input Data → Outlier Removal → Feature Engineering → Preprocessing → Random Forest → Prediction
```

## 📊 Dataset  
The model is trained on the **[Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)** from Kaggle, containing health records with features like:  
- Demographics (age, gender)  
- Medical history (hypertension, heart disease)  
- Lifestyle factors (smoking history)  
- Health metrics (BMI, HbA1c, blood glucose)  

## 🙏 Acknowledgements  
- **Phitron** – For the AI/ML course and guidance  
- **Scikit-learn** – For the machine learning library  
- **Gradio** – For the amazing web UI framework  

---

Made with ❤️ by **[Joyant Sheikhar Gupta Joy](https://joyant.me)** | Phitron AI/ML Final Project 🎓
