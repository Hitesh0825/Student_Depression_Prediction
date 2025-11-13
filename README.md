# Depression Prediction App

A machine learning-powered web application that predicts depression risk based on various lifestyle and academic factors using a Gradient Boosting Classifier model.

### Project Overview

This project uses student lifestyle data to predict depression risk with approximately 76.5% accuracy. The application is built using Streamlit and a trained machine learning pipeline for real-time predictions.

## Key Features

User-friendly Interface: Simple form-based input for all required features

Real-time Predictions: Instant depression risk assessment

Interactive UI: Built with Streamlit for a seamless user experience

Pre-trained Model: Gradient Boosting Classifier with optimized hyperparameters

## Model Performance

### Best Model: Gradient Boosting Classifier

- Accuracy: 76.5%

- F1 Score: 75.6% (macro average)

### Hyperparameters:

- n_estimators: 300

- max_depth: 3

- min_samples_split: 2

### Technology Stack

- Python 3.x

- Streamlit – Web interface

- scikit-learn – ML pipeline

- pandas – Data manipulation

- joblib – Model persistence

### Installation & Setup

1.  **Clone the Repository**
 ```bash
git clone <repository-url>
cd depression-prediction-app
 ```

 2. **Install Dependencies**
```bash
pip install streamlit scikit-learn pandas joblib
 ```

 3. **Ensure Required Files are Present**
 

 app.py — main application

 pipeline.pkl — trained ML pipeline

 df.pkl — reference dataframe for validation
 

## Usage

Run the Streamlit app:

streamlit run app.py


The app will open at http://localhost:8501

### Input Features

### The model requires the following inputs:

- Gender: Male/Female

- Academic Pressure: Yes/No

- Study Satisfaction: Yes/No

- Sleep Duration: 0–24 hours

- Dietary Habits: Healthy/Moderate/Unhealthy/Others

- Degree: BSc, B.Tech, MBA, etc.

- Study Hours: 0–24 hours

- Financial Stress: Yes/No

- Age Category: 0–18, 19–25, 26–30, 30+

## Feature Importance

Top predictors based on feature selection:

- Academic Pressure – 29.5%

- Financial Stress – 16.4%

- Study Hours – 13.4%

- Age – 13.2%

## Model Development

- Data Processing

- Label Encoding for categorical data

- Standard Scaling for numerical features

- Full ML pipeline integrates preprocessing + model

- Feature Selection Techniques

- Correlation Analysis

- Random Forest Importance

- Gradient Boosting Importance

- Permutation Importance

- LASSO

- RFE

- Linear Regression Coefficients

- Models Evaluated

- Logistic Regression

- Decision Tree

- Random Forest

- Extra Trees

- Gradient Boosting (Best)

- AdaBoost

- SVC

- MLP

## Results
- Model	Accuracy:	F1 Score (Macro)

- Logistic Regression:	76.6% - 75.7%

- Gradient Boosting:	76.5% - 75.6%

- AdaBoost:	76.4% - 75.5%

- SVC:	75.8%	- 74.8%

## Important Notes

- This model is for screening only, not clinical diagnosis

- Age was dropped in final model due to multicollinearity

- App auto-installs scikit-learn if missing

- Model version compatibility handled through error messages

### Contributing

Contributions are welcome!

Please submit a Pull Request.

### Author

Hitesh Sharma

### Acknowledgments

Dataset sourced from student mental health surveys

- scikit-learn for ML algorithms

- Streamlit for the interactive UI

### Support

For questions or issues, please open an issue in the repository.
