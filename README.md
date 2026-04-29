# STAT3013.Q21_Group06

## FORECASTING HOSPITAL OVERLOAD FROM AIR POLLUTION: A STATISTICAL AND MACHINE LEARNING APPROACH ACROSS THREE GLOBAL CITIES
This project investigates the statistical relationship between ambient air pollution (PM2.5, PM10, O3, NO2, SO2, CO) and daily hospital admissions across three global capitals: Delhi, London, and Mexico City. We implement and evaluate 6 different models including Statistical approaches (OLS, SEM, ARIMAX) and Machine Learning/Deep Learning algorithms (SVR, QRF, TCN) to accurately forecast healthcare demand based on environmental factors.

## Dataset and Video Links
* **Dataset:** [Download Project Dataset (Google Drive)](https://drive.google.com/file/d/1CjTipIj2e3HRNtmDo20Sasb5Tj4ULlwd/view?usp=sharing)
* **Presentation Video:** [Watch Project Demo (Google Drive)](https://drive.google.com/file/d/1_RINQ5_SkSLiUNbUNPLt2HW71WGLd3_k/view?usp=sharing)

## Env Requirements
The project is built using Python 3.12+. To run the complete pipeline across all 6 models, you must install the following dependencies. 

Create a `requirements.txt` file or install them directly:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
pip install pmdarima         # For ARIMAX model
pip install quantile-forest  # For QRF (Quantile Regression Forest) model
pip install semopy           # For SEM (Structural Equation Modeling)
pip install tensorflow       # For TCN (Temporal Convolutional Network)
