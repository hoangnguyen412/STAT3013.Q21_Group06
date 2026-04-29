# STAT3013.Q21_Group06

## FORECASTING HOSPITAL OVERLOAD FROM AIR POLLUTION: A STATISTICAL AND MACHINE LEARNING APPROACH ACROSS THREE GLOBAL CITIES
This project investigates the statistical relationship between ambient air pollution (PM2.5, PM10, O3, NO2, SO2, CO) and daily hospital admissions across three global capitals: Delhi, London, and Mexico City. We implement and evaluate 6 different models including Statistical approaches (OLS, SEM, ARIMAX) and Machine Learning/Deep Learning algorithms (SVR, QRF, TCN) to accurately forecast healthcare demand based on environmental factors.

## Dataset and Video Links
* **Dataset:** [Download Project Dataset (Google Drive)](https://drive.google.com/file/d/1CjTipIj2e3HRNtmDo20Sasb5Tj4ULlwd/view?usp=sharing)
* **Presentation Video:** [Watch Project Demo (Google Drive)](https://drive.google.com/file/d/1_RINQ5_SkSLiUNbUNPLt2HW71WGLd3_k/view?usp=sharing)

## Env Requirements
The project is built using Python 3.12+. To run the complete pipeline across all 6 models, you must install the following dependencies. 

Create a `requirements.txt` file or install them directly:

pip install pandas numpy matplotlib seaborn scikit-learn statsmodels

pip install pmdarima         # For ARIMAX model

pip install quantile-forest  # For QRF (Quantile Regression Forest) model

pip install semopy           # For SEM (Structural Equation Modeling)

pip install tensorflow       # For TCN (Temporal Convolutional Networkl


Run Instructions
Ensure that the dataset files (merged_project_data.csv, delhi_project_final_data.csv, london_project_final_data.csv, mexico_project_final_data.csv) are placed in the root directory of the project.

Follow these steps to replicate the analysis for each model:

1. Statistical Models:

OLS (Multiple Linear Regression): Run python ols_model.py. This script performs chronological train/test splits, one-hot encodes city data, and prints the summary coefficients along with test set metrics (RMSE, MAE, R², MAPE).

SEM (Structural Equation Modeling): Run python sem_model.py. This analyzes the mediation effect of AQI between PM2.5 and hospital admissions. It will output path coefficients and generate 3 visualization figures (sem_path_diagrams.png, etc.).

ARIMAX (Time-Series Forecasting): Run python arimax_model.py. This performs ADF stationarity tests, automatically finds the optimal order (including seasonal factors m=7), prints accuracy metrics, and saves the forecast visualization as arimax_full_report_plot.png.

2. Machine Learning & Deep Learning Models:

SVR (Support Vector Regression): Run python svr_model.py. This script standardizes the features, applies a linear kernel SVR, and generates time series comparison plots, scatter plots, and permutation feature importance charts.

QRF (Quantile Regression Forest): Run python qrf_model.py. This evaluates non-linear relationships, predicts the 10th, 50th, and 90th percentiles, and provides robust 80% prediction interval coverage charts alongside Top 15 Feature Importance plots.

TCN (Temporal Convolutional Network): Run python tcn_model.py. This Deep Learning model processes 7-day lookback sequences (LOOKBACK=7) utilizing dilated convolutions to predict admissions. It outputs the training loss history and test set performance plots.

License
This project is licensed under the MIT License - see the LICENSE file for details.

