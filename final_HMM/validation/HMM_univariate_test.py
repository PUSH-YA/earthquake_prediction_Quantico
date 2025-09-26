import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix 
from sklearn.preprocessing import StandardScaler 
import sys 
import os


OUTPUTS_DIR=""

########### HELPERS #########

def load_data(filepath):
    df = pd.read_csv(filepath)

    years = pd.to_datetime(df['Orgin date']).dt.year
    months = pd.to_datetime(df['Orgin date']).dt.month
    days = pd.to_datetime(df['Orgin date']).dt.day

    dates = pd.DataFrame({'years': years, 'months': months, 'days': days})
    counts = dates.groupby(['years', 'months', 'days']).size()

    earthquakes = np.array(counts).reshape(-1,1)

    return earthquakes

def compute_aic_bic(model, X):
    try:
        log_likelihood = model.score(X)
        
        # Check for NaN values
        if np.isnan(log_likelihood):
            print("Warning: NaN log likelihood detected")
            print("Startprob:", model.startprob_)
            print("Lambdas:", model.lambdas_)
            print("Transmat:", model.transmat_)
            return -np.inf, np.inf, np.inf
        
        n_params = model.n_features * model.n_components  # For Poisson means
        n_params += model.n_components * (model.n_components - 1)  # Transition matrix
        n_params += model.n_components - 1  # Initial probabilities
        
        n_samples = len(X)
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n_samples)

        with open(f"{OUTPUTS_DIR}/model_metrics.txt", "w") as f:
            f.write(f"Log Likelihood: {log_likelihood}\n")
            f.write(f"AIC: {aic}\n")
            f.write(f"BIC: {bic}\n")
        
        return log_likelihood, aic, bic
        
    except Exception as e:
        print(f"Error in compute_aic_bic: {e}")
        return -np.inf, np.inf, np.inf    
    


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python HMM_multivariate_test.py <model_type>")
        sys.exit(1)

    model_type = sys.argv[1].lower()

    if model_type == 'gauss':
        INPUTS_DIR = "../training/results_univariate/Gaussian"
        OUTPUTS_DIR = "./results_univariate/Gaussian"
    elif model_type == 'pois':
        INPUTS_DIR = "../training/results_univariate/Poisson"
        OUTPUTS_DIR = "./results_univariate/Poisson"
    else:
        raise ValueError("Unknown model type, emissions must be Gaussian, Poisson")

    earthquakes = load_data('../data/testing_earthquakes.csv')
    best_model = joblib.load(f"{INPUTS_DIR}/best_model.joblib")
    log_likelihood, aic, bic = compute_aic_bic(best_model, earthquakes)

    with open(f"{OUTPUTS_DIR}/model_evaluation.txt", "w") as f:
        f.write(f"Log-Likelihood: {log_likelihood}\n")
        f.write(f"AIC: {aic}\n")
        f.write(f"BIC: {bic}\n")
    


