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
    # Load the dataset
    df = pd.read_csv(filepath, parse_dates=['Orgin date'])
    df = df.sort_values('Orgin date').reset_index(drop=True)
    return df

def compute_energy_budget(df, window_days=30):
    df = df.copy()

    # Ensure datetime index
    df = df.set_index('Orgin date')

    # Global average energy release rate (per day)
    total_energy = df['Energy'].sum()
    total_days = (df.index.max() - df.index.min()).days
    average_energy_per_day = total_energy / total_days

    # Rolling sum over the last `window_days` days
    df['RollingEnergyWindow'] = df['Energy'].rolling(f'{window_days}D', min_periods=1).sum()

    # Expected energy over the window
    expected_energy = average_energy_per_day * window_days

    # Deviation and ratio
    df['EnergyBudgetDeviation'] = df['RollingEnergyWindow'] - expected_energy
    df['EnergyBudgetRatio'] = df['RollingEnergyWindow'] / expected_energy

    # Restore original index
    df = df.reset_index()

    return df

def compute_features(df, ref_point=(24.785, 121.006)):
    df = df.copy()

    # Ensure datetime type
    df['Orgin date'] = pd.to_datetime(df['Orgin date'])

    # Step 1: Filter to start from first mag 6.5+
    first_major_event_idx = df[df['Magnitude'] >= 6.5].index.min()
    df = df.loc[first_major_event_idx:].reset_index(drop=True)

    # Step 2: Inter-event time in days
    df['InterEventTime'] = df['Orgin date'].diff().dt.total_seconds().div(86400).fillna(0)

    # Step 3: Haversine distance from reference point
    R = 6371  # Earth radius in km
    lat1 = np.radians(df['Latitude(N)'])
    lon1 = np.radians(df['Longitude(E)'])
    lat2 = np.radians(ref_point[0])
    lon2 = np.radians(ref_point[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    df['DistanceFromRef'] = R * c

    # Step 4: Time since last Mag3+, Mag5+, Mag6.5+
    df['TimeSinceMag3'] = np.zeros(len(df))
    df['TimeSinceMag5'] = np.zeros(len(df))
    df['TimeSinceMag65'] = np.zeros(len(df))

    last_mag3_time = None
    last_mag5_time = None
    last_mag65_time = None

    for idx, row in df.iterrows():
        current_time = row['Orgin date']

        # Time since last magnitude 3+
        df.at[idx, 'TimeSinceMag3'] = (current_time - last_mag3_time).total_seconds() / 86400 if last_mag3_time else 0
        if row['Magnitude'] >= 3.0:
            last_mag3_time = current_time

        # Time since last magnitude 5+
        df.at[idx, 'TimeSinceMag5'] = (current_time - last_mag5_time).total_seconds() / 86400 if last_mag5_time else 0
        if row['Magnitude'] >= 5.0:
            last_mag5_time = current_time

        # Time since last magnitude 6.5+
        df.at[idx, 'TimeSinceMag65'] = (current_time - last_mag65_time).total_seconds() / 86400 if last_mag65_time else 0
        if row['Magnitude'] >= 6.5:
            last_mag65_time = current_time


    df['Energy'] = 10 ** (1.5 * df['Magnitude'] + 4.8)
    df=compute_energy_budget(df)
    
    
    # Step 5: Select features
    features = df[['InterEventTime', 'Depth', 'Magnitude','TimeSinceMag65','Energy','DistanceFromRef']].fillna(0) #, 'DistanceFromRef'

    return df,features

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
        
        return log_likelihood, aic, bic
        
    except Exception as e:
        print(f"Error in compute_aic_bic: {e}")
        return -np.inf, np.inf, np.inf

def compute_states(best_model, df):
    # Compute features
    df, features = compute_features(df, ref_point=(24.785, 121.006))
    X = features.values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Predict the hidden states
    hidden_states = best_model.predict(X)

    # Look at the time series of the hidden states
    plt.figure(figsize=(10, 6))
    plt.plot(hidden_states[:100], label='Hidden States')
    plt.title('Hidden States Over Time')
    plt.xlabel('Time')
    plt.ylabel('Hidden State')
    plt.legend()
    plt.savefig(f'{OUTPUTS_DIR}/hidden_states_over_time.png')
    # plt.show()


    # Save the hidden states to a CSV file
    hidden_states_df = pd.DataFrame(hidden_states, columns=['Hidden State'])
    hidden_states_df.to_csv(f'{OUTPUTS_DIR}/hidden_states.csv', index=False)


    # Generate forecasted values based on the hidden states
    forecasted_values = best_model.means_[hidden_states]

    return hidden_states, features, forecasted_values

def evaluate_model(best_model, hidden_states, features):

    log_likelihood, aic, bic = compute_aic_bic(best_model, features.values)

    # Print evaluation metrics
    print(f"Log-Likelihood: {log_likelihood}")
    print(f"AIC: {aic}")
    print(f"BIC: {bic}")

    ####### Categorise the X and check against forecasted values ###################

    means = best_model.means_
    covariances = best_model.covars_

    # Define thresholds based on means and covariances
    thresholds = []
    for i in range(best_model.n_components):
        mean = means[i]
        covariance = covariances[i]
        std_dev = np.sqrt(np.diag(covariance))

        # Define thresholds as mean ± 1 standard deviation
        lower_threshold = mean - 2*std_dev
        upper_threshold = mean + 2*std_dev
        thresholds.append(list(zip(lower_threshold, upper_threshold)))

    # categorise into states based on how much % of thresholds are met
    def categorize_into_states(X, thresholds, criteria=0.8):
        states = np.full(X.shape[0], -3, dtype=int) # negative inf are not within any accepted bounds 
        for i, x in enumerate(X):
            for state, threshold in enumerate(thresholds):
                met = 0
                for xi, (low, high) in zip(x, threshold):
                    if low <= xi <= high: met += 1
                    if met >= len(threshold) * criteria:
                        states[i] = state
                        break
        return states

    X = features.values

    # plot accuracies
    critiera = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    accuracies = []
    unassigned_states = []

    for criteria in critiera:
        actual_states = categorize_into_states(X, thresholds, criteria)
        accuracy = np.sum(actual_states == hidden_states) 
        no_states = np.sum(actual_states == -3)

        accuracies.append(accuracy)
        unassigned_states.append(no_states)

    # Plot of the accuracies, and unassigned states
    plt.figure(figsize=(10, 6))
    plt.title(f'Categorization Accuracy vs Criteria ({X.shape[0]} states)')
    plt.plot(critiera, accuracies, label='Accuracy', marker='o')
    plt.plot(critiera, unassigned_states, label='Unassigned States', marker='s')
    plt.xlabel('Criteria')
    plt.ylabel('Percentage')
    plt.xticks(critiera)
    plt.legend()
    plt.savefig(f'{OUTPUTS_DIR}/categorization_accuracy_vs_criteria.png')
    # plt.show()

    best_accuracy_index = np.argmax(accuracies)
    best_criteria = critiera[best_accuracy_index]
    best_states = categorize_into_states(X, thresholds, best_criteria)

    # Save the states comparison
    values = pd.DataFrame(best_states, columns=['Categorized State'])
    values['Hidden State'] = hidden_states
    values.to_csv(f'{OUTPUTS_DIR}/state_comparison.csv', index=False)

    output = values['Categorized State']
    expected = values['Hidden State']

    report = classification_report(expected, output, output_dict=True)

    with open(f'{OUTPUTS_DIR}/model_evaluation.txt', 'w') as f:
        f.write(f"\n\nLog-Likelihood: {log_likelihood}\n")
        f.write(f"AIC: {aic}\n")
        f.write(f"BIC: {bic}\n")
        f.write(f"\nConfusion Matrix:\n{confusion_matrix(expected, output)}\n")
        f.write(f"\n\nWeighted Average Classification Report:\n{report['weighted avg']}\n")


######## DO VALIDATION ########

def perform_validation(best_model, df):

    # Define severe event condition
    df['SevereEvent'] = ((df['Magnitude'] >= 6) & (df['Depth'] <= 100)) | ((df['Magnitude'] >= 5.5) & (df['Depth'] <= 10))

    hidden_states, features, forecasted_values = compute_states(best_model, df)

    # Calculate Mean Squared Error
    mse = mean_squared_error(features.values, forecasted_values)

    print(f"Mean Squared Error: {mse}")

    for i, feature in enumerate(features.columns):
        # Plot the actual vs forecasted values
        plt.figure(figsize=(10, 6))
        plt.plot(features.values[:, i], label='Actual Values')  # Assuming the first feature is the one you want to plot
        plt.plot(forecasted_values[:, i], label='Forecasted Values', linestyle='--')
        plt.legend()
        plt.title(f'Actual vs Forecasted Values {i}\'th Feature')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.savefig(f'{OUTPUTS_DIR}/actual_vs_forecasted/{feature}_{i}.png')

    evaluate_model(best_model, hidden_states, features)
    

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python HMM_multivariate_test.py <model_type>")
        sys.exit(1)

    model_type = sys.argv[1].lower()

    if model_type == 'gauss':
        INPUTS_DIR = "../training/results_multivariate/Gaussian"
        OUTPUTS_DIR = "./results_multivariate/Gaussian"
    elif model_type == 'pois':
        raise NotImplementedError("Poisson model is not implemented")
    else:
        raise ValueError("Unknown model type, emissions must be Gaussian, Poisson")


    best_model = joblib.load(f"{INPUTS_DIR}/best_hmm_model.joblib")
    data = load_data('../data/training_earthquakes.csv')
    scaler = joblib.load(f"{INPUTS_DIR}/scaler.joblib")
    _, X = compute_features(data, ref_point=(24.785, 121.006))
    X = scaler.transform(X.values)
    print(compute_aic_bic(best_model, X))


