from pyexpat import model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.utils import shuffle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import joblib
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler 
import sys 
import os


OUTPUTS_DIR = "./results_multivariate/Gaussian"

# =============================
# 1. Data Preparation Functions
# =============================

def load_earthquake_data(filepath):
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


# =============================
# 2. HMM Training and Selection Functions
# =============================

def train_hmm(model_type, X, n_components=3, n_iter=100):
    if model_type == 'gauss':
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=n_iter, random_state=42)
    else:
        model = hmm.PoissonHMM(n_components=n_components, n_iter=n_iter, random_state=42)

    model.fit(X)
    return model

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

def select_best_model(model_type, X, min_states=2, max_states=15):
    results = []
    models = []
    for n in range(min_states, max_states + 1):
        
        # Split data into training and validation sets
        train_size = int(0.8 * len(X))
        X_train = X[:train_size]
        X_val = X[train_size:]
        
        model = train_hmm(model_type, X_train, n_components=n)
        
        # Compute metrics on validation sets
        val_log_likelihood, val_aic, val_bic = compute_aic_bic(model, X_val)
        
        results.append({
            'n_states': n, 
            'log_likelihood': val_log_likelihood, 
            'aic': val_aic, 
            'bic': val_bic
        })
        models.append(model)
        print(f"Tested {n} states: Train LogLik = {val_log_likelihood:.2f}, Val LogLik = {val_log_likelihood:.2f}, Val BIC = {val_bic:.2f}")

    results_df = pd.DataFrame(results)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['n_states'], results_df['log_likelihood'], label='Log Likelihood', marker='o')
    plt.plot(results_df['n_states'], results_df['aic'], label='AIC', marker='x')
    plt.plot(results_df['n_states'], results_df['bic'], label='BIC', marker='^')
    plt.xlabel('Number of States')
    plt.ylabel('Score')
    plt.title('Model Selection: Log-likelihood, AIC, BIC by Number of States')
    plt.legend()
    plt.grid(True)
    

    # Select best model by BIC
    best_idx = results_df['bic'].idxmin()
    best_model = models[best_idx]
    best_n_states = results_df.loc[best_idx, 'n_states']

    print(f"\nBest model selected: {best_n_states} states (BIC = {results_df.loc[best_idx, 'bic']:.2f})")
    return best_model, best_n_states, results_df

# =============================
# 3. State Mapping and Transition Analysis
# =============================

def map_states_to_labels(model, X, severe_event_mask):
    states = model.predict(X)
    state_mapping = {}
    
    # For each state, compute overlap with SevereEvent mask
    for state in range(model.n_components):
        state_mask = (states == state)
        overlap = np.sum(state_mask & severe_event_mask)
        total = np.sum(state_mask)
        proportion = overlap / total if total > 0 else 0
        state_mapping[state] = proportion

    # Find state with highest overlap
    severe_state = max(state_mapping, key=state_mapping.get)
    print(f"\nState mapping (proportion of SevereEvent): {state_mapping}")
    print(f"Identified SevereEvent state: {severe_state}")
    return severe_state, states

def analyze_transitions(model, states, severe_state):
    transmat = model.transmat_
    print("\n=== Transition Probabilities ===")
    for i in range(model.n_components):
        print(f"From State {i}: {np.round(transmat[i], 3)}")

    print(f"\nTransitions from SevereEvent state ({severe_state}): {np.round(transmat[severe_state], 3)}")
    print(f"Transitions to SevereEvent state ({severe_state}): {np.round(transmat[:, severe_state], 3)}")

# =============================
# 4. Feature Importance Functions
# =============================

def analyze_emission_parameters(model, feature_names):
    means = model.means_
    covars = model.covars_
    
    print("\n=== Emission Means per State ===")
    for idx, mean in enumerate(means):
        print(f"State {idx}: {dict(zip(feature_names, mean))}")

    print("\n=== Emission Variances per State ===")
    for idx, covar in enumerate(covars):
        variances = np.diag(covar)
        print(f"State {idx}: {dict(zip(feature_names, variances))}")

def permutation_importance(model, X, feature_names):
    baseline_score = model.score(X)
    print(f"\nBaseline log likelihood: {baseline_score:.2f}")

    for i, name in enumerate(feature_names):
        X_permuted = X.copy()
        X_permuted[:, i] = shuffle(X_permuted[:, i], random_state=42)
        score = model.score(X_permuted)
        print(f"Permuting {name}: log likelihood = {score:.2f}, impact = {baseline_score - score:.2f}")

# =============================
# 5. Visualization Functions
# =============================

def plot_state_sequence(model, X):
    states = model.predict(X)
    plt.figure(figsize=(12, 3))
    plt.plot(states, marker='o', linestyle='-')
    plt.title('Predicted Hidden States over Time')
    plt.xlabel('Event Index')
    plt.ylabel('Hidden State')
    

def export_states_to_csv(df, model, X, write_csv=False, output_path=None):
    if output_path is None:
        output_path = f'{OUTPUTS_DIR}/earthquake_states_output.csv'
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    states = model.predict(X)
    df_out = df.copy()
    df_out['HiddenState'] = states
    if write_csv:
        df_out.to_csv(output_path, index=False)
    print(f"Exported state assignments to {output_path}")



# =============================
# 7. Visualization: Geographic Map of States
# =============================

def plot_geographic_states(df, output_path=f'{OUTPUTS_DIR}/earthquake_geographic_states.png'):
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        df['Longitude(E)'], df['Latitude(N)'], c=df['HiddenState'], cmap='tab20', s=10, alpha=0.7
    )
    plt.colorbar(scatter, label='Hidden State')
    plt.title('Earthquake Events by Hidden State')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output_path, dpi=300)
    
    print(f"Geographic state plot saved to {output_path}")

# =============================
# 8. Summary Table Automation
# =============================

def generate_summary_table(df, write_csv=True, output_path=f'{OUTPUTS_DIR}/state_summary_table.csv'):
    summary = df.groupby('HiddenState').agg(
        Counts=('HiddenState', 'count'),
        Min_Magnitude=('Magnitude', 'min'), Avg_Magnitude=('Magnitude', 'mean'), Max_Magnitude=('Magnitude', 'max'),
        Min_Depth=('Depth', 'min'), Avg_Depth=('Depth', 'mean'), Max_Depth=('Depth', 'max'),
        Min_Distance=('DistanceFromRef', 'min'), Avg_Distance=('DistanceFromRef', 'mean'), Max_Distance=('DistanceFromRef', 'max'),
        Min_Time=('InterEventTime', 'min'), Avg_Time=('InterEventTime', 'mean'), Max_Time=('InterEventTime', 'max'),
        #Avg_Energy=('Avg_Energy','mean'),Min_Energy=('Min_Energy','min'),Max_Energy=('Max_Energy','max'),
        Num_Serious_Events=('SevereEvent', 'sum')
    ).reset_index()

    if write_csv:
        summary.to_csv(output_path, index=False)
        print(f"Summary table saved to {output_path}")

    # Optional: show table visually
    print(summary)
    return summary

# =============================
# 9. Transition Probability Heatmap
# =============================

def plot_transition_matrix(model,write_csv=False,output_path_csv=f'{OUTPUTS_DIR}/transition_values.csv', output_path=f'{OUTPUTS_DIR}/transition_matrix_heatmap.png'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(model.transmat_, annot=True, fmt='.2f', cmap='Blues', cbar=True)
    plt.title('HMM Transition Probability Matrix')
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.savefig(output_path, dpi=300)
    
    print(f"Transition matrix heatmap saved to {output_path}")
    
    
    n_states = model.transmat_.shape[0]
    transmat_df = pd.DataFrame(
        model.transmat_,
        index=[f'From_State_{i}' for i in range(n_states)],
        columns=[f'To_State_{j}' for j in range(n_states)]
    )
    if write_csv:
        transmat_df.to_csv(output_path_csv, index=True)
        print(f"Transition matrix exported to {output_path_csv}")
# =============================
# Update main() to include these
# =============================

def main(model_type):

    # Load and process data
    filepath = '../data/training_earthquakes.csv'
    df = load_earthquake_data(filepath)

    # Define severe event condition
    df['SevereEvent'] = ((df['Magnitude'] >= 6) & (df['Depth'] <= 100)) | ((df['Magnitude'] >= 5.5) & (df['Depth'] <= 10))

    df,features = compute_features(df, ref_point=(24.785, 121.006))
    feature_names = features.columns.tolist()
    X = features.values

    # After getting raw features, scale:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, f'{OUTPUTS_DIR}/scaler.joblib')

    # Model selection: automatic determination of number of states
    best_model, best_n_states, results_df = select_best_model(model_type, X, min_states=19, max_states=19)

    # =======================================
    # trained model is saved offline
    joblib.dump(best_model, f'{OUTPUTS_DIR}/best_hmm_model.joblib')

    # Save the best number of states
    with open(f'{OUTPUTS_DIR}/best_n_states.txt', 'w') as f:
        f.write(str(best_n_states))

    # Save the results DataFrame
    results_df.to_csv(f'{OUTPUTS_DIR}/model_selection_results.csv', index=False)

    print("Best model, number of states, and model selection results saved offline.")
    # ========================================

    # Map HMM states to SevereEvent
    severe_state, states = map_states_to_labels(best_model, X, df['SevereEvent'].values)

    # Add hidden states to dataframe for export and analysis
    df['HiddenState'] = states

    # Analyze transitions relative to the SevereEvent state
    analyze_transitions(best_model, states, severe_state)

    # Emission analysis
    analyze_emission_parameters(best_model, feature_names)

    # Permutation importance
    permutation_importance(best_model, X, feature_names)

    # Plot state sequence
    plot_state_sequence(best_model, X)

    # Export states to CSV
    export_states_to_csv(df, best_model, X,write_csv=True)

    # New: Geographic visualization
    plot_geographic_states(df)

    # New: Generate summary table
    generate_summary_table(df,write_csv=True)

    # New: Transition matrix heatmap
    plot_transition_matrix(best_model,write_csv=True)
    

if __name__ == "__main__":

    model_type = sys.argv[1].lower()

    if model_type == 'gauss':
        OUTPUTS_DIR = "results_multivariate/Gaussian"
    elif model_type == 'pois':
        raise NotImplementedError("Poisson HMM Multivariate is not implemented yet.")
    else:
        raise ValueError("Unknown model type, emissions must be Gaussian, Poisson")
        
    main(model_type)
