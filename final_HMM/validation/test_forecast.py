import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


########### HELPERS #########

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


######## DO VALIDATION ########

# Load the trained HMM model
best_model = joblib.load('results/best_hmm_model.joblib')

# Load the dataset
filepath = 'All Earthquakes.csv'
df = pd.read_csv(filepath, parse_dates=['Orgin date'])
df = df.sort_values('Orgin date').reset_index(drop=True)

# Define severe event condition
df['SevereEvent'] = ((df['Magnitude'] >= 6) & (df['Depth'] <= 100)) | ((df['Magnitude'] >= 5.5) & (df['Depth'] <= 10))

# Compute features
df, features = compute_features(df, ref_point=(24.785, 121.006))
X = features.values

# Predict the hidden states
hidden_states = best_model.predict(X)

# Look at the time series of the hidden states
plt.figure(figsize=(10, 6))
plt.plot(hidden_states[:100], label='Hidden States')
plt.title('Hidden States Over Time')
plt.xlabel('Time')
plt.ylabel('Hidden State')
plt.legend()
plt.savefig('validation/hidden_states_over_time.png')
# plt.show()


# Save the hidden states to a CSV file
hidden_states_df = pd.DataFrame(hidden_states, columns=['Hidden State'])
hidden_states_df.to_csv('validation/hidden_states.csv', index=False)


# Generate forecasted values based on the hidden states
forecasted_values = best_model.means_[hidden_states]

# Calculate Mean Squared Error
mse = mean_squared_error(X, forecasted_values)

print(f"Mean Squared Error: {mse}")

for i, feature in enumerate(features.columns):
    # Plot the actual vs forecasted values
    plt.figure(figsize=(10, 6))
    plt.plot(X[:, i], label='Actual Values')  # Assuming the first feature is the one you want to plot
    plt.plot(forecasted_values[:, i], label='Forecasted Values', linestyle='--')
    plt.legend()
    plt.title(f'Actual vs Forecasted Values {i}\'th Feature')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig(f'validation/actual_vs_forecasted/feature_{i}.png')
    # plt.show()

# Save forecasted values to a CSV file
forecasted_df = pd.DataFrame(forecasted_values, columns=features.columns)
forecasted_df.to_csv('validation/forecasted_values.csv', index=False)

# Save the comparison results
comparison_results = pd.DataFrame({
    'Actual': X[:, 0],
    'Forecasted': forecasted_values[:, 0]
})
comparison_results.to_csv('validation/comparison_results.csv', index=False)


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

# Print the thresholds for each state
for i, threshold in enumerate(thresholds):
    print(f"Thresholds for State {i}:")
    for j, (low, high) in enumerate(threshold):
        print(f"  Feature {j}: [{low:.2f}, {high:.2f}]")

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
accuracies_within_1_state = []
unassigned_states = []

for criteria in critiera:
    actual_states = categorize_into_states(X, thresholds, criteria)
    accuracy = np.sum(actual_states == hidden_states) 
    accuracy_within_1_state = np.sum(np.abs(actual_states - hidden_states) <= 1)
    no_states = np.sum(actual_states == -3)

    accuracies.append(accuracy)
    accuracies_within_1_state.append(accuracy_within_1_state)
    unassigned_states.append(no_states)

# Plot of the accuracies, accuracies within 1 state, and unassigned states
plt.figure(figsize=(10, 6))
plt.title(f'Categorization Accuracy vs Criteria ({X.shape[0]} states)')
plt.plot(critiera, accuracies, label='Accuracy', marker='o')
plt.plot(critiera, accuracies_within_1_state, label='Accuracy within 1 state', marker='x')
plt.plot(critiera, unassigned_states, label='Unassigned States', marker='s')
plt.xlabel('Criteria')
plt.ylabel('Percentage')
plt.xticks(critiera)
plt.legend()
plt.savefig('validation/categorization_accuracy_vs_criteria.png')
# plt.show()

best_accuracy_index = np.argmax(accuracies)
best_criteria = critiera[best_accuracy_index]
best_states = categorize_into_states(X, thresholds, best_criteria)

# Save the states comparison
values = pd.DataFrame(best_states, columns=['Categorized State'])
values['Hidden State'] = hidden_states
values.to_csv('validation/state_comparison.csv', index=False)

output = values['Categorized State']
expected = values['Hidden State']

matched_states = np.sum(expected == output)
matched_within_1_state = np.sum(np.abs(expected - output) <= 1)
print(f'Matched states : {matched_states}; \t Accuracy: {matched_states / len(expected):.2%}')
print(f'Matched within 1 state: {matched_within_1_state}; \t Accuracy within 1 state: {matched_within_1_state / len(output):.2%}')

np.sum(output == -3) / len(output) * 100



