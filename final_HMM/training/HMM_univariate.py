import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import poisson
from hmmlearn import hmm
import sys
import joblib


OUTPUTS_DIR = '../results_univariate/Gaussian'

####### Load Data ################

def load_data(filepath):
    df = pd.read_csv(filepath)

    years = pd.to_datetime(df['Orgin date']).dt.year
    months = pd.to_datetime(df['Orgin date']).dt.month
    days = pd.to_datetime(df['Orgin date']).dt.day

    dates = pd.DataFrame({'years': years, 'months': months, 'days': days})
    counts = dates.groupby(['years', 'months', 'days']).size()

    earthquakes = np.array(counts).reshape(-1,1)

    return earthquakes

######### Functions for HMM ##################


def save_plots(model):

    transmat_int = (model.transmat_*100).astype(int)

    _, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(transmat_int, aspect='auto', cmap='viridis')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Transition Probability (%)', rotation=-90, va="bottom")
    for i in range(transmat_int.shape[0]):
        for j in range(transmat_int.shape[1]):
            ax.text(j, i, f'{transmat_int[i, j]}%', ha='center', va='center', color='white')
    ax.set_title('Transition Matrix')
    ax.set_xlabel('State To')
    ax.set_ylabel('State From')
    ax.set_xticks(np.arange(transmat_int.shape[1]))
    ax.set_yticks(np.arange(transmat_int.shape[0]))
    ax.set_xticklabels([f'State {i+1}' for i in range(transmat_int.shape[1])])
    ax.set_yticklabels([f'State {i+1}' for i in range(transmat_int.shape[0])])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/best_model_transition.png")


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


def train_hmm(earthquakes, model_type="gauss", n_iter=100, min_states=3, max_states=10):

    models = []
    scores = []
    best_n_components = None

    for n_components in range(min_states, max_states + 1):
        # define our hidden Markov model
        if model_type == "gauss":
            model = hmm.GaussianHMM(n_components=n_components, random_state=42,
                                    n_iter=n_iter)
        elif model_type == "pois":
            model = hmm.PoissonHMM(n_components=n_components, random_state=42,
                                    n_iter=n_iter)
        model.fit(earthquakes)
        models.append(model)
        scores.append(model.score(earthquakes))

        print(f'model n_components: {n_components}; Converged: {model.monitor_.converged}\t\t'
                f'Score: {scores[-1]}')

    # get the best model
    best_model = models[np.argmax(scores)]
    best_n_components = best_model.n_components
    print(f'The best model had a score of {max(scores)} and '
        f'{best_model.n_components} components')

    # use the Viterbi algorithm to predict the most likely sequence of states
    # given the model
    # states = best_model.predict(earthquakes)

    with open(f"{OUTPUTS_DIR}/best_n_components_states.txt", "w") as f:
        f.write(f"Best number of components: {best_n_components}\n")


    return best_model


########## Main ################

if __name__ == "__main__":

    model_type = sys.argv[1].lower()

    if model_type == 'gauss':
        OUTPUTS_DIR = "./results_univariate/Gaussian"
    elif model_type == 'pois':
        OUTPUTS_DIR = "./results_univariate/Poisson"
    else:
        raise ValueError("Unknown model type, emissions must be Gaussian, Poisson")


    earthquakes = load_data("./data/training_earthquakes.csv")
    best_model = train_hmm(earthquakes, model_type, max_states=20)
    joblib.dump(best_model, f"{OUTPUTS_DIR}/best_model.joblib")
    save_plots(best_model)
    compute_aic_bic(best_model, earthquakes)