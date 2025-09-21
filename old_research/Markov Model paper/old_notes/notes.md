# $HMM$ for earthquake prediction

- they are using a bernoulli distribution to model the $MM$ where $\{S\}_{2 \times 2}$ is either the earthquake happened or not. 
- They add $T_n$ as the covariate in the $HMM$ to denote a sense of time in the models. 
- They use $EM$ to maximise the likelihoods and get the $\widehat{\theta}$ (which are different covaraites)
- They use Viterbi algorithm to find the most probable states and then use that to predict 
- The proposed model is capable
of predicting the change-in-state of the hidden Markov chain, and thus can predict the arrival time and magnitude of future earthquake occurrences simultaneously
	- model is stationary and ergodic markov chain with asymptotic normality, which is required for HMM and another thing to work.
	

# $HMM$ for earthquake declustering

- ETAS model: branching process model for immigrants
- earthquake occurs in clusters and this would be focusing more on declustring the type of earhtquake
- This would be to find the distribution of $\mathbb{P}(h_i | O_1, \dots, O_n)$
- Most model talking about $HMM$ also use poisson process for determining the waiting times between the earthquakes
- Note that the HMM does not incorporate any infor-
mation of the earthquake magnitude, only the locations of the epicenters.


## Possible idea:
- States with different risks, less correlated
- Observations: seismic activity, time covariate, geochecmical signals, ground deformation
- Training HMM: 
	- $\text{Baum Welch}\implies$ learn $\widetilde{P}_{ij}$ ($EM$ for $HMM$)
	- $\text{Viterbi }\implies$ find most likely states and then predict based on past $n$ states
	- $MCMC \implies$ We can use forward backward algorithm for it
	
	$$\mathbb{P}(z_{1:t}|X_{1:t}, A, B, \pi) \propto \mathbb{P}(X_{1:t}|z_{1:t}, B) \mathbb{P}(z_{1:t}|A, \pi)$$


We can have transition matrix as following:

$$A_i|z_{1:t} \sim \text{Dirichlet}(\alpha_i + n_i)$$

Gaussian emission parameters:

### Step 1: Sample Hidden States $z_{1:T}$

Use the Forward-Backward algorithm to compute the posterior distribution of $z_t$ and sample a state sequence:

$$P(z_{1:T} \mid x_{1:T}, A, B, \pi) \propto P(x_{1:T} \mid z_{1:T}, B) P(z_{1:T} \mid A, \pi)$$

### Step 2: Sample Parameters $A, B, \pi$

**Transition Matrix $A$**:

For each row $A_i$, sample from the Dirichlet posterior:
  
$$A_i \mid z_{1:T} \sim \text{Dirichlet}(\alpha_i + n_i)$$

  where $n_i$ counts transitions from state $i$.

- **Initial Distribution $\pi$**:
  
$$\pi \mid z_1 \sim \text{Dirichlet}(\beta + m)$$
  
  where $m$ counts initial state assignments.

- **Emission Parameters $B$**:
  For Gaussian emissions, sample $\mu_j, \sigma_j^2$ using conjugate priors:
  
$$\mu_j \mid x_{z_t = j} \sim N\left(\frac{\mu_0 / \sigma_0^2 + \sum x_t / \sigma_j^2}{1 / \sigma_0^2 + n_j / \sigma_j^2}, \cdots \right)$$

### Step 3: Repeat

Iterate Steps 1–2 for thousands of iterations to approximate the posterior.