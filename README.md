# Predict earthquakes
Work for a Quantitative research club at UBC

## 3 ideas:
- Classic ML, random forests by nature but not good 
  - this will involve both multivariate and univariate model
- HMM -- what i am working on 
- GEV -- Tarek's work
- PHSA -- currently what is happening in the market


## HMM notes:

### Model types 
Trying for Gaussian and Poisson emissions
- Gaussian cuz normal 
- Poisson cuz earthquakes modelled as poisson events
  - $\lambda>0$ and thus has more constraints
  - Poisson $\mathcal{L}$ can underflow more easily and need more data constraints
  - Poisson multivariate is not converging and getting $\lambda = \infty$ 


### Covariance assumption 
Changed covaraince type:
- from `full`
- to `diag`
- allows for less overfitting and matrix for more than 6 states were not possible with it . 

### n_state choices 
States:
- trying all from 3 states minimum to 9
