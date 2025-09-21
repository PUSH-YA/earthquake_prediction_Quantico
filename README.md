# Predict earthquakes
Work for a Quantitative research club at UBC

## 3 ideas:
- Classic ML, random forests by nature but not good 
- HMM -- what i am working on 
- GEV -- Tarek's work
- PHSA -- currently what is happening in the market


## HMM notes:

### Model types 
Trying for Gaussian and Poisson emissions
- Gaussian cuz normal
  - model 
- Poisson cuz earthquakes modelled as poisson events


### Covariance assumption 
Changed covaraince type:
- from `full`
- to `diag`
- allows for less overfitting and matrix for more than 6 states were not possible with it . 

### n_state choices 
States:
- trying all from 3 states minimum to 9
