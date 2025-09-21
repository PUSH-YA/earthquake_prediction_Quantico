functions {
  real gev_lpdf_vec(vector y, vector mu, vector sigma, real xi) {
    int N = num_elements(y);
    vector[N] z;
    vector[N] log_p;

    for (i in 1:N) {
      if (sigma[i] <= 0)
        reject("sigma must be positive; found sigma[", i, "] = ", sigma[i]);
    }

    z = (y - mu) ./ sigma;

    if (fabs(xi) < 1e-15) {
      log_p = -z - exp(-z) - log(sigma);
    } else {
      vector[N] argument = 1 + xi * z;
      for (i in 1:N) {
        if (argument[i] <= 0)
          reject("Invalid: 1 + xi * z[", i, "] <= 0; xi = ", xi, ", z[", i, "] = ", z[i]);
      }
      log_p = -(1 + 1 / xi) .* log(argument) - pow(argument, -1 / xi) - log(sigma);
    }

    return sum(log_p);
  }
}

data {
  int<lower=1> Nobs;
  int<lower=1> Nsite;
  vector[Nobs] x1; // this is the maximum earthquake magnitudes
  int<lower=1, upper=Nsite> site[Nobs];
  vector[Nobs] N1; // add other covariates here N2, N3, N4, etc.
}

parameters {
  vector[Nsite] mu0_site;
  vector[Nsite] zeta0_site;

  real betamu1;
  real betazeta1;
  real<lower=-1, upper=1> xi;  // FIXED bounds

}

transformed parameters {
  vector[Nobs] mu;
  vector[Nobs] sigma;


  mu = mu0_site + betamu1 * N1; //add betamu2 * N2, betamu3 * N3..... etc. as needed
  sigma = exp(zeta0_site+betazeta1 * N1);//add betazeta2 * N2, betazeta3 * N3..... etc. as needed for covariates
}

model {
// These priors were specified in stan. Stan tend to require a bit of guidance compared to BUGS since it uses HMC instead of MCMC. I used a non-informative prior for BUGS. If you're having difficulties, set a prior closer to the expected range with tighter bounds. 
  mu_global ~ normal(0, 10000);         
  zeta_global ~ normal(0, 10000);
  
  xi ~ uniform(-1,1);

  mu0_site ~ normal(0, 10000);           // FIXED: BYM2 needs unit-scale
  zeta0_site~ normal(0, 10000);         // FIXED: same for scale

  betamu1 ~ normal(0, 10000);
  betazeta1 ~ normal(0, 10000);

  target += gev_lpdf_vec(x1, mu, sigma, xi);
}

generated quantities {
  vector[Nobs] log_lik;
  for (i in 1:Nobs) {
    real z = (x1[i] - mu[i]) / sigma[i];
    if (fabs(xi) < 1e-15) {
      log_lik[i] = -z - exp(-z) - log(sigma[i]);
    } else {
      real arg = 1 + xi * z;
      log_lik[i] = (arg > 0) ?
        -(1 + 1 / xi) * log(arg) - pow(arg, -1 / xi) - log(sigma[i]) :
        negative_infinity();
    }
  }
}
