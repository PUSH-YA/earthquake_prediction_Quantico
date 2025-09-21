## Notes






## Update

- I had some questions regarding the model you chose 
- I have not tried the data on turkey one yet  but I was thinking we can decide which direction we want to head into

## Catchup

Questions:
- How did you ensure your model made sense
- Is it using interarrival data in days
- I am a bit suspicious of the state transition probabilities not being there

## Incorporating into the Risk model 

Whatever model we can use the time series of the state space changes as the input for the risk modelling ones

The one Torren posted seems so, we can use that and
1. Model the state transitions as the time series and then use it for the risk modelling
2. We can also feed in the specific properties of the state space models such as the magnitudes for the risk modelling
  - we can choose either to give the states times series or specific property time series and see if yields any statistically significant risk modelling 

```python
# Convert state scores to DataFrame
state_df = DataFrame({
    "date": returns.index,
    "risk_regime": state_scores
})

# Merge with existing style factors
style_scores = style_scores.join(state_df, on="date")

# Update style_df to include the HMM risk regime
style_df = ddf.select(["date", "symbol", "val_score", "mom_score", "sze_score", "risk_regime"])

# Estimate factor returns with HMM state as a new factor
fac_df, eps_df = estimate_factor_returns(
    returns_df, 
    mkt_cap_df, 
    sector_df, 
    style_df,  # Now includes risk_regime
    winsor_factor=0.1, 
    residualize_styles=False
)

```






