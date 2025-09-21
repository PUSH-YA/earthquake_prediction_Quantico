

It is an interpolation technique that can be used with unknown means and works for minimising error variance while providing confidence intervals which can again be used as parameters

It does not scale well for large large datasets and uses sparse solvers
- requires detrending or localised variograms.


The actual works on using an Autoregressive model (Markove model with a bit more lag) and then minising the variance. 

-----

Main idea:
- We can use latitude, longitude and time as 2 inputs to predict magnitude at specific points
- I then tried to use time as a vector and then see how that would work 

I tried working with sample data and seeing how the model would work:
- it ended up singular matrix each time which is because we have a lot of the same coordinates in it like if we see the image below:

<img src = "../../prelim data analysis/basic.png">

So using a coarser grid and changing the data worked better which works better for regions as opposed to exact coordinates

<img src = "./kriging output.png">

Notes:
- Small Datasets: Kriging requires ~15+ points for reliable results. With only 5 points, consider simpler methods (e.g., IDW, nearest neighbor).
- Parameter Tuning: Adjust sill, nugget, and range by trial and error. Use ok.variogram_model_parameters to see fitted values if possible.
- Uncertainty: Results are illustrative only due to limited data.

Practicality
- training time: 3 min
- prediction time: virtually instantaneous

TODO:
- make it work with time data
- try Inverse Distance Weighting model 