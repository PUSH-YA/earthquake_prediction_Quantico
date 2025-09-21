# Proposed ideas:

### Based on previous knowledge:

1. Linear/ Spline / Cubic interpolation 
2. Polynomial regression
3. Kernel methods can be used for smoothing the data 
4. Time series analysis for cyclical data
	1. where we try to predict seasonal variation and trend in the data but that gives you parameters instead of a cts average
	
#### Issues with smoothing:
- it narrows down the key innovation points, example from hackathon: guassian smoothing prevented us from noticing spikes
- there is Bilateral filter that preserves the "edges" and "corners" of the time series but it is computationally expensive and I don't think it works better than a simple $MA$ process 

#### Issues with splines:
- they are really good for smoothness and continuity but have even more of an issue for loosing information from a time series data. 


## Possible solutions:

### Radial basis function interpolation:
- it works for irregularly spaced data and works well for high dimensional and scattered data 
- The interpolant takes the form of the radial basis function 


$$ 
\begin{bmatrix}
\varphi(||x_1 - X_1||) & \varphi(||x_2 - X_1||) & \cdots & \varphi(||x_n - X_1||) \\
\varphi(||x_1 - X_2||) & \varphi(||x_2 - X_2||) & \cdots & \varphi(||x_n - X_2||) \\ 
\vdots & & \ddots \\
\varphi(||x_1 - X_2||) & \varphi(||x_2 - X_2||) & \cdots & \varphi(||x_n - X_2||) \\
\end{bmatrix}
$$

where $\varphi$ require strictly positive definite functions that require a tuning shape parameter such as:

$$\varphi(x) = e^{-(\varepsilon r)^2}$$

#### Issues: careful with shape parameter

We will have to generalise a shape parameter that works for different trends (*link* in the image).

<a href="http://www.jessebett.com/Radial-Basis-Function-USRA/">
<img src="./images/rb_interpol.png" width="300"/>
</a>

<!-- ![](./images/rb_interpol.png) -->


We will need to experiment with the data to see how well something like this would work

### Kriging (Gaussian process interpolation):
- provides uncertainty estimates
- Minimises error variance for spatial data 
- Can incorproate trends and correlations in the data
- was also referenced in an earthquake prediction models study i read

I am assuming if we are looking at features in the time domain, there will be a lot more irregularity in comparison to a sampling model
- Kriging will in general not be more effective than simpler methods of interpolation if there is little spatial autocorrelation among the sampled data points [link](https://www.publichealth.columbia.edu/research/population-health-methods/kriging-interpolation)

So, the general function is:

$$
\hat{\mathcal{Z}} = \sum_{i=1}^{N}\lambda_i \mathcal{Z}(s_i)
$$

It relies on **variogram :=**(sometimes called a “semivariogram”) is a visual depiction of the covariance exhibited between each pair of points in the sampled data. 

The $\gamma(x)$, the $acf$ is plottd agaisnt the "lag" while the model variogram is the distributional model that best fits the data.
- [link](https://www.publichealth.columbia.edu/research/population-health-methods/kriging-interpolation)

We then use different semivariogram models to fit them (*link* in the image):
- Spherical
- Exponential 
- Gaussian
- Linear


<a href="http://www.jessebett.com/Radial-Basis-Function-USRA/">
<img src="./images/kriging_maths.png"/>
</a>

<!-- ![](./images/kriging_maths.png) -->


The selected model influences the prediction of the unknown values, particularly when the shape of the curve near the origin differs significantly. The steeper the curve near the origin, the more influence the closest neighbors will have on the prediction.
![](./images/kriging_interpol.png)


#### Issues with kriging:
- it seems to work with stationary data well but not so much for non-stationary data 
	- if we need to apply to Earthquake data, we will need to make it stationary which can be done by differencing it but this will make it more difficult for future models to work on it
- The data could be sparse and we would need something called **adaptive Kriging** to work well with it. 
- There is also some issues with boundary data





### Seismic trace interpolation (CAE)

Useful for irregularly sampled data using convolutional autoencoder: [https://library.seg.org/doi/10.1190/geo2018-0699.1](https://library.seg.org/doi/10.1190/geo2018-0699.1)

the main benefit:
- The irregularly sampled data are taken as corrupted data. 
- By using a training data set including pairs of the corrupted and complete data, $CAE$ can automatically learn to extract features from the corrupted data and reconstruct the complete data from the extracted features.

I cannot seem to access it (even with UBC's open athens) and I could not find many other papers that talked about it in my preliminary research

I also have limited knowledge of AutoEncoders, especially when combined with convolutional layers so will need to research more for it. 