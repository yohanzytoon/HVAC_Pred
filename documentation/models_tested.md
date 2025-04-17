# Models tested

Starting with experimental approach, trying every models and explainning why the model is or isnt a good fit for our problem

We will use standard measures :
 - MSE ( mean square error )
    $$  \frac{1}{n} \sum(y_i - ŷ_i)^2 $$
 - RMSE ( root mean square error )
    $$ \sqrt{MSE} $$
 - $R^2$ ( coeficient of determination )
    $$  1 - (\frac{\sum(y_i - ŷ_i)^2}{\sum(y_i - ȳ)^2})$$
 - CV RMSE ( cross validation RMSE )

For convenience purposes, we will use model.train as reference using different available models to forge our intuition on what models to use.

After finding a model deemed appropriate, we will replicate it.

### Linear regression
Obviously a bad choice as it doesn't take into account the temporality factor of out problem. 
Linear regression treats each data point independently, which ignores the sequential nature of time series data. The HVAC consumption patterns are highly dependent on previous consumption values and temporal patterns (daily cycles, weekly patterns).

MSE: 1.0443
RMSE: 1.0219
R²: 0.7198
RMSE CV: 1.3160 ± 0.3062

![linear regression prediction vs real values](images/linear%20reg%20pred%20vs%20real%20values.PNG)
![linear regression residues vs predictions](images/linear%20reg%20residues%20vs%20pred.PNG)


### Ridge

MSE: 1.0443
RMSE: 1.0219
R²: 0.7198
RMSE CV: 1.3160 ± 0.3062

### Lasso 

MSE: 1.0978
RMSE: 1.0478
R²: 0.7054
RMSE CV: 1.3793 ± 0.3934

### Random forests

MSE: 0.0654
RMSE: 0.2558
R²: 0.9824
RMSE CV: 0.3734 ± 0.2181

### Support vector regression

MSE: 1.4196
RMSE: 1.1915
R²: 0.6191
RMSE CV: 1.5004 ± 0.8107

### Gradient Boosting

MSE: 0.1322
RMSE: 0.3636
R²: 0.9645
RMSE CV: 0.3763 ± 0.1903

## More specific models

We will now use some more specific models used for temporal predictions

### ARIMA ( AutoRegressive Integrated Moving Average )

