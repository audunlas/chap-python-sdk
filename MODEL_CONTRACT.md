# Model contract

This document outlines in normal language the contract/interface between chap and the models. The specifics of this contract is tested through the different test suites available in this repo.
The main job of the model in context is two steps. Training and predicting. Based on the training_data, the train functions should produce a trained model.
Based on the trained model, extended historical data, and forecasted future covariates, the model should provide probabilistic samples of future disease cases.

Through the model information, the model communicates to the chap what sort of data it expects. This is for now done through the two attributes: 'required_covariates' and 'allow_free_additional_continuous_covariates'
 - The required_covariates defines which covariates/columns should always be present. This is typically population and other specific covartiates the model needs.
 - The allow_free_additional_continuous_covariates flag tells chap whether the user can specify other covariates. The list of added covariates will be listed in additional_continous_covariates in the info part of the config object described below.

## User options
The model specifies what kind of configuration options are available through a ModelConfig class, which is a pydantic model. The actual values of these configuration options are sent to the model as the model_config part of the config object described below.

## The information object
Along with the data, chap also sends information needed by the model to train a model and predict. This information comes in two parts. The info object which includes information common to all chap models. For now this is

- prediction_length
- additional_continuous_covariates
- future_covariate_origin

The prediction_length informs the model at train time what prediction horizon it will be expected to predict for
The additional_continuous lists all additional covariates that a user has specified. These will also be coloumns in the dataframe.

## Train+Predict
The main flow of a model is to first train a model based on the provided training data and the configuration. The train function should return a pickle-serializable object representing the trained model.
The prediction function is slightly more complicated. In addition to the trained model and the configuration, the prediction function recieves two datasets:

### Historic Data
The historic data, are observed data points that can extend further into the future than the training_data. This should be used to refit the model, and update any lagged features or states.

### Future Data
This data contains forecasted data for covariates into the prediction period. These values should be used with care, as the quality of the forecasts can not be guaranted. By default, chap sends predicted values here based on location and seasonality only.
The future data contains all time points that the model is expected to predict for. It is usually simpler to use the time index of this dataset to find prediction periods, rather than extrapolation weeks and months into the future based on the historic data and prediction_length

### Predicted Data
The model is expected to return sampled predictions for all locations and all time periods present in future_data. The predictions should not contain NaN and should be nonnegative.
The format of the samples should be a dataframe with column names ['sample_0', 'sample_1', ...]
