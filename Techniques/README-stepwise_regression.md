# Stepwise regression module

Script with necessary functions to build, train and make predictions with a stepwise regression model
Backend for the SysID class 'Stepwise_Regression' option. 
  
Stepwise regression based on:
  [klein2006] V. Klein and E.A. Morelli. Aircraft System Identification: Theory and Practice. AIAA
  education series. American Institute of Aeronautics and Astronautics, 2006. ISBN:
  9781563478321.

Written by Jasper van Beers
Created: 12-01-2021
Last edit: 19-04-2021


# Dependencies
Distributed libraries:
- numpy (1.20.1)
- matplotlib (3.3.4) 
- Python (3.8.8)

Local libaries:
- .Utility.parser


# Documentation

## "Public" functions

### compile(data, polys, fixedParams, includeBias = True)
Function to build candidate model regressors (both candidate and fixed) from input data

Inputs:
- `data` - Pandas DataFrame containing inputs for the model where columns denote the input variables and rows the number of samples
- `polys` - List of the various polynomials from which the candidate regressors are based. See example below.
    - Syntax: List of dictionaries describing the candidate polynomials. Each dictionary in the list corresponds to polynomials composed of the same variables and degree. Dictionaries must have the following fields (i.e. keys)
        - 'vars' = variables, as list, which consitute the polynomial. These must correspond to the column names in input `data`. e.g. `vars:['x1', 'x2', 'x5']`
        - 'degree' = Integer dictating degree of the polynomial
        - 'sets' = List of additional constants/variables which the polynomial is multiplied with. 
            - For just the polynomial itself, `'sets':[1]` the resultant polynomial is multiplied by 1. 
            - For more complex variable relations, for example $y_{1}\cdot P(x_{1},...,x_{n})$, then `sets = ['y1']` or `sets = [1, 'y1']` if both the base and y1 polynomials are required. 
            - It is also possible to specify combinations of variables (e.g. 'x1 + x2'), so long as the components are columns in `data`
            - Other combinations, such as `sets = ['y1*y2^(y3 + y1)']` are also possible, given that the components are all valid columns in `data`
        - Example of a possible `polys`: 
            - `P = [{'vars':['x1', 'x2'], 'degree':2, sets:[1, 'x3']}]`
            - This is equivalent to $P = (1 + x_{3})*(x_{1}^{2} + x_{1} + x_{1}x_{2} + x_{2} + x_{2}^{2})$. **Note** The bias vector is not included in these candidate sets as it is included by default (see `includeBias`). 
- `fixedParams` - List of strings indicating the fixed regressors. See example below
    - Syntax: list of the fixed regressors (i.e. variables always in polynomial model structure), as strings. These should correspond to columns of data.
        - Examples:
            - If `x1` is a fixed regressor then `fixedParams = ['x1']`
            - Multiple variables may be inputted: `fixedParams = ['x1', 'x2', '-1*x3']`
            - Combinations of variables are also valid inputs: `x1*x2 + x3/x4` can be encoded as `fixedParams = ['x1*x2 + x3/x4']` or `fixedParams = ['x1*x2', 'x3/x4']`
            - If there are no fixed regressors then `fixedParams = []` 
- `includeBias` - Boolean indicating if a bias vector should be included. Default is True. 

Outputs:
- `CompiledModel` - Dictionary containing the necessary polynomial regressors for the model. 


### train(CompiledModel, TrainingInputs, TrainingTargets, stop_criteria = 'PSE', Fin = 4, Fout = 4, k_lim = 10, force_k_lim = False)
Function to identify a polynomial model using stepwise regression 

Inputs:
- `CompiledModel` - Dictionary containing the necessary polynomial regressors for the model. Equivalent to the output of function `compile`. ***Note** that this is automatically specified when using the `SysID.Model.train()` method*
- `TrainingInputs` - Unused but remains for compatibility with `SysID.Model.train` method. The training inputs are already mapped to regressors in `compile`.
- `TrainingTargets` - Pandas DataFrame of the target data used for model identification. Shape is [N x 1] where N = the number of observations (samples). 
- `stop_criteria` - Dictates what stopping criteria to use. Default is 'PSE' (Predict Square Error)
    - Options are: 
        - `'PSE'` - Predict square error
        - `'R2'` - Coefficient of determination
        - `'F0'` - F-test
        - `None` - stops after iteration k_lim unless natural termination condition is met (i.e. removed regressor = just added regressor)
- `Fin` - F-test bound for adding a regressor to model structure, default `Fin = 4` (See Klein 2006)
- `Fout` - F-test bound for removing a regressor from model structure, default `Fout = 4` (See Klein 2006)
- `k_lim` - Upper bound for the number of iterations if stop_criteria is not given (i.e. is `None`), default `k_lim = 10`. 
- `force_k_lim` - Boolean which enforces a strict upper bound of `k_lim`, regardless of the chosen stopping condition. Default is `False`.

Outputs:
- `TrainedModel` - Dictionary containing the identified model and some additional information in the `SysID.Model` standard format.
    - Syntax:
        - `TrainedModel['Model']` - Dictionary of model elements
            - `TrainedModel['Model']['A']` - Final regressor matrix
            - `TrainedModel['Model']['Parameters']` - Ordered list of polynomial coefficients, including bias coefficient if model is compiled with bias vector
            - `TrainedModel['Model']['Regressors']` - Ordered list of selected polynomial regressors (in-line with coefficients above), including bias term if model is compiled with bias vector
            - `TrainedModel['Model']['Has Bias']` - Boolean indicating if model has a bias vector
            - `TrainedModel['Model']['_sigma2']` - Variance of modelling errors. Used to estimate prediction intervals. 
            - `TrainedModel['Model']['_inv(XtX)']` - Matrix of observed inputs. Used to estimate prediction intervals. 
        - `TrainedModel['Additional]` - Dictionary of any additional information
            - `TrainedModel['Additional']['Log']` - Log detailing stepwise procedure 


### predict(Model, x)
Function to make predictions on input data using the model identified through stepwise regression

Inputs:
- `Model` - Dictionary containing model values. Equivalent to the output of function `train`. ***Note** that this is automatically specified when using the `SysID.Model.predict()` method*
- `x` - Pandas DataFrame containing the input data upon which predictions should be made. Shape [N x M] with N = number of observations and M = number of independent variables.

Outputs:
- `pred` - numpy.matrix of model predictions based on x, shape [N x 1].
- `var` - numpy.matrix variance of errors associated with the predictions, based on the training data 


### evaluate(Model, inputs, target, showPlots = True)
Function to validate the chosen model through an analysis of the model residuals and parameter covariances

Inputs:
- `Model` - Dictionary containing model values, in `SysID.Model` format. Equivalent to output of function `train`. ***Note** that this is automatically specified when using the `SysID.Model.evaluate()` method*
- `inputs` - Pandas DataFrame containing the input data for evaluation. Shape [N x M] with N = number of observations and M = number of independent variables.
- `target` - Corresponding targets to evaluate against. Shape [N x 1], N = number of observations
- `showPlots` - Dictates if plots should be shown or not. Possible values are: `True`, `False`, and `'plot'`.Default is `True`, plots will be drawn and shown. `False` means plots will not be drawn or shown. `'plot'` means plots will be drawn but not shown. Users can call `matplotlib.pyplot.show()` when ready to show drawn plots. 

Outputs:
- `evaluation` - Dictionary of evaluation containing fields:
    - `evaluation['Residual Error']` - numpy array of model residual errors to target data
    - `evaluation['Autocorrelation']` - numpy array of model resiudal esidual error autocorrelation
    - `evaluation['Index']` - index corresponding to autocorrelation above
    - `evaluation['COV']` - numpy matrix of covariances of identified polynomial parameter 
    - `evaluation['Coefficient Variance']` - numpy array of variance of a parameter (i.e. diagonal of covariance matrix above)
    

### summary(Status, Model)
Function to print a summary of the model to the terminal

Inputs:
- `Status` - string describing the status of the model. Options are `compiled` or `trained`. ***Note** that this is automatically specified when using the `SysID.Model.summary()` method*
- `Model` - Model corresponding to the state (i.e. output of `compile` and/or `train` method). ***Note** that this is automatically specified when using the `SysID.Model.summary()` method*

Outputs:
- None - prints model summary to terminal


### save(path, model)
Function to save (trained) stepwise regression model 

Inputs:
- `path` - string of the destination directory
- `model` - Dictionary of trained model to save (i.e. output of `train`). ***Note** that this is automatically specified when using the `SysID.Model.save()` method*

Outputs:
- None


### load(path)
Function to load trained stepwise regression model

Inputs:
- `path` - string of the source directory

Outputs:
- `TrainedModel` - dictionary of the TrainedModel, in `SysID.Model` format. Equivalent to the output of `train`. 


### reduceModel(polyModel, inputs, targets, covarianceThreshold = 0.1)
Function to reduce model by removing high covariance regressor terms, if any

Inputs:
- `polyModel` - `SysID.Model` object of the trained polynomial model. ***Note** that this is automatically specified when using the `SysID.Model.reduceModel()` method*
- `inputs` - Pandas DataFrame (Shape NxM where N = number observations, M = number input variables) of input data used to obtain covariances. Ideally, this dataset is different that that used for training. 
- `targets` - Pandas DataFrame of associated targets for `inputs`
- `covarianceThreshold` - float (<1) which denotes the maximum allowable covarainace, when expressed as a ratio of the associated coefficient's magnitude. Default `covarianceThreshold = 0.1` (i.e. 10%).

Outputs:
- `reducedModel` - Returns a copy of the `SysID.Model` object where the `SysID.Model.TrainedModel` attribute is modified to the reduced model


### trimModel(polyModel, cutoff)
Function to remove all regressors added after a specified index

Inputs:
- `polyModel` - `SysID.Model` object of the trained polynomial model.
- `cutoff` - Integer index for and after which regressors should be removed. Thus, the cutoff index will also be dropped. 

Outputs:
- `reducedModel` - Returns a copy of the `SysID.Model` object where the `SysID.Model.TrainedModel` attribute is modified to the reduced model


### dropRegressor(polyModel, index, retrain = True)
Function to drop a regressor, at index i, from a polynomial model

Inputs:
- `polyModel` - `SysID.Model` object of the trained polynomial model. 
- `index` - Integer index corresponding to the regressor to be removed
- `retrain` - Boolean to indicate if model should be re-fit to the training data using OLS. Default is `True`. 

Outputs:
- `reducedModel` - Returns a copy of the `SysID.Model` object where the `SysID.Model.TrainedModel` attribute is modified to the reduced model


### copy(model)
Function to make a copy of a polynomial model

Inputs:
- `model` - `SysID.Model.TrainedModel` object of the trained polynomial model. Synonymous to the output of `train` method. ***Note** that this is automatically specified when using the `SysID.Model.copy()` method*

Outputs:
- 'modelCopy' - Copy of the `SysID.Model.TrainedModel` object


### plotRegressorContributions(polyModel, inputs = None, x = None, returnFig = False, normalizer = None, colors = None, legendLoc = 'best')
Function to plot the individual regressor contributions in the context of the final polynomial model structure and parameter values. Also returns the regressor contributions in terms of RMSE and R2.

Inputs:
- `polyModel` - `SysID.Model` object of the trained polynomial model
- `inputs` - Pandas DataFrame (Shape NxM where N = number observations, M = number input variables) of input data for which these regressor contributions should be plotted. Default = `None` (Will use training data from `polyModel.x_train`, if available)
- `x` - Array-like to use as x-axis in plot. Default = `None` (behavior is that x-axis will represent the index)
- `returnFig` - Boolean to indicate if figure object should be returned or not. Default = `False`
- `normalizer` - Array-like of normalizing factor for data, if used. Default = `None` (i.e. unused and/or no normalization)
- `colors` - List-like of colors to be used for plotting. Colors will be cycled through so repetitions will occur if `len(colors)` < number of regressors. Default = `None` (colors will be inferred from a `cmap`)
- `legendLoc` - String indicating location of legend. Locations have to be compatible with the equivalent parameter in the `matplotlib` library.

Outputs:
- `outputs` - Figure object (if returnFig = True) in a dictionary accessible with key 'fig'.



## "Private" functions
These are functions which are used by other functions in `stepwise_regression.py` but may nonetheless be accessed through the `SysID.Model.UtilityFuncs` attribute. 

### _OLS(A, z, hasBias = True)
Function to determine ordinary least squares regression

Inputs:
- `A` - Regressor Matrix, as numpy.matrix with shape [N x M] where N = number observations, M = number regressors
- `z` - Measurements of target, array with shape [N x 1]

Outputs:
- List of `[params, pred]`. params denotes the model parameters, array with shape [M x 1]. pred gives model predictions, array with shape [N x 1]


### _PSE(predictions, targets, p)
Function to determine the predict square error, PSE

Inputs:
- `predictions` - Model predictions, array-like with shape [N x 1] where N = number observations
- `targets` - Targets (Measurements), array-like with shape [N x 1]
- `p` - Integer of the number of regressor terms (including bias term, if model has bias)

Outputs:
- `PSE` - Float of predict square error


### _CoeffOfDetermination_R2(predictions, targets)
Function to calculate the coefficient of determination (R squared)

Inputs:
- `predictions` - Model predictions, array-like with shape [N x 1] where N = number observations
- `targets` - Targets (Measurements), array-like with shape [N x 1]

Outputs:
- `R2` - Float of the coefficient of determination


### _GenPowers(d, numX)
Function to determine all possible terms in a polynomial of degree, d, excluding the bias term
Example:
- If `P(x1, x2)` is a polynomial of degree `2` then the possible terms are
    - `x1`, `x2`, `x1^2`, `x2^2`, and `x1x2`
    - The power lists are correspondingly
    - `[1, 0]`, `[0, 1]`, `[2, 0]`, `[0, 2]` and `[1, 1]` (Here, the integer at index i denotes the power of variable x_i)
    - `_GenPowers` produces such a list of powers for a given polynomial degree and number of independent variables
    
Inputs:
- `d` - Polynomial degree, as integer
- `numX` - Number of independent variables, as integer
    - e.g. In the example above, `x1` and `x2` are the independent variables, so `numX = 2`
    
Outputs:
- `validPerms` - a list of valid power lists


### _BuildCandidateRegressors(polys, data)
Function to build the candidate polynomial regressors (excluding bias vector)

Inputs:
- `polys` - List of dictionaries describing the candidate polynomials. *For Syntax, see `compile` method above*
- `data` - pandas DataFrame containing the relevant data for building the polynomials with shape [N x R] where N = number observations and R = number columns

Outputs:
- `Regressors` - Dictionary mapping of regressors to their values derived from `data`. Each entry in Regressors corresponds to a single regressor.
    - e.g. Regressors = {'x1^2':[n1, n2, ..., nN], 'x1*x2':[h1, h2, ..., hN], ...}
    

### _BuildFixedRegressors(variables, data, addBiasVector = True)
Function to build the fixed regressors 

Inputs:
- `variables` - list of the fixed regressors, as strings. These should correspond to columns of data. *For Syntax and possibilities, see `compile` method above*
- `data` - Pandas DataFrame containing the data necessary for building the fixed regressors where columns correspond to input variables and rows samples 
- `addBiasVector` - Boolean indicating if a bias vector (i.e. [1,...,1]) should be added. By default addBiasVector is True.

Outputs:
- `Regressors` -  Dictionary mapping of regressors to their values derived from `data`. Each entry in Regressors corresponds to a single regressor. 


### _BuildRegressorMatrix(regressors, data, hasBias = True)
Function to build a regressor matrix from a set of (currently selected) regressors

Inputs:
- `regressors` - List of strings corresponding to the regressors (i.e. algebraic equations of base variables in columns of `data`)
- `data` - Pandas DataFrame of the identification data necessary to build the regressor matrix. Column names should correspond to base variables used to construct the regressors. 
    - e.g. for `regressors = ['x1 + x2', '5*x3']`, terms `'x1'`, `'x2'`, and `'x3'` should be columns in data. 
- `hasBias` - Boolean that indicates if the regressor list has a bias a bias vector. Default is True. 

Outputs:
- `regMat` - numpy matrix containing the regressors (i.e. A in z = A*parameters) of shape [N x M] where N = number of observations and M = number regressors
