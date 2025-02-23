# SysID library

The `SysID` library provides a data-driven approach to identify models of a given system, from observations made on this system. 

Currently, the `SysID` library supports the identification of:
- Polynomial models as identified through step-wise regression
- (Dense) Feed-forward neural network models of arbitrary depth and number of neurons

The identified models are also accompanied by prediction intervals, which provide insight into the reliability (i.e. confidence) of the associated model predictions. 

## Dependencies
The dependencies for the base SysID class and modules located under `Utility` are required whereas the dependencies of the identification techniques are only necessary if that technique is used. 

### SysID & Utility (i.e. required) dependencies
- numpy (1.20.1)
- pandas (1.2.4)
- scipy (1.4.1)

### (Additional) dependencies for polynomial models (stepwise regression)
- matplotlib (3.3.4) *Note: if plotting is not necessary, models can still be identified without this package, but make sure to comment it out in stepwise_regression.py*

### (Additional) dependences for Feed-forward neural network models
- tensorflow (2.3.0)
- matplotlib (3.3.4) *Note: if plotting is not necessary, models can still be identified without this package, but make sure to comment it out in stepwise_regression.py*



# Usage
For ease of use and readability, the `SysID` class works with `pandas DataFrames` to identify models. This is because the columns of the DataFrame can be named, typically in accordance with the variables used as predictors for a given target. For example, the polynomial technique uses these column names to find the relevant columns in the DataFrame to retrieve and construct the regressors. With appropriate column names, the resultant polynomial model is also more readable as the chosen predictors, and interactions therein, are immediately visible. 

It is therefore important to also be consistent with the naming schemes of the column variables and model inputs. For example, when identifying a neural network model, it is necessary to specify an `input vector` which corresponds to the column names of the DataFrame, describing which of these should be used as predictors for model identification. 

In general, models are identified in the following procedure:
1. Initialize model through `SysID.Model(Technique)` specifying the desired technique (see **Documentation of SysID methods** below)
2. Construct model and prepare for training through the `SysID.Model.compile()` method (see **Documentation of SysID methods** below)
3. Train model on input DataFrame through the `SysID.Model.train` method (see **Documentation of SysID methods** below)
4. Make predictions using the `SysID.Model.predict()` method. This outputs both a tuple (pred, predVar) of the prediction itself (pred) and the associated reliability as a variance (predVar).

At any point after initialization, the a summary of the model may be printing to the command line interface through the `SysID.Model.summary()` method. 


# Documentation of SysID methods

## Initialization
Models are initialized by specifying the `Technique` to be used for identification. In essence, the SysID module decorates and standardizes the technique-specific identification functions for ease of use. 
```python
MyModel = SysID.Model(Technique)
```
Inputs:
- `Technique` - string - indicates the technique to be used for identification. Options are "stepwise_regression" (Polynomial model) or "ann" (feed-forward neural network). If the technique is unknown, then the initialization attempt will raise a `ValueError`. 

Outputs:
- A `SysID.Model` object

`SysID.Model` attributes:
Upon initialization, the `SysID.Model` will have various attributes:
- SysID.Model.technique - string - the chosen identification technique
- SysID.Model.ModelState - string - the current state (['Initialized', 'Compiled', or 'Trained']) of the model 
- SysID.Model.ModelStateHistory - list of strings - logs the changes in model state
- SysID.Model.UtilityFuncs - object - method to provide access to technique-specific functions (see associated files and readmes in *Techniques* subfolder) that are otherwise unused by the SysID.Model class. 
- SysID.Model.CompiledModel - object - The compiled model, is `None` until model compilation is complete. 
- SysID.Model.TrainedModel - object - The trained model, is `None` until after the model is trained. 
- SysID.Model.x_train - array-like - The input data used for training. Is `None` until after training. 
- SysID.Model.y_train - array-like - The target data used for training. Is `None` until after training.

Example:
```python
>>> MyPolyModel = SysID.Model('stepwise_regression')
>>> MyPolyModel.technique
'stepwise_regression'
>>> MyPolyModel.ModelState
'Initialized'
>>> MyPolyModel.CompiledModel
None
>>> MyPolyModel.ModelStateHistory
['Initialized']
```


## Model compilation
Before training can occur, models must first be constructed using the `compile()` method. The exact input parameters for model compilation are technique specific (for more details, see the associated README files in the *Techniques* subfolder). This method modifies the `Model` object directly, and does not return anything. 
```python
Model.compile(*args, **kwargs)
```
Inputs:
- *are technique specific, see corresponding documentation*

Outputs:
- `None`


## Model training
Once a model has been compiled, it may be trained on the data through the `train()` method. This method modifies the `Model` object directly, so nothing is returned. As with the `compile()` method, there additional technique-specific (keyword) arguments which can be passed onto the train function to configure the training phase. 
```python
Model.train(x, y, *args, **kwargs)
```
Inputs:
- `x` - pandas DataFrame - The input data (i.e. states) for the model. 
- `y` - pandas DataFrame - The target data which the model will attempt to fit, using the input data, `x`

Outputs: 
- `None`


## Making predictions
A trained model may be used to make predictions using the `predict()` method. Additional technique specific keyword arguments may be passed as necessary. 
```python
pred, predVar = Model.predict(x, **kwargs)
```
Inputs:
- `x` - pandas DataFrame - Data for which predictions should be made

Outputs:
- `pred` - numpy array - Model prediction
- `predVar` - numpy array - Variance in model predictions (used to construct prediction intervals)


## Evaluating the models 
The quality of an identified model may be checked through the `evaluate()` method. The exact input parameters for model evaluation are technique specific (for more details, see the associated README files in the *Techniques* subfolder). 
```python
evaluation = Model.evaluate(*args, **kwargs)
```
Inputs:
- *are technique specific, see corresponding documentation*

Outputs:
- `evaluation` - dictionary - of evaluation results



## Mitigating over-fitting
To mitigate the over-fitting of the training data, the `reduceModel()` method may be used. 
```python
Model = Model.reduceModel(x = None, y = None, inplace = False, **kwargs)
```
Inputs:
- `x` - pandas DataFrame - input data used to evaluate model for over-fitting (e.g. cross-validation). If `x = None` then the training data will be used (Not recommended). 
- `y` - pandas DataFrame - target data used to evaluate model for over-fitting. If `y = None` then the training data will be used (Not recommended). 
- `inplace` - boolean - describes if a copy of the model should be returned, or if modifications should be made inplace. Default is False, so a copy of the model will be returned. 

Outputs:
- reduced model



## Overview of model
Users can obtain an overview of the model through the `summary()` method. 
```python
Model.summary()
```
Inputs:
- None

Outputs:
- Prints summary of the model in CLI



## Saving and loading models
Models can be saved through the `save` method and loaded into other scripts using the `load` method. *Note that the `save` method does not save information regarding the training data and targets (i.e. `Model.x_train` and `Model.y_train` are lost upon saving.)*

Saving models
```python
Model.save(path)
```
Inputs:
- `path` - string - directory in which to save the model

Outputs:
- None

Loading models
```python
import SysID
Model = SysID.load(path)
```
Inputs:
- `path` - string - directory in which the model is saved

Outputs:
- A `SysID.Model` object


## Copying models
Deep copies of (trained) SysID models can be made through the `copy()` method.
```python
copiedModel = Model.copy()
```
Inputs:
- None

Outputs:
- A copy of the (trained) `SysID.Model` object