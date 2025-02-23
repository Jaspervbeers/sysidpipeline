'''
Script with necessary functions to build, train and make predictions with a stepwise regression-ANN 
hybrid model. 
Backend for the SysID class 'Stepwise_ANN_Hybrid' option. 

Written by Jasper van Beers
Created: 16-02-2021
Last edit: 05-04-2021
'''
# ================================================================================================================================ #
# Global Imports
# ================================================================================================================================ #
import numpy as np

# ================================================================================================================================ #
# Local Imports
# ================================================================================================================================ #
try:
    from .Techniques import stepwise_regression, ANN
except ImportError:
    from Techniques import stepwise_regression, ANN
try:
    from .Utility import Utility
except ImportError:
    from Utility import Utility



# ================================================================================================================================ #
# Functions
# ================================================================================================================================ #

def parse_kwargs(func, kwargs):
    '''Utility function to parse keyword arguments passed onto the stepwise_regression and/or ANN functions
    
    :param func: Function to which the keyword arguments should be passed 
    :param kwargs: Dictionary of keyword arguments. The keys should correspond to relevant keywords in func
    :return: Dictionary of keywork arguments to pass on to func
    ''' 
    # If keyword arguments are passed
    if kwargs is not None:
        # Probe function for keyword arguments
        _, all_func_kwargs = Utility.getArgs(func)
        # Create dictionary of keyword arguments, if they exist in func. 
        input_kwargs = {k:kwargs[k] for k in kwargs if k in all_func_kwargs}
    # If no keyword arguments, return empty dictionary
    else:
        input_kwargs = {}
    return input_kwargs



def compile(X, Y, candidate_polynomials, fixed_regressors, compensator_x_vec = None, stepwise_kwargs = None, compensator_kwargs = None):
    '''Function to compile (build) hybrid model. Relies on stepwise_regression.py and ANN.py

    :param X: Training input data, as a Pandas DataFrame/Series. Shape of NxM where N = Number of data points and M = number of inputs
    :param Y: Training target data, as a Pandas DataFrame/Series. Shape of NxP where N = Number of data points and P = number of model outputs
    :param candidate_polynomials: List of dictionaries describing the candidate polynomials where each item in the list corresponds to a polynomial. See note **
    :param fixed_regressors: List of fixed regressors as strings. The bias term is included by default (i.e. with fixed_regressors = []), unless 'includeBias=False' in <stepwise_kwargs> is specified. 
    :param compensator_x_vec: List of variables (i.e. columns) in <X> to be used as ANN inputs. Default = None. Thus, all columns of X will be used. 
    :param stepwise_kwargs: A dictionary of keyword arguments to be passed onto the stepwise.compile function. See stepwise_regression.py for further documentation. 
    :param compensator_kwargs: A dictionary of keyword arguments to be passed onto the ANN.compile function. See ANN.py for further documentation. 
    :return: Dictionary containing the hybrid model. See documentation for the output of stepwise_regression.compile() for dictionary entries. This dictionary is augmented with an additional entry 'Compensator' which hosts the ANN model(s).

    :Notes:
    ** Dictionaries must have the following fields (i.e. keys)
        - 'vars' = variables, as list, which consitute the polynomial. These must correspond to the column names in input <X>. 
        - 'degree' = Integer dictating degree of the polynomial
        - 'sets' = Additional constants/variables. It is also possible to specify combinations of variables (e.g. 'x1 + x2'), so long as the components are columns in <X>
        
        Example of a dictionary: P = {'vars':['x1', 'x2'], 'degree':2, sets:[1, 'x3']} = (1 + x3)*(x1^2 + x1 + x1x2 + x2 + x2^2). 
        Note that the bias term is not included as this is part of the <fixed_regressors>
    '''
    # Extract keyword arguments for polynomial model and compensator (ANN), if any
    input_kwargs_stepwise = parse_kwargs(stepwise_regression.compile, stepwise_kwargs)
    # input_kwargs_ANN = parse_kwargs(ANN.compile, compensator_kwargs)
    input_kwargs_ANN = compensator_kwargs

    # Compile polynomial and compensator ANN and fuse the two
    if compensator_x_vec is None:
        compensator_x_vec = list(X.columns)
    CompiledModel_stepwise = stepwise_regression.compile(X, candidate_polynomials, fixed_regressors, **input_kwargs_stepwise)
    CompiledModel_ANN = ANN.compile(X[compensator_x_vec], Y, **input_kwargs_ANN)
    CompiledModel_stepwise.update({'Compensator':CompiledModel_ANN['Model'], 'Compensator Inputs':compensator_x_vec, 'Compensator modelOutput':CompiledModel_ANN['modelOutput']})

    return CompiledModel_stepwise



def train(CompiledModel, X, Y, stepwise_kwargs = None, compensator_kwargs = None):
    '''Function to train the hybrid model. Relies on stepwise_regression.py and ANN.py

    :param CompiledModel: Compiled hybrid model. Output of function <compile>.
    :param X: Training input data, as a Pandas DataFrame/Series. Shape of NxM where N = Number of data points and M = number of inputs
    :param Y: Training target data, as a Pandas DataFrame/Series. Shape of NxP where N = Number of data points and P = number of model outputs
    :param stepwise_kwargs: A dictionary of keyword arguments to be passed onto the stepwise.train function. See stepwise_regression.py for further documentation. 
    :param compensator_kwargs: A dictionary of keyword arguments to be passed onto the ANN.train function. See ANN.py for further documentation. 
    :return: Dictionary containing the trained hybrid model. See documentation for the output of stepwise_regression.compile() for dictionary entires. This dictionary is augmented with additional entires corresponding to the ANN compensator. These are:
        
        - 'Compensator' = List of the ANN model(s)
        - 'Compensator Norms' = Parameters used for normalization. Required to convert ANN outputs back into the correct magnitude. 
    '''
    # Extract keyword arguments for polynomial model and compensator (ANN), if any
    input_kwargs_stepwise = parse_kwargs(stepwise_regression.train, stepwise_kwargs)
    # input_kwargs_ANN = parse_kwargs(ANN.compile, compensator_kwargs)
    input_kwargs_ANN = compensator_kwargs

    # Reshape Y if necessary
    if len(Y.shape) <= 1:
        Y = np.matrix(Y).T

    # Train stepwise model
    print('[ INFO ] Training base polynomial model')
    Base_Model = stepwise_regression.train(CompiledModel, X, Y, **input_kwargs_stepwise)
    Base_pred = Base_Model['Model']['A']*Base_Model['Model']['Parameters']

    print('[ INFO ] Training ANN Compensator for base model prediction error')
    error_pred = Y.reshape(Base_pred.shape) - Base_pred
    # Normalize input 
    # (error_pred_Norm, error_mean, error_std) = ANN.Normalize(error_pred, returnStats=True)
    # (X_norm, X_mean, X_std) = ANN.Normalize(X[CompiledModel['Compensator Inputs']], returnStats=True)
    # Compensator = ANN.train({'Model':CompiledModel['Compensator'], 'modelOutput':CompiledModel['Compensator modelOutput']}, X_norm, error_pred_Norm, **input_kwargs_ANN)
    # Norms = {'Error mean':error_mean, 'Error std':error_std, 'Input mean':X_mean, 'Input std':X_std}
    Compensator = ANN.train({'Model':CompiledModel['Compensator'], 'modelOutput':CompiledModel['Compensator modelOutput']}, X, error_pred, applyNormalization = True, **input_kwargs_ANN)

    # Augment model with ANN results
    Model = Base_Model['Model']
    Model.update({'Compensator':Compensator['Model']})
    # Model.update({'Compensator Norms':Norms})
    Model.update({'Compensator Inputs':CompiledModel['Compensator Inputs']})
    Base_Model.update({'Model':Model})
    Additional = Base_Model['Additional']
    Additional.update({'Compensator':Compensator['Additional']})

    return Base_Model



def AugmentExistingPolyModel(HybridModel, PolyModel, x_train, y_train, compensator_x_vec, compensatorCompileKwargs = None, compensatorTrainKwargs = None):
    '''Function to add a hybrid component to an existing (i.e. already trained) polynomial model. 
    
    :param HybridModel: (Initialized) SysID.Model object representing the hybrid model. The trained hybrid model will be placed in this object. 
    :param PolyModel: The polynomial model, as a SysID.Model object, upon which the hybrid component should be added.
    :param x_train: Training input data, as a Pandas DataFrame/Series. Shape of NxM where N = Number of data points and M = number of inputs
    :param y_train: Training target data, as a Pandas DataFrame/Series. Shape of NxP where N = Number of data points and P = number of model outputs
    :param compensator_x_vec: List of variables (i.e. columns) in <X> to be used as ANN inputs. Default = None. Thus, all columns of X will be used. 
    :param compensatorCompileKwargs: A dictionary of keyword arguments to be passed onto the ANN.compile function for compiling the ANN compensator. See stepwise_regression.py for further documentation. 
    :param compensatorTrainKwargs: A dictionary of keyword arguments to be passed onto the ANN.train function for training the ANN compensator. See ANN.py for further documentation. 
    :return: Modified HybridModel with hybrid components included, as a SysID.Model object
    '''

    # Extract PolyModel
    trainedPolyModel = PolyModel.TrainedModel['Model'].copy()
    compiledPolyModel = PolyModel.CompiledModel.copy()

    # Reshape y_train if necessary
    if len(y_train.shape) <= 1:
        y_train = np.matrix(y_train).T

    # Compile Compensator
    if compensator_x_vec is None:
        compensator_x_vec = list(x_train.columns)
    CompiledModel_ANN = ANN.compile(x_train[compensator_x_vec], y_train, **compensatorCompileKwargs)
    compiledPolyModel.update({'Compensator':CompiledModel_ANN['Model'], 'Compensator Inputs':compensator_x_vec, 'Compensator modelOutput':CompiledModel_ANN['modelOutput']})
    # compiledPolyModel.update({'Compensator Inputs':compensator_x_vec})
    # for k, v in CompiledModel_ANN.keys():
    #     compiledPolyModel.update({'Compensator {}'.format(k):v})
    HybridModel.ModelStateHistory.append('Compiled')

    # Train Compensator
    Base_pred = trainedPolyModel['A']*trainedPolyModel['Parameters']

    print('[ INFO ] Training ANN Compensator for base model prediction error')
    error_pred = y_train.reshape(Base_pred.shape) - Base_pred
    # Normalize input 
    # (error_pred_Norm, error_mean, error_std) = ANN.Normalize(error_pred, returnStats=True)
    # (X_norm, X_mean, X_std) = ANN.Normalize(x_train[compensator_x_vec], returnStats=True)
    # Compensator = ANN.train(CompiledModel_ANN, X_norm, error_pred_Norm, **compensatorTrainKwargs)
    # Norms = {'Error mean':error_mean, 'Error std':error_std, 'Input mean':X_mean, 'Input std':X_std}
    Compensator = ANN.train(CompiledModel_ANN, x_train[compensator_x_vec], error_pred, applyNormalization = True, **compensatorTrainKwargs)

    # Combine poly and compensator models to HybridModel
    Model = trainedPolyModel
    Model.update({'Compensator':Compensator['Model']})
    # Model.update({'Compensator Norms':Norms})
    Model.update({'Compensator Inputs':compensator_x_vec})
    Additional = PolyModel.TrainedModel['Additional']
    Additional.update({'Compensator':Compensator['Additional']})

    HybridModel.TrainedModel = {}
    HybridModel.TrainedModel.update({'Model':Model})
    HybridModel.TrainedModel.update({'Additional':Additional})

    # Set model states
    HybridModel.CompiledModel = compiledPolyModel
    HybridModel.CurrentModel = HybridModel.TrainedModel
    HybridModel.ModelState = 'Trained'
    HybridModel.ModelStateHistory.append('Trained')

    return HybridModel



def predict(TrainedModel, X, decompose_prediction = False):
    ''' Function to make predictions with the trained hybrid model. Relies on stepwise_regression.py and ANN.py

    :param TrainedModel: The trained hybrid model, output of the function <train()>
    :param X: Input data, as a Pandas DataFrame/Series, from which predictions should be made with shape KxM where K = Number of data points and M = number of inputs. Note that the columns, M, should correspond (in name) to those used for training
    :param decompose_prediction: Boolean indicating if the individual contributions of the underlying polynomial and compensator should be outputted in addition to the total prediction. Default = False (i.e. only total prediction is outputted). If True, a dictionary is outputted instead, with entires:
            
            - 'Prediction' = total prediction
            - 'Base' = polynomial prediction
            - 'Compensator' = ANN prediction of error between targets and 'Base' 
            - 'Forecast error variance' = Total forecasting error variance (informs prediction
                interval for associated predictions)
            - 'Error variance polynomial' = Component of forecasting error due to polynomial
            - 'Error variance compensator' = Component of forecasting error due to compensator
    
    :return: Hybrid model prediction. See documentation for input <decompose_prediction> for details on all possible function outputs. Also outputs Forecasting error variance. 
    '''    
    # Make base model prediction
    pred_base, sig_base = stepwise_regression.predict(TrainedModel, X)

    # Normalize inputs to ANN
    # Norms = TrainedModel['Model']['Compensator Norms']
    # X_norm = ANN.Normalize(X[TrainedModel['Model']['Compensator Inputs']], X_bar = Norms['Input mean'], X_std = Norms['Input std'])
    # Make prediction on modelling error
    CompensatorModel = {'Model':TrainedModel['Model']['Compensator'], 'Additional':TrainedModel['Additional']['Compensator']}
    # pred_error_norm, sig_error = ANN.predict(CompensatorModel, X_norm)

    # # Convert modelling error back to appropriate units
    # pred_error = ANN.DeNormalize(pred_error_norm, (Norms['Error mean'], Norms['Error std']))
    # sig_error = sig_error * np.array(np.square(Norms['Error std']))
    pred_error, sig_error = ANN.predict(CompensatorModel, X[TrainedModel['Model']['Compensator Inputs']])

    # 'Compensate' for polynomial prediction error
    prediction = pred_base + pred_error.reshape(pred_base.shape)
    sig = np.array(sig_base) + np.array(sig_error).reshape(sig_base.shape)

    if decompose_prediction:
        return {'Prediction':prediction, 'Base':pred_base, 'Compensator':pred_error, 'Forecast error variance':sig, 'Error variance polynomial':sig_base, 'Error variance compensator':sig_error}
    else:
        return prediction, sig



def evaluate(TrainedModel, X, Y, showPlots=True, method_inputSensitivity = 'perturbation', inputSensitivityKwargs = {}, weightSensitivityKwargs = {}):
    ''' Function to evaluate the quality of the hybrid model by investigating the quality of the polynomial and ANN separately. Relies on stepwise_regression.py and ANN.py
    
    :param TrainedModel: The trained hybrid model, output of the function <train()>
    :param X: Training input data, as a Pandas DataFrame/Series with shape NxM where N = Number of data points and M = number of inputs
    :param Y: Training target data, as a Pandas DataFrame/Series with shape NxP where N = Number of data points and P = number of model outputs
    :param showPlots: Boolean to indicate if plots should be shown. Default = True.
    :return: Dictionary containing the evaluation results, split into entries of  'Polynomial' and 'Compensator' which contain the evaluations of each respectively. See documentation corresponding documentation for details on the individual evaluation entries. 
    '''
    # Create evaluation dictionary
    evaluation = {}

    # Evaluate polynomial
    print('[ INFO ] Evaluating polynomial component of hybrid model...')
    poly_evaluation = stepwise_regression.evaluate(TrainedModel, X, Y, showPlots='plot')
    # Add polynomial evaluation
    evaluation.update({'Polynomial':poly_evaluation})

    # Evaluate compensator
    print('[ INFO ] Evaluating compensator...')
    comp_Model = {'Model':TrainedModel['Model']['Compensator'], 'Additional':TrainedModel['Additional']['Compensator']}
    comp_evaluation = ANN.evaluate(comp_Model, X[TrainedModel['Model']['Compensator Inputs']], Y, showPlots='plot',
                                   method_inputSensitivity = method_inputSensitivity, inputSensitivityKwargs = inputSensitivityKwargs, weightSensitivityKwargs = weightSensitivityKwargs)
    # Add compensator evaluation
    evaluation.update({'Compensator':comp_evaluation})

    if showPlots:
        ANN.plt.show()

    return evaluation



def summary(Status, Model):
    '''Function to print a summary of the model to the command line
    
    :param Status: State of the model, inherited from SysID class. 
    :param Model: Model corresponding to the state (e.g. TrainedModel corresponds to Status = 'trained')
    :return: None
    '''    
    if Status.lower() in ('compiled', 'trained'):
        # Polynomial summary
        print('{:{fill}^65}'.format('Polynomial Model', fill='='))
        stepwise_regression.summary(Status, Model)

        print('\n')

        # ANN Summary
        if Status.lower() == 'compiled':
            Compensator = Model['Compensator']
        else:
            Compensator = Model['Model']['Compensator']
        ANN.summary(Status, {'Model':Compensator})
    else:
        print('{:^65}'.format('Model not compiled!'))

    return None



def save(path, model):
    '''Function to save trained hybrid model, assumes SysID.Model object format 

    :param path: Save directory
    :param model: TrainedModel to save
    :return: None
    '''
    # Need to split model into polynomial and ANN parts
    Compensator = {'Model':model['Model']['Compensator']}
    # Add compensator normalization and input vector to additional
    compAdd = model['Additional']['Compensator']
    for k in ('Compensator Inputs',):
        compAdd.update({k:model['Model'][k]})
    # Update compensator
    Compensator.update({'Additional':compAdd})

    # We do not want to modify the original dictionary. Typically we could
    # make a deepcopy, but iis does not work due to the use of tf models in the dictionary
    # therefore, we need to define a new dictionary
    polyModelModel = {}
    for k, v in model['Model'].items():
        if k not in ('Compensator', 'Compensator Inputs'):
            polyModelModel.update({k:v})
    polyModelAdd = {}
    for k, v in model['Additional'].items():
        if k not in ('Compensator'):
            polyModelAdd.update({k:v})

    polyModel = {'Model':polyModelModel, 
                 'Additional':polyModelAdd}

    # Use method-specific saving strategies
    stepwise_regression.save(path, polyModel)
    ANN.save(path, Compensator)

    return None



def load(path):
    '''Function to load hybrid model from directory, uses SysID.Model object format

    :param path: Model directory
    :return: TrainedModel in SysID.Model format
    '''
    polyModel = stepwise_regression.load(path)
    compModel = ANN.load(path)

    # Build Hybrid model from poly and comp components
    polyModel['Model'].update({'Compensator':compModel['Model']})
    for k in ('Compensator Inputs',):
        polyModel['Model'].update({k:compModel['Additional'][k]})

    polyModel['Additional'].update({'Compensator':compModel['Additional']})

    return polyModel



def copy(model):
    '''Function to make a copy of hybrid trained model
    
    :param model: SysID.Model.TrainedModel object of the trained model
    :return: Copy of model 
    '''    
    return stepwise_regression.copy(model)
