'''
Script with necessary functions to build, train and make predictions with an Artificial Neural Network model
Backend for the SysID class 'ANN' option. 

Written by Jasper van Beers
Created: 12-01-2021
Last edit: 02-06-2021
'''

# ================================================================================================================================ #
# Global Imports
# ================================================================================================================================ #
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.keras.backend as KB #pylint: disable=F0401
import os
import pickle as pkl



class StoreWeightsOnTrain(tf.keras.callbacks.Callback):
    '''
    Custom Callback to store weights during training after each epoch.
    '''
    def __init__(self):
        super().__init__()
        self.SysIDWeightHistory = []
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_weights = self.model.get_weights()
        self.SysIDWeightHistory.append(epoch_weights)



def Normalize(X, returnStats = False, X_bar = None, X_std = None):
    '''Utility function to normalize the inputs to the ANN to ~N(0, 1), based on the average and standard deviations of the data. 
    
    :param X: Data to be normalized with shape [N x M] where N = number of observations and M = number of states to be normalized. 
    :param returnStats: Boolean that dictates if the mean and standard deviation should be returned by the function, default is False
    :param X_bar: Force a mean of X. If None, then the mean will be computed using numpy.nanmean() method. Default = None
    :param X_std: Force a Standard deviation of X. If None, then the standard deviation will be computed using numpy.nanstd() method. Default = None
    :return: Normalized values of X. If returnStats is true, then also return X_bar and X_std
    '''    
    if X_bar is None:
        X_bar = np.nanmean(X, 0)
    if X_std is None:
        X_std = np.nanstd(X, 0)
    Norm = (X - X_bar)/X_std
    if returnStats:
        Norm = (Norm, X_bar, X_std)
    return Norm



def DeNormalize(NX, stats, X = None):
    '''Utility function to transform normalized data to their original values, based on the average and standard deviations of the original data. A complimentary function to <Normalize()>. 
    
    :param NX: Normalized values of the data
    :param stats: [mean, std] where <mean> is the average of X and <std> the standard deviation of X. 
    :param X: Original data used for normalization. If provided, the statistics (i.e. mean and std) derived from X take priority over the inputted <stats>. Default = None. 
    :return: 'De-Normalized' values of NX. 
    '''
    # Prefer deriving mean and std from X directly, if provided
    if X is not None:
        DeNorm = NX*np.nanstd(X, 0) + np.nanmean(X, 0)
    else:
        DeNorm = NX*stats[1] + stats[0]
    return DeNorm



def pearsonCorr(a, b):
    '''Utility function to compute the pearson correlations between two 2-D arrays along the first axis

    :param a: 2-D array
    :param b: 2-D array
    :return: Pearson correlations between a and b along first axis
    '''
    s_m_ab = np.sum((a - np.nanmean(a, axis = 0))*(b - np.nanmean(b, axis = 0)), axis = 0)
    s_v_ab = np.sqrt(np.sum(np.square((a - np.nanmean(a, axis = 0))), axis = 0)*np.sum(np.square((b - np.nanmean(b, axis = 0))), axis = 0))
    return s_m_ab/s_v_ab



def makeNdArray(x, reshape = False, reshapeN = 1):
    '''Utility function to convert 2-D data into numpy arrays and, if selected, generate a 3-D array from a 2-D array compatible with the expected tensorflow input dimensions (i.e. batch x dim1 x dim2). 
    
    :param x: 2-D data (pandas DataFrame, Series or numpy array, matrix) to be converted
    :param reshape: Boolean to indicate if data should be reshaped into a 3-D array. Default = False
    :return: x = Converted 2-D data, n = Number of columns in x
    '''    
    # Infer type of x and y and convert shapes accordingly
    try:
        # Check DataFrame
        n = len(x.columns)
        x = x.to_numpy()
    except AttributeError:
        # Check Series
        try:
            _ = x.name
            n = 1
            x = x.to_numpy()
        # Is array
        except AttributeError:
            if len(x.shape) > 1:
                n = x.shape[-1]
            else:
                n = 1

    if reshape:
        x = np.array(x).reshape(-1, reshapeN, n)

    return x, n



def _build_FNN_Layers(model, numLayers, numNeurons, activation, dropout):
    '''Build individual layers of a feed forward neural network 
    
    :param model: Keras model of the neural network, before compilation
    :param numLayers: number of layers to build, as int
    :param numNeurons: number of neurons per hidden layer (constant across all layers), as int
    :param activation: activation function per layer (constant across all layers), see keras documentation for possible activation functions: https://keras.io/api/layers/activations/ (accessed 08-03-2021)
    :param dropout: neuron dropout rate [0, 1) for each layer (constant across all layers), as float
    :return: updated Keras model of the neural network with new layers
    '''    
    for lyr in range(numLayers):
        model.add(keras.layers.Dense(numNeurons, activation=activation))
    model.add(keras.layers.Dropout(dropout))
    return model



def _build_RNN_Layers(model, numLayers, numNeurons, activation, dropout):
    '''Build single-layer RNN neural network
    
    :param model: Keras model of the neural network, before compilation
    :param numLayers: Number of layers, unused by RNN, but present for compatibility
    :param numNeurons: Number of neurons per layer (constant across all layers), as int
    :param activation: activation function per layer (constant across all layers), see keras documentation for possible activation functions: https://keras.io/api/layers/activations/ (accessed 08-03-2021)
    :param dropout: neuron dropout rate [0, 1) for each layer (constant across all layers), as float
    :return: updated Keras model of the neural network with RNN layer
    '''    
    model.add(keras.layers.SimpleRNN(numNeurons, activation=activation, recurrent_dropout=dropout))
    # model.add(keras.layers.SimpleRNN(numNeurons, activation=activation, recurrent_dropout=dropout, return_sequences = True))
    return model



def _parse_kwargs(var, value_if_absent, kwargs):
    '''Utility function to check inputted keyword arguments and replace them with defaults if absent
    
    :param var: keyword argument to check 
    :param value_if_absent: default value, in case keyword argument was not passed in higher level function
    :param kwargs: dictionary of all passed keyword arguments  
    :return: Keyword argument value
    ''' 
    if var in kwargs.keys():
        value = kwargs[var]
    else:
        value = value_if_absent
    return value



def _check_depth(x, curr_depth = 0):
    '''Utility function to compute depth of nested lists, assuming bottom layer is a dictionary
    
    :param x: Nested lists
    :param curr_depth: Current depth level, default is 0. 
    :return: Depth level of nested lists as int. e.g. A list structure of: [[[obj]]] has a depth of 3 
    ''' 
    # There is still a list-like object, so add to current depth
    try:
        curr_depth += len(x[0])/len(x[0])
        curr_depth = _check_depth(x[0], curr_depth)
    # No more list-like objects, so we expect to hit a dictionary
    except KeyError:
        pass
    return curr_depth



def compile(InputData, TargetData, modelOutput = 'point', num_ensembles = 1, ANN_type = 'FNN', num_hidden_layers = 1, hidden_activation = 'relu', num_hidden_neurons = 100, CustomStructure = None, **kwargs):
    '''Function to compile ANN model
    
    :param InputData: Pandas DataFrame or numpy array of ANN inputs with shape [N x n] where N = number observations, n = number inputs 
    :param TargetData: Pandas DataFrame or numpy array of ANN targets with shape [N x m] where m = number of outputs
    :param modelOuput: String of the type of prediction the ANN model should output. Options are:
            - point = Point predictions (i.e. x -> y_pred) where y_pred denotes the ANN prediction of TargetData
            - interval = Interval predictions (i.e. x -> [y_L, y_U]) where y_L and y_U denote the upper and lower prediction bounds for the each point in TargetData with confidence 1-alpha (default alpha = 0.05 -> 95% confidence interval). point predictions of y_pred may be taken as the average of y_L and y_U. NOTE: Currently, only 1-D TargetData is supported for 'interval'
            - NOTE: Users can still obtain prediction intervals even if 'point' is chosen through bootstrapping model ensembles. This is done during the training of the network. See documentation of <train> for more details. 
    
    :param num_ensembles: Number of model instances, as int, to create for model regularization. Default = 1 (i.e. no ensembles)
    :param ANN_Type: Neural network type. Currently only Feed forward (FNN) and recurrent (RNN) neural networks are supported. Default = 'FNN'
    :param num_hidden_layers: Number of hidden layers. Ignored if ANN_Type = 'RNN' or if CustomStructure is not None.
    :param hidden_activation: Hidden layer activation function. Ignored if CustomStructure is not None.
    :param num_hidden_neurons: Number of neurons per layer (constant across layers). Ignored if CustomStructure is not None.
    :param CustomStructure: List of (list of) dictionaries describing the (FNN) network architecture. Each dictionary represents a layer of the network and is required to have the keys
                - 'Number Neurons' = Number of neurons in said layer,
                - 'Activation' = Activation function of the layer,
                - 'Dropout' = Dropout rate for the layer
            The number of dictionaries corresponds to the number of layers in the network. If a list of lists of dictionaries is passed (i.e. structure following: [[{}]]), then each inner list corresponds
            to a different neural network (i.e. model ensemble). Thus, a structure following [[{...}], [{...}], [{...}]] is a three network model. 
    
    :param **kwargs: Additional keyword arguments relevant for tensorflow models. Namely,
            - optimizer = Model optimizer,
            - loss = Model loss function,
            - metrics = Metrics to evaluate model,
            - dropout = Neuron dropout rate
    :return: Dictionary containing list of ANN models following SysID.Model format. The models can be accessed through the key 'Model'. Order corresponds to order of models in CustomStructure, if provided. 
    '''
    # Check inputs
    accepted_ANNs = ['FNN', 'RNN']
    accepted_Outputs = ['point', 'interval']
    if ANN_type not in accepted_ANNs:
        raise ValueError('Invalid ANN_type: "{}". Acceptable ANN_types are: {}'.format(ANN_type, accepted_ANNs))
    if modelOutput.lower() not in accepted_Outputs:
        raise ValueError('Invalid modelOutput: "{}". Expected "point" or "interval".')

    # Specify compilation parameters
    print('[ INFO ] Compiling <{}> model ensembles of type <{}> with output method <{}>'.format(num_ensembles, ANN_type, modelOutput))

    # Check CustomStructure, if passed:
    if CustomStructure:
        depth = int(_check_depth(CustomStructure, 0))

        if depth == 1:
            # Broadcast customStructure
            CustomStructure = [CustomStructure]*num_ensembles
        elif depth == 2:
            # Check if number of elements matches number of ensembles
            if len(CustomStructure) != num_ensembles:
                raise ValueError('Number of model structures in CustomStructure does not match number of ensembles ({} != {})'.format(len(CustomStructure), num_ensembles))
        else:
            raise ValueError('Invalid CustomStructure depth. Expected depth between [1, 2], current depth is {}'.format(depth))

    # Raise warning if 'interval' is selected and num_ensembles < 5
    if num_ensembles < 5 and modelOutput == 'interval':
        print('[ WARNING ] Attempting to estimate prediction intervals with less than 5 models may produce poor results. It is recommended that at least 5-10 models are used.')

    # Parse keyword arguments
    for k in kwargs.keys():
        if k not in ('optimizer', 'metrics', 'dropout', 'loss'):
            raise KeyError('Unexpected keyword argument: "{}" in compile'.format(k))

    opt = _parse_kwargs('optimizer', 'adam', kwargs)
    metrics = _parse_kwargs('metrics', ['mae'], kwargs)
    dropout = _parse_kwargs('dropout', 0.0, kwargs)

    # Obtain input and output shapes 
    # Convert to arrays
    try: 
        x_vec = list(InputData.columns)
    except AttributeError:
        x_vec = None
    InputData, n = makeNdArray(InputData)
    TargetData, m = makeNdArray(TargetData)

    models = []
    for mdl in range(num_ensembles):
        # Initialize Model
        model = keras.Sequential()

        # (Linear) Input Layer
        model.add(keras.Input(shape=(1, n)))

        # Check if FNN or RNN type
        buildHidden = {'FNN':_build_FNN_Layers, 'RNN':_build_RNN_Layers}

        # Build hidden layer(s)
        if CustomStructure:
            for lyr in CustomStructure[mdl]:
                lyr_neurons = lyr['Number Neurons']
                lyr_activation = lyr['Activation']
                lyr_dropout = lyr['Dropout']
                model = buildHidden[ANN_type](model, 1, lyr_neurons, lyr_activation, lyr_dropout)
        else:
            model = buildHidden[ANN_type](model, num_hidden_layers, num_hidden_neurons, hidden_activation, dropout)

        # Output Layer, activation = None -> Linear activation (since we are creating regression models)
        if modelOutput.lower() == 'interval':
            lossFunc = _parse_kwargs('loss', _loss_deepNetPI(), kwargs)
            model.add(keras.layers.Dense(2, activation = None))
        else:
            lossFunc = _parse_kwargs('loss', 'mean_squared_error', kwargs)
            model.add(keras.layers.Dense(m, activation = None))

        # Compile model
        model.compile(optimizer=opt, loss=lossFunc, metrics = metrics)
        
        # Append model to ensemble
        models.append(model)

        out = {'Model':models, 'modelOutput':modelOutput.lower(), 'x_vec':x_vec}
        # if modelOutput.lower() == 'interval':
        #     out.update({'customLoss':lossFunc})

    return out



def train(CompiledModel, x, y, applyNormalization = False, ensembleTrainingRatio = 0.8,
          addStackedGeneralization = False, stackedKwargs = {},
          bootstrapPIs = False, bootstrapPI_kwargs = {}, 
          storeTrainingWeights = True, **kwargs):
    '''Function to train ANN model(s)
    
    :param CompiledModel: SysID.Model.CompiledModel format, Dictionary containing list of ANN models under key 'Model' (i.e. output of function <compile()>)
    :param x: Training input data. Shape = (N, 1, n) -> N = number observations, n = number inputs
    :param y: Training target data. Shape = (N, 1, m) -> m = number outputs
    :param applyNormalization: Boolean to normalize the input and target data such that mean = 0 and standard deviation = 1. Default = False. 
    :param ensembleTrainingRatio: Ratio of training data set to be used for ensemble training data subsets. Only relevant for ensembles. Default = 0.8 -> 80% of the training data is used for the ensemble training subset. 
    :param addStackedGeneralization: Boolean to indicate if stacked generalization should be used to combine model ensemble outputs or not. Default = False; stacked generalization network will not be trained. If True, the stacked model is stored under the key 'Additional' -> 'Stacked' and can be accessed using Model['Additional']['Stacked']
    :param stackedKwargs: Only relevant if addStackedGeneralization is True. Keyword arguments to be passed to the tensorflow training function (i.e. model.fit()) for the stacked generalization ANN
    :param bootstrapPIs: Boolean indicating if the prediction intervals (PIs) should be calculated with network outputs through the bootstrap approach (see [1]). This approach requires a separate NN to be trained to estimate errors arising from noise, as such, use of PIs needs to be specified during training. Ignored if modelOutput='interval' from <compile>
    :param bootstrapPI_kwargs: Only relevant if bootstrapPIs is True. Dictionary of keyword arguments to be passed to the PI NN for model compilation and training. Default is an empty dictionary (i.e. default values in <_getBootstrapPI> function)
    :param storeTrainingWeights: Boolean indicating if the weights, after each epoch of training, should be stored. This is a prerequisite for weight sensitivity analyses. Default = True. 
    :param **kwargs: Keyword arguments to be passed to the tensorflow training function (i.e. model.fit())
    :return: Dictionary containing TrainedModel in SysID.Model format. 

    [1] - Khosravi, A., Nahavandi, S., Creighton, D., & Atiya, A. F. (2011). Comprehensive review of neural 
        network-based prediction intervals and new advances. IEEE Transactions on neural networks, 22(9), 1341-1356.
    '''

    # Raise warning based on input combinations
    if addStackedGeneralization and bootstrapPIs:
        print('[ WARNING ] Estimating model prediction intervals with a stacked generalization output may produce unreliable PIs')

    # Raise warning if bootstrapPIs is True and modelOutput = 'interval' (from <compile>)
    # Raise error if stacked generalization is used 
    if CompiledModel['modelOutput'] == 'interval' and bootstrapPIs:
        print('[ WARNING ] User selected to bootstrap prediction intervals but the compiled model is already set up to output prediction intervals! Ignoring bootstrapPIs')
        bootstrapPIs = False
        if addStackedGeneralization:
            print('[ WARNING ] Estimating model prediction intervals with a stacked generalization output may produce unreliable PIs')

    if 'x_vec' in CompiledModel.keys():
        if CompiledModel['x_vec'] is not None:
            x = x[CompiledModel['x_vec']]

    normalizer = {'x_bar':None, 'x_std':None, 'y_bar':None, 'y_std':None}
    if applyNormalization:
        x, x_bar, x_std = Normalize(x.copy(), returnStats=True)
        y, y_bar, y_std = Normalize(y.copy(), returnStats=True)
        normalizer.update({'x_bar':x_bar, 'x_std':x_std, 'y_bar':y_bar, 'y_std':y_std})

    # Check if weights should be stored during training, after each epoch
    WeightStorage = StoreWeightsOnTrain()
    if storeTrainingWeights:
        # Add WeightStorage to passed callbacks, if present
        if 'callbacks' in kwargs.keys():
            currentCallbacks = kwargs['callbacks']
            currentCallbacks.append(WeightStorage)
        else:
            kwargs.update({'callbacks':[WeightStorage]})

    # Extract model(s)
    models = CompiledModel['Model']
    # Infer type of x and y and convert shapes accordingly
    x, n = makeNdArray(x, reshape = True)
    y, m = makeNdArray(y, reshape = True)

    # If using stacked generalization, save part of the data set for training the stacker.
    if addStackedGeneralization or bootstrapPIs:
        print('[------------------------------]')
        print('[ INFO ] Splitting training data into training subsets...')
        idx_bool = np.ones(len(x))
        idx_D2 = np.sort(np.random.choice(len(x), int(0.2*len(x)), replace = False))
        x_D2 = x[idx_D2]
        y_D2 = y[idx_D2]
        idx_bool[idx_D2] = 0
        new_train_idx = np.where(idx_bool)[0]
        x = x[new_train_idx]
        y = y[new_train_idx]

    if len(models) > 1:
        N_training = int(ensembleTrainingRatio * len(x))
        for i, mdl in enumerate(models):
            print('[------------------------------]')
            print('[ INFO ] Training model [{}/{}]'.format(i+1, len(models)))
            # Split dataset into ensemble training subsets
            indices_training = np.sort(np.random.choice(len(x), N_training, replace = False))
            # Train model using tensorflow model.fit method. 
            mdl.fit(x[indices_training], y[indices_training], **kwargs)
    else:
        print('[------------------------------]')
        print('[ INFO ] Training model')
        # Train model using tensorflow model.fit method. 
        models[0].fit(x, y, **kwargs)

    try:
        num_epochs = kwargs['epochs']
    except KeyError:
        num_epochs = 1

    weightHistory = np.array(WeightStorage.SysIDWeightHistory, dtype='object').reshape(len(models), num_epochs, -1)
    additional = {'modelOutput':CompiledModel['modelOutput'], 
                  'normalizer':normalizer, 
                  '_normalized':applyNormalization,
                  'weightHistory':weightHistory}
    if 'x_vec' in CompiledModel.keys():
        x_vec = CompiledModel['x_vec']
        additional.update({'x_vec':x_vec})
    # if 'customLoss' in CompiledModel.keys():
    #     additional.update({'customLoss':CompiledModel['customLoss']})

    if addStackedGeneralization or bootstrapPIs:
        print('[------------------------------]')
        # Make predictions on second data set
        if CompiledModel['modelOutput'] == 'interval':
            ensemble_pred = np.zeros((x_D2.shape[0], len(models)))
            for i, mdl in enumerate(models):
                # pred_i[:, 0] = lower bound prediction, pred_i[:, 1] = upper bound prediction
                pred_i = mdl.predict(x_D2).reshape(-1, 2)
                # take mean of lower and upper bound prediction as target prediction
                ensemble_pred[:, i] = np.nanmean(pred_i, axis = 1).reshape(-1)

        else:
            ensemble_pred = np.zeros((x_D2.shape[0], len(models)))
            for i, mdl in enumerate(models):
                ensemble_pred[:, i] = mdl.predict(x_D2).reshape(-1)
                
        # Train NN to estimate noise error variance for bootstrap PIs, if selected
        if bootstrapPIs:
            BootstrapNN = _getBootstrapPI(models, x_D2, y_D2, boostrapKwargs=bootstrapPI_kwargs)
            additional.update({'BootstrapPINN':BootstrapNN})

    return {'Model':models, 'Additional':additional}


@tf.autograph.experimental.do_not_convert
def _bootstrapLossCBS(y_true, y_pred):
    '''Loss function to implicitly approximate PIs when using the bootstrap approach
    See:    A. Khosravi, S. Nahavandi, D. Creighton and A. F. Atiya, 
            "Comprehensive Review of Neural Network-Based Prediction Intervals and New Advances," 
            in IEEE Transactions on Neural Networks, vol. 22, no. 9, pp. 1341-1356, Sept. 2011, 
            doi: 10.1109/TNN.2011.2162110.

    :param y_true: Targets
    :param y_pred: Predictions
    :return: CBS (bootstrap) loss function
    '''
    r2 = KB.flatten(y_true)
    sige2 = KB.flatten(y_pred)
    # try:
    #     tf.debugging.check_numerics(sige2, message = 'Checking sige2')
    # except Exception as e:
    #     assert "Checking sige2 : Tensor has nan or inf values" in e.message
    # try:
    #     tf.debugging.check_numerics(r2, message = 'Checking r2')
    # except Exception as e:
    #     assert "Checking r2 : Tensor has nan or inf values" in e.message
    try:
        tf.debugging.check_numerics(r2, message = 'Checking r2')
    except Exception as e:
        r2_nan_idxs = tf.math.is_nan(r2)
        # r2[r2_nan_idxs] = 0
        r2[r2_nan_idxs] = tf.math.reduce_max(r2)
    try:
        tf.debugging.check_numerics(sige2, message = 'Checking sige2')
    except Exception as e:
        sige2_nan_idxs = tf.math.is_nan(sige2)
        # sige2[sige2_nan_idxs] = 100000
        sige2[sige2_nan_idxs] = tf.math.reduce_max(r2)

    CBS = 0.5*KB.sum((KB.log(sige2) + r2/sige2))
    return CBS



def _getBootstrapPI(TrainedModels, x, y, boostrapKwargs = {}):
    '''Function to get ANN prediction intervals (PIs) through bootstrapping. 
    
    The 'bootstrap' NN implicitly estimates the noise variance of the data. Modelling uncertainty is obtained through the variance of the model ensemble predictions. 
    
    :param TrainedModels: Dictionary containing ANN models and additional information in SysID.Model format. Output of <train>.
    :param x: Input data to be used to train bootstrap PIs
    :param y: Targets to be used to train bootstrap PIs
    :param bootstrapKwargs: Keyword arguments to pass to the function <compile> to create bootstrap ANN
    :return: Keras model of the Bootstrap PI ANN. 
    '''
    # Infer type of x and convert shape accordingly
    x, n = makeNdArray(x, reshape=True)
    y, m = makeNdArray(y, reshape=True)

    print('[ INFO ] Estimating prediction intervals through bootstrapping...')

    # Check how TrainedModels is passed
    try:
        models = TrainedModels['Model']
    except TypeError:
        models = TrainedModels
    
    # Check number of ensembles
    B = len(models)
    if B <= 1:
        raise ValueError('Running bootstrap PIs requires an ensemble of models. Only one model is passed.')
    elif B < 5:
        print('[ WARNING ] Running bootstrap PIs with less than 5 models may produce poor results. It is recommended that at least 5-10 models are used.')
    
    var_y, y_avg = _bootstrapGetVarY(models, x)
    var_e_hat = np.square((y.reshape(-1, 1) - y_avg)) - var_y
    # Variance can only be positive so
    # apply max(var_e_hat, 0) element wise
    idx_negative = np.where(var_e_hat <= 0)[0]
    r2_i = var_e_hat.copy()
    if len(idx_negative):
        r2_i[idx_negative] = 0
    
    # default values for configuring Bootstrap ANN
    buildKwargs = {'numLayers':1, 'numNeurons':25, 'activation':'relu', 'dropout':0.1}
    compileKwargs = {'optimizer':'adam', 'loss':_bootstrapLossCBS, 'metrics':['mae']}
    trainKwargs = {'batch_size':400, 'epochs':200, 'verbose':2}

    # parse kwargs, replace defaults if user specifies different values
    for k, v in boostrapKwargs.items():
        if k in buildKwargs.keys():
            buildKwargs.update({k:v})
        elif k in compileKwargs.keys():
            compileKwargs.update({k:v})
        elif k in trainKwargs.keys():
            trainKwargs.update({k:v})
        else:
            raise KeyError('Unexpected keyword argument "{}" in <bootstrapKwargs> of function <_getBootstrapPI>'.format(k))

    # Build bootstrap ANN to implicitly approximate PIs 
    model = keras.Sequential()
    model.add(keras.Input(shape=(1, n)))
    model = _build_FNN_Layers(model, **buildKwargs)
    model.add(keras.layers.Dense(m, activation = 'exponential'))
    model.compile(**compileKwargs)

    model.fit(x, r2_i, **trainKwargs)

    return model



# NOTE: Keras backend (KB) functions: https://keras.rstudio.com/articles/backend.html
def _loss_deepNetPI(alpha=0.05, softenFactor = 160, lambdaFactor = 1):
    '''Decorator for the loss function of the quality driven direct-PI method [1]. This allows for the specification of the hyperparameters while having a compatible loss function configuration with tensorflow. 

    :param alpha: float, significance level where (1 - alpha) gives the confidence level for the prediction intervals
    :param softenFactor: Hyperparameter denoting the instensity of the softening of the loss function non-linearities. Default = 160 following from [1]
    :param lambdaFactor: Hyperparameter scaling the penalty of invalid prediction intervals. Default = 1 following from [1]
    :return: QD loss function, compatible for use with tensorflow. 

    [1] Pearce, Tim, Alexandra Brintrup, Mohamed Zaki, and Andy Neely. 
        "High-quality prediction intervals for deep learning: A distribution-free, 
        ensembled approach." In International Conference on Machine Learning, 
        pp. 4075-4084. PMLR, 2018.
    '''


    def QD_loss(y_true, y_pred):
        '''QD loss function [1]

        :param y_true: Training targets
        :param y_pred: Model predictions of targets
        :return: Loss value

        [1] Pearce, Tim, Alexandra Brintrup, Mohamed Zaki, and Andy Neely. 
            "High-quality prediction intervals for deep learning: A distribution-free, 
            ensembled approach." In International Conference on Machine Learning, 
            pp. 4075-4084. PMLR, 2018.
        
        '''


        alpha_ = KB.constant(alpha)
        softenFactor_ = KB.constant(softenFactor)
        lambdaFactor_ = KB.constant(lambdaFactor)

        if len(y_pred.shape) == 2:
            y_L = y_pred[:, 0]
            y_U = y_pred[:, 1]
            y = y_true[:, 0]
        elif len(y_pred.shape) == 3:
            y_L = y_pred[:, :, 0]
            y_U = y_pred[:, :, 1]
            y = y_true[:, :, 0]
        else:
            raise ValueError('[ ERROR ] Incompatible shape of target data with <_loss_deepNetPI>. Expected dim 2 or 3.')


        kH_L = KB.maximum(KB.zeros_like(y), KB.sign(y - y_L))
        kH_U = KB.maximum(KB.zeros_like(y), KB.sign(y_U - y))
        kH = kH_L*kH_U

        kS_L = KB.sigmoid((y - y_L)*softenFactor_)
        kS_U = KB.sigmoid((y_U - y)*softenFactor_)
        kS = kS_L*kS_U

        # MPIW captured -> Calculate mean prediction interval given that y lies
        #                  in interval [y_L, y_U]
        # NOTE: When the network is totally wrong, i.e. kH = 0 everywhere, then 
        # MPIW_c gives nans, leading to an untrainable network. Solve this by
        # adding small val in case sum(kH) = 0
        MPIW_c = KB.sum((y_U-y_L)*kH)/(KB.sum(kH) + 0.001) # Add small val to penalize sum(kH) = 0
        # Calculate the coverage probability
        PICP_S = KB.mean(kS)

        n = KB.sum(KB.ones_like(y))

        # Loss_RHS = lambdaFactor_*n/(alpha_*(1-alpha_))*KB.square(KB.maximum(KB.zeros_like(y), ((1-alpha_) - PICP_S)))
        # Pearce et al. remove the alpha factors in their GitHub code: 
        # https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals/blob/master/code/DeepNetPI.py
        Loss_RHS = lambdaFactor_*KB.sqrt(n)*KB.square(KB.maximum(KB.constant(0), ((1-alpha_) - PICP_S)))

        Loss_S = MPIW_c + Loss_RHS

        return Loss_S

    return QD_loss



def deepNetPI(x_train, y_train, ensembles = 10, ensembleTrainingRatio = 0.8, numLayers = 3, numNeurons = 80, activation = 'relu', dropout = 0.1, CustomStructure = None, alpha = 0.05, softenFactor = 160, lambdaFactor = 1, epochs=50, batches=400):
    ''' Function to create ensemble of ANN models which directly approximate the prediction intervals following [1]
    
    :param x_train: Array of training input data. Shape = (N, 1, n) -> N = number observations, n = number inputs
    :param y_train: Array of training target data. Shape = (N, 1, 1)
    :param ensembles: Number of model instances, as int, to create for model regularization. Default = 10. 
    :param numLayers: Number of hidden layers as int. Ignored if ANN_Type = 'RNN' or if CustomStructure is not None.
    :param activation: Hidden layer activation function. Ignored if CustomStructure is not None.
    :param numNeurons: Number of neurons per layer as int (constant across layers). Ignored if CustomStructure is not None.
    :param dropout: Neuron dropout rate as float. Default = 0.1 
    :param CustomStructure: List of (list of) dictionaries describing the (FNN) network architecture. Each dictionary represents a layer of the network and is required to have the keys
                - 'Number Neurons' = Number of neurons in said layer,
                - 'Activation' = Activation function of the layer,
                - 'Dropout' = Dropout rate for the layer
            The number of dictionaries corresponds to the number of layers in the network. If a list of lists of dictionaries is passed (i.e. structure following: [[{}]]), then each inner list corresponds
            to a different neural network (i.e. model ensemble). Thus, a structure following [[{...}], [{...}], [{...}]] is a three network model. 
    :param alpha: QD loss function hyperparameter; dictates confidence level for the PIs. Default = 0.05 (95% confidence level)
    :param softenFactor: QD Loss function hyperparameter; dictates how closely the sigmoid resembles a step function. The higher the value, the more 'steep' the sigmoid. Default = 160 following [1]
    :param lambdaFactor: QD Loss function hyperparameter; scales magnitude of the invalid prediction interval penalty. Default = 1 following [1]
    :param epochs: Number of training epochs
    :param batches: Batch size for training
    :return: List of ANN models

    [1] Pearce, Tim, Alexandra Brintrup, Mohamed Zaki, and Andy Neely. 
        "High-quality prediction intervals for deep learning: A distribution-free, 
        ensembled approach." In International Conference on Machine Learning, 
        pp. 4075-4084. PMLR, 2018.        
    '''
    # Infer type of x and convert shape accordingly
    x_train, n = makeNdArray(x_train, reshape=True)
    y_train, _ = makeNdArray(y_train, reshape=True)

    # Check CustomStructure, if passed:
    if CustomStructure:
        depth = int(_check_depth(CustomStructure, 0))

        if depth == 1:
            # Broadcast customStructure
            CustomStructure = [CustomStructure]*ensembles
        elif depth == 2:
            # Check if number of elements matches number of ensembles
            if len(CustomStructure) != ensembles:
                raise ValueError('Number of model structures in CustomStructure does not match number of ensembles ({} != {})'.format(len(CustomStructure), ensembles))
        else:
            raise ValueError('Invalid CustomStructure depth. Expected depth between [1, 2], current depth is {}'.format(depth))

    models = []
    N_training = int(ensembleTrainingRatio * len(x_train))
    for i in range(ensembles):
        indices_training = np.sort(np.random.choice(len(x_train), N_training, replace = False))
        model = keras.Sequential()
        model.add(keras.Input(shape=(1, n)))
        # Build hidden layer(s)
        if CustomStructure:
            for lyr in CustomStructure[i]:
                lyr_neurons = lyr['Number Neurons']
                lyr_activation = lyr['Activation']
                lyr_dropout = lyr['Dropout']
                model = _build_FNN_Layers(model, 1, lyr_neurons, lyr_activation, lyr_dropout)
        else:
            model = _build_FNN_Layers(model, numLayers, numNeurons, activation, dropout)
        model.add(keras.layers.Dense(2, activation = None))  # None activation = Linear
        model.compile(optimizer='adam', loss=_loss_deepNetPI(alpha=alpha, softenFactor=softenFactor, lambdaFactor=lambdaFactor), metrics=['mae'])
        model.fit(x_train[indices_training], y_train[indices_training], epochs=epochs, batch_size = batches)
        models.append(model)

    return models



def _ensemble_avg_y(models, x, *args, **kwargs):
    '''Utility function to average model ensemble predictions
    
    :param models: List of ANN models (i.e. ensembles)
    :param x: Numpy array of input data for which predictions should be made. Shape = (N, 1, n) -> N = number observations, n = number inputs
    :param *args: Additional arguments (ignored by this function)
    :param **kwargs: Additional keyword arguments, passed on to tensorflow model.predict method
    :return: Averaged ensemble model prediction
    '''
    n_models = len(models)
    # Use first model to infer shape of y. 
    y_tot = models[0].predict(x, **kwargs)
    for i in range(n_models-1):
        y_tot += models[i+1].predict(x, **kwargs)

    return y_tot/n_models



def _bootstrapGetVarY(models, x):
    '''Utility function to get variance of model predictions across model ensembles
    
    :param models: List of ANN models
    :param x: Numpy array of input data for which predictions should be made. Shape = (N, 1, n) -> N = number observations, n = number inputs
    :return: var_y = Variance due to model uncertainty and misspecification, y_avg = Model prediction
    '''    
    B = len(models)
    y_i = np.zeros((len(x), B))
    # Get predictions of all B models
    for b in range(B):
        y_i[:, b] = models[b].predict(x).reshape(-1)

    y_avg = np.nanmean(y_i, axis = 1).reshape(-1, 1)
    var_y = 1/(B-1)*np.sum(np.square(y_i - y_avg), axis = 1).reshape(-1, 1)
    return var_y, y_avg



def _predictWithBootstrapPI(models, x, bootstrapNN):
    '''Utility function to make predictions using the bootstrap PI method
    
    :param models: List of ANN models (i.e. ensembles)
    :param x: Numpy array of input data for which predictions should be made. Shape = (N, 1, n) -> N = number observations, n = number inputs
    :param bootstrapNN: Bootstrap prediction interval ANN
    :return: y_hat = Model Prediction, var_i = Variance in predictions (used to construct prediction interval)
    '''    
    # Infer type of x and convert shape accordingly
    x, n = makeNdArray(x, reshape=True)
    # Check if input is standardized SysID model output. 
    try:
        models = models['Model']
    except TypeError:
        pass
    
    # Make predictions, and variance thereof, across ensembles
    var_y, y_hat = _bootstrapGetVarY(models, x)
    # approximate full PIs
    var_e = bootstrapNN.predict(x).reshape(-1, 1)
    var_i = var_y + var_e
    return y_hat.reshape(x.shape[0], -1), var_i.reshape(x.shape[0], -1)



def predict(TrainedModels, x, **kwargs):
    '''Function to make predictions of y based on x using trained ANN model(s)
    
    :param TrainedModels: Dictionary containing trained ANN model(s), output of <train>. SysID.Model format.  
    :param x: Numpy array of input data for which predictions should be made. Shape = (N, 1, n) -> N = number observations, n = number inputs
    :param **kwargs: Additional arguments to be passed to tensorflow model.predict method
    :return: y_hat = Array of (Combined ensemble) model predictions, sig_hat = Array of forecast error variance (is array of zeros if no prediction intervals are computed)
    '''    
    restrictedPredict = False
    # Check if x_vector is known
    if 'x_vec' in TrainedModels['Additional'].keys():
        x_vec = TrainedModels['Additional']['x_vec']
        if x_vec is not None:
            x = x[x_vec]
    # Check if x should be normalized
    isNormalized = TrainedModels['Additional']['_normalized']
    if isNormalized:
        x = Normalize(x.copy(), X_bar = TrainedModels['Additional']['normalizer']['x_bar'], X_std = TrainedModels['Additional']['normalizer']['x_std'])
    # Infer type of x and convert shape accordingly
    x, n = makeNdArray(x, reshape=True)
    # Check if input is standardized SysID model output. 
    try:
        models = TrainedModels['Model']
    except TypeError:
        models = TrainedModels
        restrictedPredict = True
    
    # Assuming TrainedModels is standard SysID model structure
    if not restrictedPredict:
        if TrainedModels['Additional']['modelOutput'] == 'interval':
            ensemble_pred_L = np.zeros((x.shape[0], len(models)))
            ensemble_pred_U = np.zeros((x.shape[0], len(models)))
            ensemble_pred = np.zeros((x.shape[0], len(models)))
            for i, mdl in enumerate(models):
                pred_i = mdl.predict(x).reshape(-1, 2)
                ensemble_pred_L[:, i] = pred_i[:, 0].reshape(-1)
                ensemble_pred_U[:, i] = pred_i[:, 1].reshape(-1)
                # take mean of lower and upper bound prediction as target prediction
                ensemble_pred[:, i] = np.nanmean(pred_i, axis = 1).reshape(-1)

            std_L = np.nanstd(ensemble_pred_L, axis = 1)
            std_U = np.nanstd(ensemble_pred_U, axis = 1)
            mean_L = np.nanmean(ensemble_pred_L, axis = 1)
            mean_U = np.nanmean(ensemble_pred_U, axis = 1)

            # Value of z_conf does not matter (as long as != 0), only need to find variance
            # associated with the interval, and not the interval itself
            z_conf = 1.96 # 95% confidence
            pred_PI_L = mean_L - z_conf*std_L
            pred_PI_U = mean_U + z_conf*std_U
            y_hat = np.nanmean(np.vstack((pred_PI_L.reshape(-1), pred_PI_U.reshape(-1))), axis = 0)
            sig_hat = (pred_PI_U - y_hat)/z_conf
            sig_hat = np.square(sig_hat)

            # Modify y_hat if Stacked is selected. 
            if 'Stacked' in TrainedModels['Additional'].keys():
                stacker = TrainedModels['Additional']['Stacked']
                y_hat = stacker.predict(ensemble_pred.reshape(-1, 1, len(models)), **kwargs)

        else:
            if 'Stacked' in TrainedModels['Additional'].keys():
                stacker = TrainedModels['Additional']['Stacked']
                # Build predictor ANN inputs
                ensemble_pred = np.zeros((x.shape[0], len(models)))
                for i, mdl in enumerate(models):
                    ensemble_pred[:, i] = mdl.predict(x, **kwargs).reshape(-1)
                y_hat = stacker.predict(ensemble_pred.reshape(-1, 1, len(models)), **kwargs)
            else:
                y_hat = _ensemble_avg_y(models, x, **kwargs)

            if 'BootstrapPINN' in TrainedModels['Additional'].keys():
                _, sig_hat = _predictWithBootstrapPI(models, x, TrainedModels['Additional']['BootstrapPINN']) 
            else:
                sig_hat = np.zeros(y_hat.shape)
    else:
        print('[ WARNING ] The input for <TrainedModel> in <predict> is not a SysID.Model.TrainedModel object. Predictions will therefore not have prediction intervals.')
        y_hat = _ensemble_avg_y(models, x, **kwargs)
        sig_hat = np.zeros(y_hat.shape)

    # Un-normalize output if input is normalized
    if isNormalized:
        y_hat = DeNormalize(y_hat.reshape(x.shape[0], -1), stats=[TrainedModels['Additional']['normalizer']['y_bar'], TrainedModels['Additional']['normalizer']['y_std']])
        sig_hat = sig_hat * np.array(TrainedModels['Additional']['normalizer']['y_std'])**2

    return y_hat.reshape(x.shape[0], -1), sig_hat



def evaluate(TrainedModels, x, y, showPlots = True, method_inputSensitivity = 'perturbation', inputSensitivityKwargs = {}, weightSensitivityKwargs = {}):
    '''Function to evaluate input and weight sensitivity of trained ANN model(s)
    
    :param TrainedModels: Dictionary containing trained ANN model(s), output of <train>. SysID.Model format.
    :param x: Input data upon which evaluation is conducted. Array of shape = (N, 1, n) -> N = number observations, n = number inputs
    :param y: Not currently used by the ANN model. Target data upon which evaluation is conducted. Array of hape = (N, 1, m) -> m = number of outputs. 
    :param showPlots: Indicates if evaluation should be plotted or not. Options are:
            - True = Draw and show plots,
            - False = Do not draw or show plots,
            - 'plot' = Only draw plots (plots can be shown later by calling matplotlib.pyplot.show()),
            Default = True
    :param method_inputSensitivity: Method used to calculate the input sensitivities. Options are:
            - 'perturbation' -> Applies the perturbation method. Each input is individually scaled by some (user-specifiable) factor and the corresponding change in output is observed || 
            - 'correlation' -> Applies the correlation method. All inputs are randomly perturbed simultaneously over a number of (user-specifiable) runs. Changes in inputs are correlated with changes in output through pearson correlations. ||
            Default = 'perturbation' 
    :param inputSensitivityKwargs: Dictionary of keyword arguments to pass to the chosen input sensitivity method. See specific input sensitivity functions for possible kwargs. If empty, default values are taken. 
    :param weightSensitivityKwargs: Dictionary of keyword arguments to pass to the weight sensitivity method. If empty, default values are taken. 
    :return: Dictionary containing sensitivity results. 
    '''
    # Check if method_inputSensitivity is known
    knownMethods = {'perturbation':{'function':_inputSensitivity_perturbation, 
                                    'fargs':{'perturbationScale':0.95}
                                    },
                    'correlation':{'function':_inputSensitivity_correlation,
                                    'fargs':{'perturbationMagnitude':0.05,
                                             'numberRuns':50}
                                  }
                    }
    if method_inputSensitivity is None:
        print('[ WARNING ] method_inputSensitivity is None, as such input sensitivity is being skipped!')
        inputSensitivity = None
    else:
        if not method_inputSensitivity.lower() in knownMethods.keys():
            raise ValueError('Unknown method_inputSensitivity: "{}". Expected one of {}'.format(method_inputSensitivity, list(knownMethods.keys())))

        # Extract function-specific kwargs from inputSensitivityKwargs, use defaults if not present
        kwargsIn = knownMethods[method_inputSensitivity]['fargs']
        for k in inputSensitivityKwargs.keys():
            if k not in kwargsIn.keys():
                raise ValueError('Unknown Keyword Argument "{}" in inputSensitivityKwargs'.format(k))
            else:
                kwargsIn.update({k:inputSensitivityKwargs[k]})

        # Input sensitivity
        print('\n[ INFO ] Evaluating input sensitivity of neural network(s) through {} method'.format(method_inputSensitivity))
        inputSensitivity = knownMethods[method_inputSensitivity]['function'](TrainedModels, x, **kwargsIn)
    
    evaluation = {'Input sensitivity':inputSensitivity}

    # Weight sensitivity
    print('\n[ INFO ] Evaluating weight sensitivity of neural network(s)')
    if len(TrainedModels['Additional']['weightHistory']):
        # Extract function-specific kwargs from weightSensitivityKwargs, use defaults if not present
        kwargsWs = {'threshold':0.01, 'inputWeightHorizonFactor':0.1, 'epochWindow':10}
        for k in weightSensitivityKwargs.keys():
            if k not in kwargsWs.keys():
                raise ValueError('Unknown Keyword Argument "{}" in weightSensitivityKwargs'.format(k))
            else:
                kwargsWs.update({k:weightSensitivityKwargs[k]})
        weightSensitivity = _getWeightSensitivity(TrainedModels, **kwargsWs)
    else:
        print('[ WARNING ] Expected to have weight history from training, but found an empty list. Please set <storeTrainingWeights = True> in <Model.train()>')
        print('[ WARNING ] Skipping weight sensitivity')
        weightSensitivity = None

    evaluation.update({'Weight sensitivity':weightSensitivity})

    if showPlots is not False:
        # If x is a pandas DataFrame, infer variable names from columns
        try:
            x_vec = x.columns
        except AttributeError:
            x_vec = None
        
        figs = _plotSensitivity(x_vec, {'method':method_inputSensitivity, 'Input sensitivity':evaluation['Input sensitivity']},
                                {'Show traces':True, 'Sensitivity':weightSensitivity}, showPlots=showPlots, returnFigs=False)

    return evaluation



def _inputSensitivity_perturbation(TrainedModels, x, perturbationScale):
    '''Utility function to compute the input sensitivities using the perturbation method
    
    :param TrainedModels: SysID.Model.TrainedModel object; Dictionary containing tensorflow models and assets. 
    :param x: Input data upon which evaluation is conducted. Array of shape = (N, 1, n) -> N = number observations, n = number inputs
    :param perturbationScale: Factor to which the inputs, x, should be scaled by as a float. For example, perturbationScale = 1.1 increases the input by 10%, while perturbationScale 0.9 decreases it by 10%. 
    :return: 2-D array of shape (M,N) summarizing the change in outputs for each model in M, and input in N. 
    '''    
    models = TrainedModels['Model']
    # Infer type of x and convert shapes accordingly
    x, n = makeNdArray(x, reshape=True)
    N = x.shape[0]
    
    # Pre-allocate error array. 
    diff_map = np.zeros((len(models), n))
    # For each of the inputs
    for i in np.arange(n):
        print('[ INFO ] Perturbing input [{}/{}]'.format(i+1, n))
        x_i = x.copy()
        # Set the input to zero and see influence on each of the models
        x_i[:, 0, i] = x_i[:, 0, i] * perturbationScale
        # Pre-allocate error array
        diffs = np.zeros(len(models))
        for m, mdl in enumerate(models):
            print('[ INFO ] Testing input {} with model {} [{}/{}]'.format(i+1, m+1, m+1, len(models)))
            # Make predictions with original inputs
            pred_x = mdl.predict(x)
            # Make predictions with input x_i = 0
            pred_xi = mdl.predict(x_i)
            # Observe influence of setting x_i = 0
            try:
                diff_mdl = np.sqrt(np.sum(np.square((pred_x - pred_xi)))/N)
            except TypeError:
                import code
                code.interact(local=locals())
            diffs[m] = diff_mdl
        diff_map[:, i] = diffs
    return diff_map



def _inputSensitivity_correlation(TrainedModels, x, numberRuns, perturbationMagnitude):
    '''Utility function to compute the input sensitivities using the correlation method
    
    :param TrainedModels: SysID.Model.TrainedModel object; Dictionary containing tensorflow models and assets. 
    :param x: Input data upon which evaluation is conducted. Array of shape = (N, 1, n) -> N = number observations, n = number inputs
    :param numberRuns: Number of runs to run correlations (i.e. samples for correlation). Due to the inherent stochasticity of the method, more runs means better reliability of results. 
    :param perturbationMagnitude: Magnitude of change, as a percentage of the associated input, to be applied to that input. For example, perturbationMagnitude = 0.1 applies between a -10% to +10% perturbation to the input. Perturbations are sampled from a normal distribution N(0, 1) * perturbationMagnitude
    :return: 3-D matrix of shape (M,D,N) where M is the number of models, D the number of observations (i.e. length of x) and N the number of inputs. Entries in this matrix contain the correlations between the change in inputs and outputs as a result of the perturbations. 
    '''    
    # Extract number of observations, N, and number of inputs n
    N, n = x.shape
    # Pre-allocate arrays 
    dy = np.zeros((len(TrainedModels['Model']), numberRuns, N, 1))
    dx = np.zeros((1, numberRuns, N, n))
    # Randomly pertrub inputs and observe changes in outputs for a specified number of runs (i.e. samples)
    for r in range(numberRuns):
        print('[ INFO ] Evaluating changes in inputs and outputs [run {}/{}]'.format(r+1, numberRuns))
        # Apply perturbations to x and observe change in y
        dy_r, dx_r = __evaluatePerturbation(TrainedModels, x, perturbMag=perturbationMagnitude)
        # Store results
        dy[:, r, :, :] = dy_r.reshape(len(TrainedModels['Model']), N, 1)
        dx[:, r, :, :] = dx_r.reshape(1, N, n)
    # Pre-allocate model correlation arrays 
    mdl_corrs = np.zeros((len(TrainedModels['Model']), N, n))
    for i in range(len(TrainedModels['Model'])):
        dy_m = dy[i]
        dx_m = dx[0]
        corrs = np.zeros((N, n))
        # Apply pearson correlations of sampled dy and dx, element-wise along N
        for v in range(n):
            corr = pearsonCorr(dy_m.reshape(numberRuns, -1), dx_m[:, :, v].reshape(numberRuns, -1))
            corrs[:, v] = corr
        mdl_corrs[i, :, :] = corrs
    return mdl_corrs



def __evaluatePerturbation(TrainedModels, x_test, perturbMag = 0.01):
    '''Utility function to apply random perturbations to the inputs of an ANN and observe the changes in outputs

    :param TrainedModels: SysID.Model.TrainedModel object; Dictionary containing tensorflow models and assets. 
    :param x: Input data upon which evaluation is conducted. Array of shape = (N, 1, n) -> N = number observations, n = number inputs
    :param perturbMag: Magnitude of change, as a percentage of the associated input, to be applied to that input. For example, perturbationMagnitude = 0.1 applies between a -10% to +10% perturbation to the input. Perturbations are sampled from a normal distribution N(0, 1) * perturbationMagnitude
    Outputs: pred_diffs = Change in outputs, perturbMat = Change in inputs
    '''    
    # Extract model ensembles
    mdls = TrainedModels['Model']
    # Reshape x_test, if needed 
    x_test, n = makeNdArray(x_test, reshape=True)
    N = x_test.shape[0]
    # Get reference performance
    x_clean = x_test.copy()
    # pred_clean, pred_var_clean = model.predict(x_clean)
    clean_preds = np.zeros((len(mdls), N))
    for i, mdl in enumerate(mdls):
        pred_i = mdl.predict(x_clean)
        # Here we take the mean columnwise for ANNs which predict the 
        # prediction intervals directly (so, mean is y_pred). For ANNs
        # which predict y_pred directly, pred_i is unchanged. 
        clean_preds[i, :] = np.nanmean(pred_i, axis=1).reshape(-1)
    # Add perturbations to the input
    # Generate perturbation matrix by drawing random samples from a uniform distribution [0, 1]
    # and scale perturbations to desired magnitude
    perturbMat = np.random.normal(size=(N, n))*perturbMag
    # Scale noise to associated input, so the noise is proportional to input
    scaling = np.max(x_test, axis=0) - np.min(x_test, axis=0)
    perturbMat = perturbMat*scaling
    # Add perturbation and get performance
    x_in = x_test.copy() + perturbMat.reshape(-1, 1, n)
    perturb_preds = np.zeros((len(mdls), N))
    pred_diffs = np.zeros((len(mdls), N))
    for i, mdl in enumerate(mdls):
        pred_i = mdl.predict(x_in)
        perturb_preds[i, :] = np.nanmean(pred_i, axis=1).reshape(-1)
        pred_diff_i = clean_preds[i, :] - perturb_preds[i, :]
        pred_diffs[i, :] = pred_diff_i
    return pred_diffs, perturbMat



def _getWeightSensitivity(TrainedModels, threshold, inputWeightHorizonFactor, epochWindow):
    '''Utility function to calculate the weight sensitivities of the ANN ensembles
    
    :param TrainedModels: SysID.Model.TrainedModel object; Dictionary containing tensorflow models and assets. 
    :param threshold: The threshold for convergence of the models as a ratio. For example, threshold = 0.01 indicates that convergence is reached when the change in weights is less than 1% of its average value. 
    :param inputWeightHorizonFactor: Ratio to the total number of epochs to be used for computing the final input weights. The number of epochs is then propagated backwards from the convergence/last epoch, to compute the input sensitivity as derived from the weight magnitudes. 
    :param epochWindow: Number of epochs used as a window to derive the weight sensitivity statistics of individual link weights
    :return: Dictionary of weight sensitivity results
    '''

    '''
    Extra notes:
        Weight sensitivity dictionary entries. 
            - weightSensitivity = Dictionary containing the weight sensitivity metrics with entries
            - W = Weights from training epochs, list of arrays of shape (L,M,E,N1,N2) where 
                L = number of layers, M = number of models, E = number of epochs, 
                N1 and N2 are the dimensions of the weights inside the layer. 
            - dW = change in weights from training epochs, list of arrays of shape (L,M,(E-1),N1,N2)
            - converged = Boolean indicating if the training has converged under the specified threshold
            - convergence epoch = Epoch of convergence, if converged
            - Input weights = Final weights of each of the ensemble ANNs averaged over the inputWeightHorizonFactor
            - Average horizon = Number of epochs used to compute input weights
            - metrics = Dictionary with entries
                * average = (Layer-normalized) average weight value computed over successive windowed epochs of
                    size epochWindow. Shape is (L, B, M, N1, N2) where L is the number of layers, B is the number 
                    of bins (i.e. int(numEpochs/epochWindow)), M the number of models, N1 and N2 are the shapes
                    of the weights in a given layer.
                * variance = (Layer-normalized) variance in weight value computer over successive windowed epochs
                    of size epochWindow. Shape is (L, B, M, N1, N2)
                * model weight change = the mean of entry <variance> for a given layer. Shape = (L, B, M). Gives an
                    indication of the degree of change within a given layer and bin
                * model weight spread = the variance of entry <mean> for a given layer. Shape = (L, B, M). Gives an
                    indication of the degree of spread in weight values within a given layer and bin
    '''

    # Derive shapes of network
    Ws = TrainedModels['Additional']['weightHistory']
    Weights = []
    dWeights = []
    numLayers = len(Ws[0][0])
    numEpochs = len(Ws[0])
    numModels = len(Ws)

    # Set default values
    hasConvergence = {'idx':0, 'layer':0, 'model':0}
    hasConverged = True
    convergenceEpoch = None

    # Extract and re-structure weight history such that it is easier to manipulate for computations
    for lyr in range(numLayers):
        # Layers stored as list since these can be variable depending on ANN structure and cells
        Wl = []
        dWl = []
        for m in range(numModels):
            Wm = []
            for i in range(numEpochs):
                Wm.append(Ws[m][i][lyr])
            Wm = np.array(Wm)
            dWm = Wm[1:] - Wm[:-1]
            Wl.append(Wm)
            dWl.append(dWm)
            # Find where weights approximately do not change any more. Take the change in weights and scale by the weight value, 
            # then observe where this signal is below the convergence threshold
            idxConvergence = np.where(np.nanmean(np.abs(dWm.reshape(numEpochs - 1, -1)), axis = 1)/np.nanmean(np.abs(Wm.reshape(numEpochs, -1)[:numEpochs-1]), axis=1) <= threshold)[0] + 1
            if not len(idxConvergence):
                print('[ WARNING ] Model weights have not converged with threshold = {}'.format(threshold))
                print('[ WARNING ] Convergence test failed for model {} layer {} ({} </= {})'.format(m+1, lyr+1, np.nanmin(np.nanmean(np.abs(dWm.reshape(numEpochs - 1, -1)), axis = 1)/np.nanmean(np.abs(Wm.reshape(numEpochs, -1)[:numEpochs-1]), axis=1)) ,threshold))
                hasConverged = False
            else:
                # Store which model and layer is the last to converge
                if idxConvergence[0] > hasConvergence['idx']:
                    hasConvergence['idx'] = idxConvergence[0]
                    hasConvergence['model'] = m
                    hasConvergence['layer'] = lyr

        Wl = np.array(Wl)
        dWl = np.array(dWl)
        Weights.append(Wl)
        dWeights.append(dWl)

    if hasConverged:
        print('[ INFO ] Weights appear to have converged with threshold = {}'.format(threshold))
        print('[ INFO ] Last model to converge is model {} for layer {}'.format(hasConvergence['model'] + 1, hasConvergence['layer'] + 1))
        print('[ INFO ] Convergence @ epoch = {}'.format(hasConvergence['idx']))
        convergenceEpoch = hasConvergence['idx']

    # Calculate average weights stemming from the inputs -> gives an indication of the 'importance' of a given input. Note, however, that this
    # importance may be diluted/amplified as the signal progresses through the nextwork. Nonetheless, high input weights mean stronger signals 
    # propagated through the network. 
    
    # infer the number of inputs
    n = Weights[0].shape[2]
    inputWeights = np.zeros((numModels, n))
    averageHorizon = int(inputWeightHorizonFactor*numEpochs)
    # Take last averageHorizon epochs before convergence, if available, otherwise just the last averageHorizon epochs
    if hasConverged:
        if averageHorizon > convergenceEpoch:
            lb = 0
        else:
            lb = convergenceEpoch - averageHorizon
        ub = convergenceEpoch
    else:
        lb = -1*averageHorizon
        ub = None
    for m in range(numModels):
        Wm = Weights[0][m]
        inputWeights[m, :] = np.nanmean(np.nanmean(np.abs(Wm[lb:ub, :, :]), axis = 0), axis = 1)



    # Calculate individual weight mean and variance across epochs with sample size of epochWindow
    print('[ INFO ] Evaluating individual weight statistics over epochs, with sample size {} epochs'.format(epochWindow))
    averageMetrics = []
    varianceMetrics = []
    overallChangeW = []
    overallSpreadW = []
    for lyr in range(numLayers):
        lyrAvgMetrics, lyrVarMetrics = __getWeightChangeOverWindowedEpochs(Weights, lyr, epochWindow)
        averageMetrics.append(lyrAvgMetrics)
        varianceMetrics.append(lyrVarMetrics)
        lyrChangeW = np.nanmean(lyrVarMetrics.reshape(lyrAvgMetrics.shape[0], numModels, -1), axis = 1)
        lyrSpreadW = np.nanvar(lyrAvgMetrics.reshape(lyrAvgMetrics.shape[0], numModels, -1), axis = 1)
        # lyrChangeW = np.nanmean(lyrVarMetrics.reshape(numModels, -1), axis = 1)
        # lyrSpreadW = np.nanvar(lyrAvgMetrics.reshape(numModels, -1), axis = 1)
        overallChangeW.append(lyrChangeW)
        overallSpreadW.append(lyrSpreadW)

    metrics = {'average':averageMetrics, 'variance':varianceMetrics, 'model weight change':overallChangeW, 'model weight spread':overallSpreadW}

    weightSensitivity = {'W':Weights, 'dW':dWeights, 
                        'converged':hasConverged, 'convergence epoch':convergenceEpoch, 
                        'Input weights':inputWeights, 'Average horizon':averageHorizon, 
                        'metrics':metrics}
    return weightSensitivity



def __getWeightMetrics_range(Ws, layer, lb, ub):
    '''Utility function to compute the (layer-normalized) average and variance of weights within a given range (i.e. epoch window)
    
    :param Ws: Weights from training epochs, list of arrays of shape (L,M,E,N1,N2) where L = number of layers, M = number of models, E = number of epochs, N1 and N2 are the dimensions of the weights inside the layer. 
    :param layer: Layer for which metrics should be obtained
    :param lb: Lowerbound of the range (i.e. epoch window)
    :param ub: Upperbound of the range (i.e. epoch window)
    :return avgW: (Layer-normalized) average weight values computed over the specified range. Shape is (M, N1, N2) where M the number of models, N1 and N2 are the shapes of the weights in a given layer.
    :return varW: (Layer-normalized) variance in weight value computer over over the specified range of size epochWindow. Shape is (M, N1, N2)
    '''
    numModels = Ws[layer].shape[0]
    # Extract layer-specific structure
    numIn = Ws[layer].shape[2]
    if len(Ws[layer].shape) > 3:
        numOut = Ws[layer].shape[3]
    else:
        numOut = 1
    avgW = np.zeros((numModels, numIn, numOut))
    varW = np.zeros((numModels, numIn, numOut))
    for m in range(numModels):
        Wm = Ws[layer][m]
        # Get average weight and variance therein for each link, wij, within interval [lb, ub[ epochs. 
        avgW[m] = np.nanmean(Wm[lb:ub], axis = 0).reshape(numIn, numOut)
        varW[m] = np.nanvar(Wm[lb:ub], axis = 0).reshape(numIn, numOut)
    # Normalize average (link) weights and variances with respect to the average weight within the entire layer
    avgW = avgW/np.nanmean(np.abs(avgW), axis = 0).reshape(1, numIn, numOut)
    varW = varW/np.nanmean(np.abs(avgW), axis = 0).reshape(1, numIn, numOut)
    return avgW, varW



def __getWeightChangeOverWindowedEpochs(Ws, layer, epochWindow):  
    '''Utility function to compute the (layer-normalized) average and variance of weights over successive epoch windows

    :param Ws: Weights from training epochs, list of arrays of shape (L,M,E,N1,N2) where L = number of layers, M = number of models, E = number of epochs, N1 and N2 are the dimensions of the weights inside the layer. 
    :param layer: Layer for which metrics should be obtained
    :param epochWindow: Number of epochs to be used for calculating the weight sensitivity metrics as int
    :return averageValues: (Layer-normalized) average weight value computed over successive windowed epochs of size epochWindow. Shape is (B, M, N1, N2) where B is the number of bins (i.e. int(numEpochs/epochWindow)), M the number of models, N1 and N2 are the shapes of the weights in a given layer.
    :return varianceValues: (Layer-normalized) variance in weight value computer over successive windowed epochs of size epochWindow. Shape is (B, M, N1, N2)
    '''
    # Infer model structure
    if len(Ws[layer].shape) > 3:
        numModels, numEpochs, numIn, numOut = Ws[layer].shape
    else:
        numModels, numEpochs, numIn = Ws[layer].shape
        numOut = 1
    if epochWindow >= numEpochs:
        raise ValueError('Specified epochWindow <{}> is greater than number of epochs <{}>'.format(epochWindow, numEpochs))
    numEpochBins = int(numEpochs/epochWindow)
    varianceValues = np.zeros((numEpochBins, numModels, numIn, numOut))
    averageValues = np.zeros((numEpochBins, numModels, numIn, numOut))
    lowerBound = 0
    upperBound = epochWindow
    for b in range(numEpochBins):
        # Get normalized average weight and variance therein for each link, wij, within interval [lowerBound, upperBound[ epochs. 
        averageValues[b, :, :, :], varianceValues[b, :, :, :] = __getWeightMetrics_range(Ws, layer, lowerBound, upperBound)
        # Increment counters
        lowerBound = upperBound
        upperBound += epochWindow
    return averageValues, varianceValues



def _plotSensitivity(x_vec, inputSensitivityVars, weightSensitivityVars, showPlots = True, returnFigs = True):
    '''Utility function to plot the results of the input and weight sensitivity evaluation

    :param x_vec: list of strings corresponding the to labels of the input variables 
    :param inputSensitivityVars: Dictionary containing the input sensitivity results, as obtained from the <evaluate()> method
    :param weightSensitivityVars: Dictionary containing the weight sensitivity results, as obtained from the <evaluate()> method
    :param showPlots: Indicates if evaluation should be plotted or not. Options are:
            - True = Draw and show plots,
            - False = Do not draw or show plots,
            - 'plot' = Only draw plots (plots can be shown later by calling matplotlib.pyplot.show()),
            Default = True
    :param returnFigs: Boolean dictating if figures should be returned, Default = True
    :return: Generated pyplot figure objects, if returnFigs is True. Otherwise returns None. 
    '''
    from itertools import cycle
    import matplotlib.patches as mpatches

    figs = {}

    # Unpack input sensitivity variables 
    inFigs = []
    method_inputSensitivity = inputSensitivityVars['method']
    inputSensitivity = inputSensitivityVars['Input sensitivity']
    if method_inputSensitivity is None:
        figs.update({'Input Sensitivity':None})
    else:
        if method_inputSensitivity.lower() == 'correlation':
            toPlot = np.nanmean(np.abs(inputSensitivity), axis = 1)
        else:
            toPlot = inputSensitivity
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(toPlot)
        ax.set_xlabel(r'$\mathbf{Input}$', fontsize=16)
        ax.set_ylabel(r'$\mathbf{Ensemble \quad Model}$', fontsize=16)
        ax.set_yticks(np.arange(len(inputSensitivity)))
        ax.set_yticklabels(np.arange(1, len(inputSensitivity)+1, 1))
        if x_vec is not None:
            ax.set_xticks(np.arange(len(x_vec)))
            ax.set_xticklabels(x_vec)
        # ax.tick_params(which='both', direction='in')
        cbar = fig.colorbar(cax, ticks=[np.min(toPlot), np.max(toPlot)], orientation = 'horizontal')
        cbar.ax.set_xticklabels([r'$\mathbf{Minimal \quad influence}$', r'$\mathbf{Maximal \quad influence}$'])
        
        inFigs.append(fig)

        figs.update({'Input sensitivity':inFigs})


    # Check if weight sensitivity has been run
    weightSensitivity = weightSensitivityVars['Sensitivity']
    wsFigs = []
    if weightSensitivity is not None:
        # Unpack weight sensitivity variables
        showTraces = weightSensitivityVars['Show traces']
        Weights = weightSensitivity['W']
        dWeights = weightSensitivity['dW']
        numLayers = len(Weights)
        numModels, numEpochs, numIn, _ = Weights[0].shape
        hasConverged = weightSensitivity['converged']
        convergenceEpoch = weightSensitivity['convergence epoch']
        inputWeights = weightSensitivity['Input weights']
        averageHorizon = weightSensitivity['Average horizon']
        for lyr in range(numLayers):
            colors = cycle(('orangered', 'darkorange', 'gold', 'mediumseagreen', 'lightseagreen', 'cornflowerblue'))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('ANN Weight Sensitivity: Layer {}'.format(lyr + 1), fontsize = 16)
            for m in range(numModels):
                if showTraces:
                    ax.plot(np.abs(dWeights[lyr][m]).reshape(numEpochs-1, -1), zorder=m, c='silver', alpha = 0.1)
                ax.plot(np.nanmean(np.abs(dWeights[lyr][m]).reshape(numEpochs-1, -1), axis=-1), c = next(colors), zorder = m + numModels, label = 'Model {}'.format(m+1))
            
            handles, labels = ax.get_legend_handles_labels()
            if hasConverged:
                ax.axvspan(convergenceEpoch, numEpochs, color = 'mediumorchid', alpha = 0.2)
                handles.append(mpatches.Patch(color='mediumorchid', label='Over-fitting', alpha = 0.2))
                labels.append('Over-fitting')

            ax.legend(handles = handles, labels=labels, loc='upper right')
            ax.set_xlabel(r'$\mathbf{Epochs} \quad [-]$', fontsize = 16)
            ax.set_ylabel(r'$\mathbf{Average \quad change \quad in \quad weights} \quad [-]$', fontsize = 16)
            ax.tick_params(which='both', direction='in', labelsize=14)

            wsFigs.append(fig)

        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Averaged input layer weights for last {} epochs before convergence'.format(averageHorizon), fontsize = 16)
        cax = ax.imshow(inputWeights, cmap='RdPu')
        ax.set_xlabel(r'$\mathbf{Input}$', fontsize = 16)
        ax.set_ylabel(r'$\mathbf{Ensemble \quad model}$', fontsize = 16)
        ax.tick_params(which='both', direction='in', labelsize=14)
        ax.set_yticks(np.arange(numModels))
        ax.set_yticklabels(np.arange(1, numModels+1, 1))
        if x_vec is not None:
            ax.set_xticks(np.arange(len(x_vec)))
            ax.set_xticklabels(x_vec)
        
        cbar = fig.colorbar(cax, orientation = 'horizontal')
        cbar.set_label(r'$\mathbf{Average \quad weight}$', fontsize=16)

        wsFigs.append(fig)
        
    figs.update({'Weight sensitivity':wsFigs})

    if showPlots is True:
        plt.show()  

    if returnFigs:
        return figs
    else:
        return None



def reduceModel(ANNModel, inputs, targets, threshold = 0.05, averageHorizon = 10):
    '''Function to 'reduce model' over-fitting by averaging network weights over a specified number of epochs
    
    :param ANNModel: SysID.Model object of the trained neural network model. 
    :param inputs: Unused by ANN reduceModel
    :param targets: Unused by ANN reduceModel
    :param threshold: Convergence threshold as float. A network has converged when their weight variance is less than this threshold times the average weights.
    :param averageHorizon: Number of epochs over which to average weights, going backwards from convergence epoch (if no convergence epoch, then final averageHorizon epochs is taken instead)
    :return: None. Modifies weights of ANNModel inplace. 
    '''
    # Derive shapes of network
    Ws = ANNModel.TrainedModel['Additional']['weightHistory']
    numLayers = len(Ws[0][0])
    numEpochs = len(Ws[0])
    numModels = len(Ws)

    # Set default values
    hasConvergence = {'idx':0, 'layer':0, 'model':0}

    # Extract and re-structure weight history such that it is easier to manipulate for computations
    for lyr in range(numLayers):
        # Layers stored as list since these can be variable depending on ANN structure and cells
        Wl = []
        dWl = []
        for m in range(numModels):
            Wm = []
            for i in range(numEpochs):
                Wm.append(Ws[m][i][lyr])
            Wm = np.array(Wm)
            dWm = Wm[1:] - Wm[:-1]
            Wl.append(Wm)
            dWl.append(dWm)
            # Find where weights approximately do not change any more. Take the change in weights and scale by the weight value, 
            # then observe where this signal is below the convergence threshold
            idxConvergence = np.where(np.nanmean(np.abs(dWm.reshape(numEpochs - 1, -1)), axis = 1)/np.nanmean(np.abs(Wm.reshape(numEpochs, -1)[:numEpochs-1]), axis=1) <= threshold)[0] + 1
            if len(idxConvergence):
                # Store which model and layer is the last to converge
                if idxConvergence[0] > hasConvergence['idx']:
                    hasConvergence['idx'] = idxConvergence[0]
                    hasConvergence['model'] = m
                    hasConvergence['layer'] = lyr
    
    # print('Imma bootstrap something here to check something')
    # hasConvergence['idx'] = numEpochs - averageHorizon - 1

    if hasConvergence['idx'] != 0:
        for m in range(numModels):
            Ws_m = Ws[m]
            model = ANNModel.TrainedModel['Model'][m]
            modelWeights = []
            for lyr in range(numLayers):
                if hasConvergence['idx'] + averageHorizon >= numEpochs:
                    averageHorizon = numEpochs - hasConvergence['idx'] - 1
                Ws_m_lyr = Ws_m[:, lyr]
                averagedLayerWeights = Ws_m_lyr[hasConvergence['idx']]
                for epoch in range(averageHorizon):
                    averagedLayerWeights += Ws_m_lyr[hasConvergence['idx'] + epoch + 1]
                    
                averagedLayerWeights = averagedLayerWeights*(1/averageHorizon)
                modelWeights.append(averagedLayerWeights)
            model.set_weights(modelWeights)
        return None
    else:
        print('[ INFO ] Could not reduce model with given threshold.')
        return None



def dropFromEnsemble(Model, columns):
    ''' Function to remove models from ensemble based on index in ensemble. NOTE: This currently does not work with bootstrapped PIs as the bootstrap PI ANN assumes a certain number of ANN models specified upon initialization. 
    
    :param Model: SysID.Model object of the trained neural network model
    :param columns: List of indices corresponding to ANN models to be dropped from ensemble 
    :return: None. Modifies SysID.Model object inplace. 
    '''
    # Does not work with bootstrap PIs, need to retrain bootstrapped PI with new ensembles
    if Model.TrainedModel['Additional']['modelOutput'] != 'interval':
        raise ValueError('Dropping ensemble models is incompatible with Bootstrap PIs.')
    else:
        # Extract information pertaining to the models
        ensemble = Model.TrainedModel['Model']
        weightHistory = Model.TrainedModel['Additional']['weightHistory']
        mask = list(np.arange(0, len(ensemble)))
        for c in columns:
            toDrop = np.where(c == np.array(mask))[0]
            ensemble.pop(toDrop[0])
            weightHistory = np.delete(weightHistory, toDrop, axis = 0)
            mask.pop(toDrop[0])
        # Adjust model
        Model.TrainedModel['Model'] = ensemble
        Model.TrainedModel['Additional']['weightHistory'] = weightHistory
        Model.Currentmodel = Model.TrainedModel
    return None



def summary(Status, Models):
    '''Function to print a summary of the model to the terminal

    :param Status: State of the model, inherited from SysID class. 
    :param Models: Models corresponding to the state, inherited from SysID class
    :return: None. 
    '''
    if Status.lower() in ('compiled', 'trained'):
        print('{:{fill}^65}'.format('ANN Model Structures', fill='-'))
        for i, mdl in enumerate(Models['Model']):
            print('{:{fill}^65}'.format('ANN {}'.format(i), fill='='))
            mdl.summary()
            print('-'*65)
    else:
        print('{:^65}'.format('Model not compiled!'))

    return None



def save(path, model):
    '''Function to save trained ANN model ensembles, assumes SysID.Model object format 

    :param path: Save directory
    :param model: TrainedModel to save
    :return: None
    '''
    # Extract list of models and (any) additional information
    models = model['Model']
    additional = model['Additional']

    # Create sub-directory to save ensemble models to
    saveDir = os.path.join(path, 'ANN')
    if not os.path.isdir(saveDir):
        os.mkdir(saveDir)

    # Save ANN models using the default method of TensorFlow. 
    # Each model and its associated assets are stored in a 
    # subfolder corresponding to their model number (from 0)
    for i, mdl in enumerate(models):
        mdl.save(os.path.join(saveDir, '{}'.format(i)))

    # Create sub-directory to store additional information to
    addDir = os.path.join(path, 'add')
    if not os.path.isdir(addDir):
        os.mkdir(addDir)
    # Check for other ANNs in additional and save them using
    # the tensorflow method
    if 'Stacked' in additional.keys():
        additional['Stacked'].save(os.path.join(addDir, 'Stacked'))
        additional.pop('Stacked')
    if 'BootstrapPINN' in additional.keys():
        additional['BootstrapPINN'].save(os.path.join(addDir, 'BootstrapPINN'))
        additional.pop('BootstrapPINN')
    # Save the rest of the information in additional, which should 
    # not raise any pickle errors
    with open(os.path.join(addDir, 'add.pkl'), 'wb') as f:
        pkl.dump(additional, f)
        f.close()

    return None



def load(path):
    '''Function to load ANN model ensembles from directory, uses SysID.Model object format

    :param path: Model directory
    :return: TrainedModel in SysID.Model format
    '''

    loadedModel = {}

    # Load additional information for model (e.g. use of bootstrap PI ANN)
    addDir = os.path.join(path, 'add')
    with open(os.path.join(addDir, 'add.pkl'), 'rb') as f:
        additional = pkl.load(f)
        f.close()

    # Load subfolders in the additional directory, if present
    subFolders = [fol for fol in os.listdir(addDir) if os.path.isdir(os.path.join(addDir, fol))]
    for fol in subFolders:
        # Special case for boostrap PI ANNs since they require a custom loss function, which must be compiled explicitly
        if fol == 'BootstrapPINN':
            folMdl = tf.keras.models.load_model(os.path.join(addDir, fol), compile=False)
            folMdl.compile(optimizer='adam', loss=_bootstrapLossCBS, metrics=['mae'])
        # Otherwise they subfolders are likely tf models using default functions and can be compiled automatically
        else:
            folMdl = tf.keras.models.load_model(os.path.join(addDir, fol))
        additional.update({fol:folMdl})

    # Rebuild 'Additional' key of the SysID.Model object
    loadedModel.update({'Additional':additional})

    # Load ANN predictor models
    modelDir = os.path.join(path, 'ANN')
    models = []
    # Check if PIs are computed directly, which must be loaded differently due to use of custom loss function
    if additional['modelOutput'].lower() == 'interval':
        for i in os.listdir(modelDir):
            # mdl = tf.keras.models.load_model(os.path.join(modelDir, i), custom_objects={'QD_loss':_loss_deepNetPI()})
            mdl = tf.keras.models.load_model(os.path.join(modelDir, i), compile=False)
            # Do dummy compile
            # TODO: Can make it such that the optimizer, loss, etc. are consistent with <compile>
            mdl.compile(optimizer='adam', loss=_loss_deepNetPI(), metrics=['mae'])
            models.append(mdl)
    # Otherwise tf models use default tf functions 
    else:
        for i in os.listdir(modelDir):
            mdl = tf.keras.models.load_model(os.path.join(modelDir, i))
            models.append(mdl)

    # Rebuild 'Model' key of SysID.Model object
    loadedModel.update({'Model':models})

    return loadedModel



def copy(model):
    '''Function to make a copy of ANN trained model
    
    :param model: SysID.Model.TrainedModel object of the trained model
    :return: Copy of model 
    '''
    # Copy ANN ensemble
    modelCopy = {}
    modelCopy['Model'] = [i for i in model['Model']]
    additional = {}
    for k, v in model['Additional'].items():
        additional.update({k:v})
    modelCopy.update({'Additional':additional})
    return modelCopy
