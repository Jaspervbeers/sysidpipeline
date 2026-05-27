'''
Main system identification module. Imports the various techniques contained in the Techniques folder. 

Written by Jasper van Beers
Created: 12-01-2021
Last edit: 23-02-2022
'''

# ================================================================================================================================ #
# Global Imports
# ================================================================================================================================ #
from numpy import square, sqrt
from numpy import sum as npsum 
import pickle as pkl
import os
import importlib
import json
import pandas as pd

# ================================================================================================================================ #
# Local Imports
# ================================================================================================================================ #
from pathlib import Path
import sys
LIB_PATH = Path(__file__).resolve().parent.resolve()
sys.path.insert(0, str(LIB_PATH))
try:
    from Utility import Utility
except ImportError:
    from .Utility import Utility


# ================================================================================================================================ #
# Classes
# ================================================================================================================================ #
class Model:

    def __init__(self, Technique):
        '''__init__ function which imports functions from the specified system identification methods in Techniques and initializes model states.

        :param Technique: System identification method to use for modelling. String.
        :return: None. Raises Value error if specified Technique is undocumented in the Model class.  
        '''
        # Define known techniques
        self._techniques = ['stepwise_regression', 'ann', 'stepwise_ann_hybrid']
        # Check that specified technique is defined in self._techniques
        if Technique.lower() in self._techniques:
            self.technique = Technique.lower()
            # Import chosen identification library
            self._techniqueModule = importlib.import_module('Techniques.{}'.format(self.technique))
            self.UtilityFuncs = self._techniqueModule
            self.ModelState = 'Initialized'
            self.ModelStateHistory = [self.ModelState]
            self.CompiledModel = None
            self.TrainedModel = None
            self.CurrentModel = None
            self.x_train = None
            self.y_train = None
            self._save_as_pkl = False
        else:
            raise ValueError('[ ERROR ] Technique: "{}" not recognized. \nExpected: {}'.format(Technique, self._techniques))



    def compile(self, *args, **kwargs):
        '''Decorator for self.technique.compile. Prepares models for training using technique specific compilation. 
        
        :param *args: Technique specific arguments, see relevant documentation
        :param *kwargs: Technique specific keyword arguments, see relevant documentation
        :return: None, but updates self.ModelState to 'Compiled'. Access compiled model through self.CompiledModel
        '''
        # Check if model is initialized
        if 'Initialized' in self.ModelStateHistory:
            # Extract appropriate compiler for specified technique
            compiler = self._techniqueModule.compile
            # Check that all necessary positional arguments are provided. 
            posArgs, _ = Utility.getArgs(compiler, hasSelf=False)
            if len(posArgs) != len(args):
                for arg in posArgs:
                    if arg not in kwargs.keys():
                        # Output missing positional argument
                        raise ValueError('[ ERROR ] Necessary argument "{}" not found in function arguments.'.format(arg))
            # Compile model using technique specific compile function
            self.CompiledModel = compiler(*args, **kwargs)
            # Update model
            self.CurrentModel = self.CompiledModel
            self.ModelState = 'Compiled'
            self.ModelStateHistory.append(self.ModelState)
        else:
            raise ValueError('[ ERROR ] Please initialize model before compiling')
        return None



    def train(self, x, y, *args, **kwargs):
        '''Decorator for self.technique.train. Trains models.
        
        :param x: Pandas DataFrame (columns = variables, rows = samples) of input training data.
        :param y: Pandas DataFrame (columns = targets, rows = samples) of target training data. 
        :param *args: Technique specific arguments, see relevant documentation
        :param *kwargs: Technique specific keyword arguments, see relevant documentation
        :return: None, but updates self.ModelState to 'Trained'. Access compiled model through self.TrainedModel
        '''        
        # Check if model is compiled
        if 'Compiled' in self.ModelStateHistory:
            # Extract appropriate identification algorithm of technique
            identificationAlgorithm = self._techniqueModule.train
            # Train model
            TrainedModel = identificationAlgorithm(self.CompiledModel, x, y, *args, **kwargs)
            self.TrainedModel = TrainedModel
            self.x_train = x
            self.y_train = y
            # Update model
            self.ModelState = 'Trained'
            self.CurrentModel = TrainedModel
            self.ModelStateHistory.append(self.ModelState)
        else:
            self.TrainedModel = None
            raise ValueError('[ ERROR ] Please compile model before training')
        # return self.TrainedModel
        return None



    def predict(self, x, **kwargs):
        '''Decorator for self.technique.predict. Make predictions using model. 
        
        :param x: Pandas DataFrame (columns = variables, rows = samples) of inputs on which to make predictions.
        :param *kwargs: Technique specific keyword arguments, see relevant documentation
        :return: Tuple of (prediction, prediction variance). prediction variance may be used to compute prediction intervals
        '''                
        # Check if model has been trained
        if 'Trained' in self.ModelStateHistory:
            # Use appropriate prediction method of chosen technique
            Prediction = self._techniqueModule.predict(self.TrainedModel, x, **kwargs)
        else:
            raise ValueError('[ ERROR ] Please train model before making predictions')
        return Prediction

    

    def evaluate(self, *args, **kwargs):
        '''Decorator for self.technique.evaluate. Evaluates the sensitivities of the model.
        
        :param *args: Technique specific arguments, see relevant documentation
        :param *kwargs: Technique specific keyword arguments, see relevant documentation
        :return: Dictionary of evaluation result, see relevant documentation
        '''          
        # Check if model has been trained

        if 'Trained' in self.ModelStateHistory:
            # Use appropriate evaluation method of chosen technique
            evaluation = self._techniqueModule.evaluate(self.TrainedModel, self.x_train, self.y_train, *args, **kwargs)
        else:
            raise ValueError('[ ERROR ] Please train model before evaluating')            
        return evaluation




    def reduceModel(self, x = None, y = None, inplace = False, **kwargs):
        '''Decorator for self.technique.reduce, if available. Attempts to mitigate model over-fitting. 
        
        :param x: Pandas DataFrame (columns = variables, rows = samples) of inputs needed to evaluate model for over-fitting.
        :param y: Pandas DataFrame (columns = variables, rows = samples) of targets needed to evaluate model for over-fitting.
        :param inplace: Boolean. If True reduced model will replace current model. 
        :param *kwargs: Technique specific keyword arguments, see relevant documentation. 
        :return: None if inplace is True, reduced model otherwise as a SysID.Model object. 
        '''     
        # Check if model has been trained
        if self.ModelState == 'Trained':
            if x is None and y is None:
                x = self.x_train
                y = self.y_train
            # Use appropriate reduction method of chosen technique
            reducedModel = self._techniqueModule.reduceModel(self, x, y, **kwargs)
            if inplace:
                # Replace existing trained and current models with reduced model 
                self.TrainedModel.update({'Model':reducedModel})
                self.CurrentModel.update({'Model':reducedModel})
                return None
            else:
                # Create new Model object for reduced model 
                newModel = Model(self.technique)
                newModel.ModelState = self.ModelState
                newModel.ModelStateHistory = self.ModelStateHistory
                newModel.TrainedModel = {}
                newModel.TrainedModel.update({'Model':reducedModel})
                newModel.TrainedModel.update({'Additional':self.TrainedModel['Additional']})
                newModel.CurrentModel = newModel.TrainedModel
                newModel.x_train = self.x_train
                newModel.y_train = self.y_train
                return newModel
        else:
            raise ValueError('[ ERROR ] Cannot reduce an un-trained model.')



    def summary(self, state='current'):
        '''Decorator for self.technique.summary. Displays summary of model in terminal.  
        
        :param state: String of the model state to see summary of. Default is 'current' but 'trained' or 'compiled' are also possible. 
        :return: None.
        '''         
        states = {'current':(self.ModelState, self.CurrentModel),
                  'compiled':('Compiled', self.CompiledModel),
                  'trained':('Trained', self.TrainedModel)}
        print('\n')
        print('#'*65)
        print('{:^65}'.format('Model Summary'))
        print('#'*65)
        print('{:<25} {:>39}'.format('Technique:', self.technique))
        print('{:<25} {:>39}'.format('Model State:', self.ModelState))
        print('_'*65)
        # Call appropriate summary method of chosen technique to print technique specific information
        self._techniqueModule.summary(*states[state.lower()])
        print('#'*65)
        print('\n')
        return None


    @staticmethod
    def _RMSE(y, yhat):
        '''Static function to compute root mean square error
        
        :param y: Numpy array of true targets
        :param y_hat: Model predictions of targets
        :return: Root-mean squared error 
        ''' 
        N = max(y.shape)
        e2 = npsum(square((y - yhat.reshape(y.shape))))
        return sqrt(e2/N)



    def save(self, path, saveTrainingData = False):
        '''Decorator for self.technique.save. Saves model. 
        
        :param path: Save directory.
        :return: None. 
        '''         
        # Package model MetaData
        metadata = {'technique':self.technique,
                    'state':self.ModelState,
                    'state history':self.ModelStateHistory,
                    'has training data':saveTrainingData}
        # Build subpaths
        ModelPath = os.path.join(path, 'model')
        MetaDataPath = os.path.join(path, 'metadata')
        # Check if subpaths exists, otherwise create
        if not os.path.isdir(ModelPath):
            os.mkdir(ModelPath)
        if not os.path.isdir(MetaDataPath):
            os.mkdir(MetaDataPath)
        # Save metadata
        #   Prioritize saving as json, if possible 
        with open(os.path.join(MetaDataPath, 'mdlinfo.json'), 'w') as f:
            json.dump(metadata, f, indent = 4)
        # Pass model instance to internal save function of self.technique
        self._techniqueModule.save(ModelPath, self.CurrentModel)
        # Save training data, if true
        if saveTrainingData:
            if self._save_as_pkl:
                with open(os.path.join(MetaDataPath, 'trainingData.pkl'), 'wb') as f:
                    pkl.dump({'x':self.x_train, 'y':self.y_train}, f)
            else:
                self.__pd_to_dict()
                with open(os.path.join(MetaDataPath, 'trainingData.json'), 'w') as f:
                    json.dump(self.packed, f, indent=4)
        return None



    @staticmethod
    def load(path):
        '''Static function to load SysID.Model objects saved using the Model.save() protocol. Modelling technique is inferred. 
        
        :param path: Save directory.
        :return: loaded SysID.Model object. 
        '''         
        # Build subpaths 
        ModelPath = os.path.join(path, 'model')
        MetaDataPath = os.path.join(path, 'metadata')
        # Retrieve model metadata
        try:
            # Expect newest version of metadata file type
            with open(os.path.join(MetaDataPath, 'mdlinfo.json'), 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            # Otherwise, open legacy version
            with open(os.path.join(MetaDataPath, 'mdlinfo.pkl'), 'rb') as f:
                metadata = pkl.load(f)

        # Initialize model
        mdl = Model(metadata['technique'])
        # Load model
        currentModel = mdl.UtilityFuncs.load(ModelPath)
        mdl.CurrentModel = currentModel
        # Reassign model state and history
        mdl.ModelState = metadata['state']
        mdl.ModelStateHistory = metadata['state history']
        if mdl.ModelState == 'Trained':
            mdl.TrainedModel = currentModel
        elif mdl.ModelState == 'Compiled':
            mdl.CompiledModel = currentModel
        # Add training data, if available
        if 'has training data' in metadata.keys():
            try:
                if metadata['has training data']:
                    if os.path.exists(os.path.join(MetaDataPath, 'trainingData.pkl')):
                        with open(os.path.join(MetaDataPath, 'trainingData.pkl'), 'rb') as f:
                            trainingData = pkl.load(f)
                        mdl.x_train = trainingData['x']
                        mdl.y_train = trainingData['y']
                    elif os.path.exists(os.path.join(MetaDataPath, 'trainingData.json')):
                        with open(os.path.join(MetaDataPath, 'trainingData.json'), 'r') as f:
                            packed = json.load(f)
                        unpacked = mdl.__dict_to_pd(packed)
                        mdl.x_train = unpacked['x']
                        mdl.y_train = unpacked['y']
            except Exception as e:
                print(f'[ WARNING ] Found trainingData is metadata but failed to load them.\nThe following error occurred:\n\t{e}')
        return mdl


    def copy(self):
        '''Decorator for self.technique.copy function, if available. Copies a trained model. 
        
        :return: SysID.Model object of the copied model.  
        '''        
        if self.ModelState == 'Trained':
            modelCopy = Model(self.technique)
            modelCopy.TrainedModel = self._techniqueModule.copy(self.TrainedModel)
            modelCopy.CurrentModel = modelCopy.TrainedModel
            modelCopy.CompiledModel = modelCopy.CompiledModel
            modelCopy.ModelState = self.ModelState
            modelCopy.ModelStateHistory = self.ModelStateHistory
            modelCopy.x_train = self.x_train.copy()
            modelCopy.y_train = self.y_train.copy()
        else:
            raise NotImplementedError('Can only copy TrainedModel')
        return modelCopy
    
    def __pd_to_dict(self):
        self.packed = {}
        for k, pd_data in {'x':self.x_train, 'y':self.y_train}.items():
            if isinstance(pd_data, pd.Series):
                pd_data = pd_data.to_frame(name = pd_data.name)
            self.packed.update({k:{
                'df':pd_data.to_dict(orient = 'split'),
                'dtypes': {c:str(dt) for c, dt in pd_data.dtypes.items()}
            }})

    def __dict_to_pd(self, packed):
        unpacked = {}
        for k, d_data in packed.items():
            df = pd.DataFrame(d_data['df']['data'], index = d_data['df']['index'], columns = d_data['df']['columns'])
            #TODO: Restore appropriate dtypes. In current use, all entries (aside from index) are floats. 
            unpacked.update({k:df})