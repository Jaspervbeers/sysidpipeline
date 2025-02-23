'''
Script to create a standalone barebones polynomial model file from a trained SysID polynomial model, without the local dependencies of the SysID class.
Resultant models are much smaller in size, since they only contain information necessary for making predictions. However, including functionality for
calculating prediction intervals will increase filesize proportional to training data size, due to required knowledge on observed training data. 

Written by: Jasper van Beers
Contact: j.j.vanbeers@tudelft.nl
Date: 24-02-2022
'''
# ================================================================================================================================ #
# Imports
# ================================================================================================================================ #
import os
import dill as pickle
from numpy import matrix, ones, hstack, dot, add, subtract, divide, multiply, power, nan, array, where, arange, isnan
from re import sub as reSub
from io import StringIO
from tokenize import generate_tokens
import SysID

# ================================================================================================================================ #
# Classes
# ================================================================================================================================ #
'''
Class bundles SysID polynomial model into an independent object that can be used for predictions without the need for local modules.
Predictions are made using the .predict(x) method, where x holds the same structure as the data used as model inputs for training. 
'''
# TODO: Add ReadMe which documents what is expected of x, or reference to other document/file/script which outlines this. 
class PolynomialModel:
    
    def __init__(self):
        # Import necessary functions
        self.npMatrix = matrix
        self.npOnes = ones
        self.npHstack = hstack
        self.npDot = dot
        self.npAdd = add
        self.npSubtract = subtract
        self.npDivide = divide
        self.npMultiply = multiply
        self.npPower = power
        self.npNan = nan
        self.npWhere = where
        self.npArray = array
        self.npArange = arange
        self.npIsnan = isnan
        self.StringIO = StringIO
        self.generate_tokens = generate_tokens
        self.isExtracted = False
        self._usePI = False
        self.coefficients = None
        self.polynomial = None
        return None


    def extractModel(self, sysIDModel, predictionIntervals = False, forceExtraction = False):
        if not self.isExtracted or forceExtraction: 
            self.coefficients = self._getCoefficients(sysIDModel)
            self.polynomial = self._getPolynomial(sysIDModel)
            if predictionIntervals:
                self._usePI = True
                self._inv_XtX = sysIDModel.TrainedModel['Model']['_inv(XtX)']
                self._s2 = sysIDModel.TrainedModel['Model']['_sigma2']
            self.makeRegressors()
        else:
            raise AttributeError('A polynomial model has already been extracted. Set forceExtraction = True extract anyway (will overwrite existing polynomial).')
        return None


    def predict(self, x):
        A = self._BuildRegressorMatrix(x, hasBias = ('bias' in self.polynomial))
        if self._usePI:
            pred = A*self.coefficients
            var = pred.copy()
            AT = A.T
            for i in range(len(var)):
                var[i] = self._s2 + self._s2*self.npDot(self.npDot(A[i, :], self._inv_XtX), AT[:, i])
            return A*self.coefficients, var
        else:
            return A*self.coefficients

    
    def _getCoefficients(self, sysIDModel):
        return sysIDModel.TrainedModel['Model']['Parameters']


    def _getPolynomial(self, sysIDModel):
        return sysIDModel.TrainedModel['Model']['Regressors']
    

    def _BuildRegressorMatrix(self, data, hasBias = True):
        # Pre-allocate matrix
        N = len(self.regressors)
        regMat = self.npMatrix(self.npOnes((data.shape[0], N)))
        # Fill in matrix using regressors and data
        for i, reg in enumerate(self.regressors):
            regMat[:, i] = self.npMatrix(reg.resolve(data)).T
        # Pre-pend the bias vector, if present
        if hasBias:
            biasVec = self.npMatrix(self.npOnes((data.shape[0], 1)))
            regMat = self.npHstack((biasVec, regMat))
        return regMat


    def makeRegressors(self):
        parsing = self.Parser()
        self.regressors = []
        for p in self.polynomial:
            if p != 'bias':
                p_RPN = parsing.parse(p)
                reg = self.Regressor(p_RPN)
                self.regressors.append(reg)

    # Class to convert string equations to postfix form (i.e. Reverse Polish Notation - RPN) which can be easily interpretted from left to right.
    class Parser:
        # Initialize the RPN (output) stack and operator stack (which handles order of operations prior to addition in the output stack)
        def __init__(self):
            self.sub = reSub
            self.operatorStack = []
            self.outputStack = []
            # Define allowable operators, along with their precedence and associativity
            self.operatorInfo = {
                '^':{'precedence':4,
                        'associativity':'R'},
                '*':{'precedence':3,
                        'associativity':'L'},
                '/':{'precedence':3,
                        'associativity':'L'},
                '+':{'precedence':2,
                        'associativity':'L'},
                '-':{'precedence':2,
                        'associativity':'L'}                   
            }

        # Main parsing function, which converts an input string equation into RPN form.
        def parse(self, inputString):
            self.refresh()
            self.tokens = self.tokenize(inputString)
            RPN = self.shuntYard(self.tokens)
            if len(RPN) == 0:
                return [inputString]
            else:
                return RPN

        # Empty (any) previously parsed information, and reset for parsing new strings
        def refresh(self):
            self.operatorStack = []
            self.outputStack = []

        # Convert input string into tokens, sliced by the operators. 
        def tokenize(self, inputString):
            # remove spaces in string
            cleanString = self.sub(r'\s+', "", inputString)
            # Convert to list of characters to isolate operators and brackets
            chars = list(cleanString)
            # Tokens
            tokens = []
            token = ""
            while len(chars) != 0:
                char = chars.pop(0)
                if char in self.operatorInfo.keys() or char in ['(', ')']:
                    if token != "":
                        tokens.append(token)
                    tokens.append(char)
                    token = ""
                else:
                    token += char
                if len(chars) == 0 and token != "":
                    tokens.append(token)
            return tokens

        # Apply the Shunting-yard algorithm to convert the tokens into RPN form. 
        def shuntYard(self, tokens):
            while len(tokens) != 0:
                token = tokens.pop(0)
                # Check if token is a known operator
                if token in self.operatorInfo.keys():
                    # Check operator priority
                    if not len(self.operatorStack) == 0:
                        sorting = True
                        while sorting:
                            push = False
                            # Check top of operator stack for brackets
                            if self.operatorStack[-1] not in ["(", ")"]:
                                if self.operatorInfo[self.operatorStack[-1]]['precedence'] > self.operatorInfo[token]['precedence']:
                                    # top operator has greater priority
                                    push = True
                                elif self.operatorInfo[self.operatorStack[-1]]['precedence'] == self.operatorInfo[token]['precedence']:
                                    if self.operatorInfo[self.operatorStack[-1]]['associativity'] == 'L':
                                        push = True
                            sorting = push and self.operatorStack[-1] != '('
                            if sorting:
                                self.outputStack.append(self.operatorStack.pop())
                            if len(self.operatorStack) == 0:
                                sorting = False
                    self.operatorStack.append(token)
                elif token == "(":
                    self.operatorStack.append(token)
                elif token == ")":
                    #Add operations to stack while in brackets
                    while True:
                        if len(self.operatorStack) == 0:
                            break
                        if self.operatorStack[-1] == "(":
                            break
                        self.outputStack.append(self.operatorStack.pop())
                    if len(self.operatorStack) != 0 and self.operatorStack[-1] == "(":
                        self.operatorStack.pop()
                else:
                    self.outputStack.append(token)
            self.outputStack.extend(self.operatorStack[::-1])
            return self.outputStack

    # Class which handles regressor evaluations. The regressor structure is stored upon initialization for efficiency. 
    class Regressor:
        def __init__(self, regressorRPN):
            self.RPN = regressorRPN
            self.numberIndices = [i for i, v in enumerate(regressorRPN) if self.isFloat(v)]
            self.knownOperators = {'+':add, '-':subtract, '/':divide, '*':multiply, '^':power}
            self.operatorIndices = [i for i, v in enumerate(regressorRPN) if v in self.knownOperators.keys()]
            self.invVariableIndices = self.numberIndices + self.operatorIndices
            self.npArange = arange
            self.npArray = array
            if len(self.invVariableIndices):
                self.variableIndices = [i for i in self.npArange(0, len(regressorRPN)) if i not in self.invVariableIndices]
            else:
                self.variableIndices = self.npArange(0, len(regressorRPN))

        def resolve(self, Data):
            # First convert RPN string into purely numbers
            RPN = self.RPN.copy()
            RPNStr = self.RPN.copy()
            for idx in self.variableIndices:
                RPN[idx] = Data[self.RPN[idx]]
            # Evaluate RPN expression
            stack = []
            if len(RPN) > 1:
                while len(RPN) > 0:
                    token = RPN.pop(0)
                    tokenStr = RPNStr.pop(0)
                    if tokenStr not in self.knownOperators.keys():
                        stack.append(token)
                    else:
                        b = self.npArray(stack.pop(), dtype=float)
                        a = self.npArray(stack.pop(), dtype=float)
                        stack.append(self.knownOperators[token](a, b))
                if len(stack) != 1:
                    raise ValueError('There are unaccounted variables in the RPN regressor stack. Please check regressor operations are parsed correctly.')
                else:
                    return stack[0]
            else:
                return self.npArray(RPN[0], dtype=float)

        def isFloat(self, string):
            try:
                float(string)
                return True
            except ValueError:
                return False

    # Extra class to convert use instead of pandas dataframe for faster (repeated) computations (i.e. useful for simulations)
    # TODO: Add functionality to convert existing dataframes. 
    class fasterDataFrame:
        def __init__(self, numRows, columns, npZeros):
            self.npZeros = npZeros
            self.shape = (numRows, len(columns))
            self.dfvalues = npZeros(self.shape)
            self.dfmapping = {k:v for v, k in enumerate(columns)}
            self.columns = columns

        def __getitem__(self, key):
            # Check if key or list is passed
            try:
                out = self.dfvalues[:, self.dfmapping[key]]
            except TypeError:
                out = self.npZeros((self.shape[0], len(key)))
                for i, k in enumerate(key):
                    out[:, i] = self.dfvalues[:, self.dfmapping[k]]
            return out

        def __setitem__(self, key, newvalue):
            try:
                self.dfvalues[:, self.dfmapping[key]] = newvalue
            except TypeError:
                for i, k in enumerate(key):
                    self.dfvalues[:, self.dfmapping[k]] = newvalue[:, i]


# ================================================================================================================================ #
# Usage
# ================================================================================================================================ #
# Load polynomail model
loadPath = os.path.join(os.getcwd(), 'exampleData', 'examplePolyModel')
loadedPolySysIDModel = SysID.Model.load(loadPath)

# Create and save standalone model
savePath = os.path.join(os.getcwd(), 'exampleData', 'standalonePolyModel')
if not os.path.exists(savePath):
    os.makedirs(savePath)

polyModel = PolynomialModel()
# Extract necessary information from saved model
polyModel.extractModel(loadedPolySysIDModel)
with open(os.path.join(savePath, 'standalonePolyModel.pkl'), 'wb') as f:
    pickle.dump(polyModel, f)

# End