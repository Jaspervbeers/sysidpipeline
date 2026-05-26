'''
Script with necessary functions to build, train and make predictions with a stepwise regression model
Backend for the SysID class 'Stepwise_Regression' option. 
  
Stepwise regression based on:
  [klein2006] V. Klein and E.A. Morelli. Aircraft System Identification: Theory and Practice. AIAA
  education series. American Institute of Aeronautics and Astronautics, 2006. ISBN:
  9781563478321.

Written by Jasper van Beers
Created: 12-01-2021
Last edit: 19-04-2021
'''

# ================================================================================================================================ #
# Global Imports
# ================================================================================================================================ #
import numpy as np
import pickle as pkl
import os
from itertools import permutations, combinations
try:
    from matplotlib import pyplot as plt
    from matplotlib import cm
    from itertools import cycle
    import matplotlib.colors as clrs
    import matplotlib.patches as mpatches
except ImportError:
    print('{}\n{:<11}{}'.format('[ WARNING ] Importing one of the plotting modules in </Techniques/stepwise_regression.py> failed.', 
                                ' ', 
                                'While not an issue for identification, plotting utilities may not work.'))


# ================================================================================================================================ #
# Local Imports
# ================================================================================================================================ #
try:
    from Utility import parser
except ImportError:
    from .Utility import parser


# ================================================================================================================================ #
# Functions
# ================================================================================================================================ #
def _OLS(A, z, hasBias = True):
    '''Function to determine ordinary least squares regression

    :param A: Regressor Matrix, as numpy.matrix with shape [N x M] where N = number observations, M = number regressors
    :param z: Measurements of target, array with shape [N x 1]
    :return: List of [params, pred]. params denotes the model parameters, array with shape [M x 1]. pred gives model predictions, array with shape [N x 1]
    '''
    # See page 113-115 of klein2006 
    # A needs to be a np.matrix class with shape [n,m]. 
    # n = number of observations, m = total number of regressors
    n, m = A.shape
    p = m - 1 

    if hasBias:
        # Check for regressors other than constant term in A
        if p > 0:
            # Scale targets
            z_centered = (z - np.nanmean(z))
            Szz = np.sum(np.square(z_centered))
            z_scaled = z_centered/np.sqrt(Szz)

            # Scale regressors
            Sjj = np.sum(np.square(A[:, 1:] - np.nanmean(A[:, 1:], 0)))
            A_scaled = (A[:, 1:] - np.nanmean(A[:, 1:], 0))/np.sqrt(Sjj)

            # Obtain (scaled) parameters through OLS
            params_scaled = (A_scaled.T*A_scaled).I*A_scaled.T*z_scaled

            # Transform parameters to original form (i.e. unscaled)
            _params = params_scaled * np.sqrt(Szz/Sjj).T
            bias_term = np.nanmean(z) - np.nanmean(A[:, 1:], 0)*_params
            params = np.concatenate((bias_term, _params))
            pred = A*params

        # Only constant term in A
        elif p == 0:
            params = (A.T*A).I*A.T*z
            pred = A*params
        
        # Invalid inputs
        else:
            params = None
            pred = None
    
    # If no bias term
    else:
        # Scale targets
        z_centered = (z - np.nanmean(z))
        Szz = np.sum(np.square(z_centered))
        z_scaled = z_centered/np.sqrt(Szz)

        # Scale regressors
        Sjj = np.sum(np.square(A - np.nanmean(A, 0)))
        A_scaled = (A - np.nanmean(A, 0))/np.sqrt(Sjj)

        # Obtain (scaled) parameters through OLS
        params_scaled = (A_scaled.T*A_scaled).I*A_scaled.T*z_scaled
        # Transform parameters to original form (i.e. unscaled)
        params = params_scaled * np.sqrt(Szz/Sjj).T
        pred = A*params

    return [params, pred]


def _OLS_FLEX(A, z, hasBias = True, fixed_mapping = {}):
    '''Wrapper for the _OLS function that accounts for regressors with fixed coefficients
    '''
    # fixed_mapping maps the index of the fixed regressor, in A, to its coefficient value
    if len(fixed_mapping):
        nr = A.shape[1]
        fixed_idxs = list(fixed_mapping.keys())
        params_fixed = np.array([v['coeff'] for v in fixed_mapping.values()]).reshape(-1, 1)
        flex_idxs = [i for i in np.arange(0, nr, 1) if i not in fixed_idxs]
        A_fixed = np.delete(A, flex_idxs, 1)
        A_flex = np.delete(A, fixed_idxs, 1)
        pred_fixed = A_fixed @ params_fixed
        z_adjusted = z - pred_fixed
        # Catch for if bias is fixed -> make hasBias = False. 
        fixed_regs = [v['regressor'] for v in fixed_mapping.values()]
        hasBias = not hasBias if 'bias' in fixed_regs else hasBias
        [params_flex, pred_flex] = _OLS(A_flex, z_adjusted, hasBias=hasBias)
        params = np.zeros((nr, 1))
        params[fixed_idxs] = params_fixed
        params[flex_idxs] = params_flex
        pred = pred_flex + pred_fixed
        return [np.matrix(params), pred]
    else:
        # No fixed_mapping
        return _OLS(A, z, hasBias=hasBias)



def _PSE(predictions, targets, p):
    '''(Internal) Function to determine the predict square error, PSE

    :param predictions: Model predictions, array with shape [N x 1] where N = number observations
    :param targets: Targets (Measurements), array with shape [N x 1]
    :param p: Number of regressor terms including bias term, as integer
    :return: Predict square error
    '''    
    N = np.max(targets.shape)
    error = targets - predictions
    s_max = np.sum(np.square(targets - np.nanmean(targets)))/N
    return error.T*error/N + s_max * p/N



def _CoeffOfDetermination_R2(predictions, targets):
    '''(Internal) Function to calculate the coefficient of determination (R squared)

    :param predictions: Model predictions, array with shape [N x 1] where N = number observations
    :param targets: Targets (Measurements), array with shape [N x 1]
    :return: Coefficient of Determination
    '''
    N = np.max(targets.shape)
    SSr = predictions.T*targets - N*np.nanmean(targets)**2
    SSt = targets.T*targets - N*np.nanmean(targets)**2
    return SSr/SSt



def _GenPowers(d, numX):
    '''Function to determine all possible terms in a polynomial of degree, d, excluding the bias term.
        For example: 
            If P(x1, x2) is a polynomial of degree 2 then the possible terms are: 
                x1, x2, x1^2, x2^2, and x1x2. 
            The power lists are correspondingly: 
                [1, 0], [0, 1], [2, 0], [0, 2] and [1, 1]
            where the integer at index i denotes the power of variable x_i
    _GenPowers produces such a list of powers for a given polynomial degree and number of independent variables

    :param d: Polynomial degree, as integer
    :param numX: Number of independent variables, as integer
    :return: List of valid powers
    '''    
    # We would like to use the permutations function of itertools to generate 
    # this list of powers. However, 'permutations' only selects each number from
    # an inputted array once. Therefore, to allow for regressors like x1 * x2, we
    # require the possibility of having repeat power permutations. In the example
    # of x1 * x2, this is requiring [1, 1] as a valid permutation. 

    # Make an empty array of zeros up to numX - 1 to enable powers of xp ^ d
    # This permutation is [0, ..., 0, n, 0, ..., 0], which requires numX - 1 zeros
    PermArray = np.zeros((numX-1))
    for p in range(d):
        # Determine the maximum number of times a given power can fit in d
        diff = d - p
        numExtraElements = int(float(d)/diff)
        numP = np.ones(numExtraElements) * diff
        # Add it to the permutation array
        PermArray = np.hstack((PermArray, numP))

    # Before obtaining the permutations, we can reduce the number of permutations
    # checked by first selecting only valid (here, unique) combinations of powers
    # For example:
    #   If the permutation array is [p1, p2] then the possible permutations are
    #       -> [p1, p2] and [p2, p1]. For the case that p1=p2, the resultant 
    #          permutations give the same power list. Hence, we only care about
    #          unique power lists. 
    Combs = combinations(PermArray, numX)
    uniqueCombs = []
    for comb in Combs:
        # Ensure that the sum of the permutations is valid (i.e. < degree) 
        # terms in the list unique
        if int(sum(comb)) <= int(d) and list(comb) not in uniqueCombs:
            uniqueCombs.append(list(comb))
    
    validPerms = []
    for comb in uniqueCombs:
        # For each valid combination of powers, we want all possible permutations
        # of these powers, so long as they are not repeated
        PermList = permutations(comb)
        for perm in PermList:
            if list(perm) not in validPerms:
                validPerms.append(list(perm))

    return validPerms



def _BuildCandidateRegressors(polys, data):
    '''Function to build the candidate polynomial regressors (excluding bias vector)

    :param polys: List of dictionaries describing the candidate polynomials. Each dictionary in the list corresponds to polynomials composed of the same variables and degree.
            Dictionaries must have the following fields (i.e. keys)

                - 'vars' = variables, as list, which consitute the polynomial. These must correspond to the column names in input X. e.g. vars:['x1', 'x2', 'x5']

                - 'degree' = Integer dictating degree of the polynomial

                - 'sets' = List of additional constants/variables which the polynomial is multiplied with. For just the polynomial itself, 'sets':[1] the resultant polynomial is multiplied by 1.
                        For more complex variable relations, for example y1*P(x1,...,xn), then sets = ['y1'] or sets = [1, 'y1'] if both the base and y1 polynomials are required.
                        It is also possible to specify combinations of variables (e.g. 'x1 + x2'), so long as the components are columns in <data>
                        Other combinations, such as sets = ['y1*y2^(y3 + y1)'] are also possible, given that the components are all valid columns in <data>

                Example of a possible dictionary: 
                    P = {'vars':['x1', 'x2'], 'degree':2, sets:[1, 'x3']}
                    = (1 + x3)*(x1^2 + x1 + x1x2 + x2 + x2^2)

    :param data: pandas DataFrame containing the relevant data for building the polynomials with shape [N x R] where N = number observations and R = number columns
    :return: Dictionary mapping of regressors to their values derived from  <data>. Each entry in Regressors corresponds to a single regressor. 
        e.g. Regressors = {'x1^2':[n1, n2, ..., nN], 'x1*x2':[h1, h2, ..., hN], ...}
    '''    
    N = data.shape[0]
    Regressors = {}
    for poly in polys:
        X = np.matrix(data[poly['vars']].to_numpy())
        # Extract the valid powers of a polynomial of degree d with n terms
        validPwrs = _GenPowers(poly['degree'], len(poly['vars']))
        Map = {}
        for pwrLst in validPwrs:
            # Build regressor ID strings
            reg = ''
            for i, v in enumerate(poly['vars']):
                # Record the powers of the regressor terms provided that its power 
                # is not 0, in which case do not add term to the regressor ID
                reg += '{}^({}) * '.format(v, pwrLst[i])*bool(pwrLst[i])
            # Clean up string and strip last '*' from reg and add brackets
            reg = '(' + reg[:-3] + ')'
            Map.update({reg:pwrLst})

        # Build base polynomial set (i.e. P(x)*1)
        BaseSet = {}
        for reg, p in Map.items():
            # Multiply out terms 
            Xp = np.multiply.reduce(np.power(X, p), 1).T
            BaseSet.update({reg:Xp})

        # For each 'set' evaluate the LHS of the polynomial
        # e.g. for (y1+y2)*P(x1,...,xn), (y1+y2) is the LHS
        for s in poly['sets']:
            # Check if the set is the base set (i.e. set = 1)
            if s != 1:
                # If not, evaluate the LHS
                fctr = np.matrix(parser.InputParser(s, data)).T
                # Record the set terms
                prefix = '({})*'.format(s)
            else:
                fctr = 1
                prefix = ''
            for reg in BaseSet.keys():
                # Prefix set terms to polynomial term
                sReg = prefix + reg
                # Multiply LHS with polynomial, elementwise
                sXp = np.multiply(fctr, BaseSet[reg])
                Regressors.update({sReg:sXp})

    return Regressors



def _BuildFixedRegressors(variables, data, addBiasVector = True):
    '''(Internal) Function to build the fixed regressors 

    :param variables: list of the fixed regressors, as strings. These should correspond to columns of data.
        For example, if x1 is a fixed regressor then -> variables = ['x1']
        Combinations of variables are also valid inputs. 
            e.g. x1*x2 + x3/x4 -> variables = ['x1*x2 + x3/x4']
        Multiple variables may be inputted 
            e.g. variables = ['x1', 'x2', '-1*x3', 'x1^(x2+x3)']
        If only the bias vector is desired, then pass variables = [] (note, addBiasVector should be True)
    :param data: Pandas DataFrame containing the data necessary for building the fixed regressors where columns correspond to input variables and rows samples 
    :param addBiasVector: Boolean indicating if a bias vector (i.e. [1,...,1]) should be added. By default addBiasVector is True.
    :return: Dictionary mapping of regressors to their values derived from data. Each entry in Regressors corresponds to a single regressor. 
        e.g. Regressors = {'x1^2':[n1, n2, ..., nN], 'x1*x2':[h1, h2, ..., hN], ...}
    '''
    N = data.shape[0]
    Regressors = {}
    
    # Create bias vector if selected
    if addBiasVector:
        Regressors.update({'bias':np.matrix(np.ones((N, 1)))})

    # Evaluate variables in the variable list
    for v in variables:
        x = parser.InputParser(v, data).to_numpy()
        Regressors.update({v:np.matrix(x).T})

    return Regressors



def BuildRegressors(variables, data, addBiasVector = True):
    '''General function to build regressors based on variables. 

    :param variables: list of the fixed regressors, as strings. These should correspond to columns of data.
        For example, if x1 is a fixed regressor then -> variables = ['x1']
        Combinations of variables are also valid inputs. 
            e.g. x1*x2 + x3/x4 -> variables = ['x1*x2 + x3/x4']
        Multiple variables may be inputted 
            e.g. variables = ['x1', 'x2', '-1*x3', 'x1^(x2+x3)']
        If only the bias vector is desired, then pass variables = [] (note, addBiasVector should be True)
    :param data: Pandas DataFrame containing the data necessary for building the fixed regressors where columns correspond to input variables and rows samples 
    :param addBiasVector: Boolean indicating if a bias vector (i.e. [1,...,1]) should be added. By default addBiasVector is True.
    :return: Dictionary mapping of regressors to their values derived from data. Each entry in Regressors corresponds to a single regressor. 
        e.g. Regressors = {'x1^2':[n1, n2, ..., nN], 'x1*x2':[h1, h2, ..., hN], ...}
    '''
    return _BuildFixedRegressors(variables, data, addBiasVector=addBiasVector)



def _BuildRegressorMatrix(regressors, data, hasBias = True):
    '''Function to build a regressor matrix from a set of regressors

    :param regressors: List of strings corresponding to the regressors IDs
    :param data: Pandas DataFrame of the identification data necessary to build the regressor matrix.
        The components of the terms in regressors should correspond to columns of data. 
        e.g. for regressors = ['x1 + x2', '5*x3'], terms 'x1', 'x2', and 'x3' should be
            columns in data. 
    :param hasBias: Boolean that indicates if the regressor list has a bias a bias vector. Default is True. 
    :return: Matrix containing the regressors (i.e. A in z = A*parameters) of shape [N x M] where N = number of observations and M = number regressors
    '''
    # Check if there is a bias vector
    if hasBias:
        regressors = regressors[1:]
    # Pre-allocate matrix
    N = len(regressors)
    # regMat = np.matrix(np.ones((max(data.shape), N)))
    regMat = np.matrix(np.ones((data.shape[0], N)))
    # Fill in matrix using regressors and data
    for i, reg in enumerate(regressors):
        regMat[:, i] = np.matrix(parser.InputParser(reg, data)).T
    
    # Pre-pend the bias vector, if present
    if hasBias:
        # biasVec = np.matrix(np.ones((max(data.shape), 1)))
        biasVec = np.matrix(np.ones((data.shape[0], 1)))
        regMat = np.hstack((biasVec, regMat))

    return regMat



def _RMSE(y, yhat):
    '''Function to calculate the root mean square error between two 1-D arrays

    :param y: Numpy array of true targets
    :param y_hat: Model predictions of targets
    :return: Root-mean squared error 
    '''    
    N = max(y.shape)
    e2 = np.sum(np.square((y - yhat.reshape(y.shape))))
    return np.sqrt(e2/N)



def compile(data, polys, fixedParams, includeBias = True, fixed_coefficients = {}):
    '''Function to build model regressors (both candidate and fixed)

    :param data: Pandas DataFrame containing inputs for the model where columns denote the input variables and rows the number of samples
    :param polys: List of the various polynomials from which the candidate regressors are based. See documentation for function <_BuildCandidateRegressors> for details on <polys>
    :param fixedParams: List of strings indicating the fixed regressors. See documentation for function <_BuildFixedRegressors> for details on <fixedParams>
    :param includeBias: Boolean indicating if a bias vector should be included. Default is True. 
    :return: Dictionary containing the necessary polynomial regressors for the model. 
        CompiledModel has entries:

            - 'Candidate Regressors' = The candidate regressors for the model

            - 'Fixed Regressors' = The fixed regressors of the model

            - 'Has Bias' = Boolean indicating if bias vector is included or not

            - 'Candidate Polynomials' = Input param polys
    '''    
    CandidateRegressors = _BuildCandidateRegressors(polys, data)
    FixedRegressors = _BuildFixedRegressors(fixedParams, data, addBiasVector=includeBias)

    # Check for duplicate fixed regressors that may appear in CandidateRegressors. There are edge cases, for example, when
    # using the fixed_coefficients where duplicates caught downstream in train(...) are no longer caught due to the offset
    # created by the fixed coefficient (e.g. let the current model be Fy = v + w) where both v and w are fixed but only w 
    # has a fixed coefficient. If 'v^1' appears in the candidate regressors, then we will see that we cannot perfectly explain
    # v with Fy due to the presence of w. Thus, the algorithm thinks v is a 'new' regressor when, in fact, it is a duplicate.
    CandidateArray = np.array(list(CandidateRegressors.values())).reshape(len(CandidateRegressors), -1).T
    CandidateKeys = list(CandidateRegressors.keys())
    dupes = []
    for reg in FixedRegressors.values():
        diff = CandidateArray - reg.__array__()
        i = np.where(np.isclose(diff.sum(axis=0), 0))[0]
        if len(i):
            for _i in i:
                _dupe = CandidateKeys[_i.astype(int)]
                if _dupe not in dupes:
                    dupes.append(_dupe)
                    # Remove dupe from CandidateRegressor pool
                    CandidateRegressors.pop(_dupe)

    # CompiledModel = {'Candidate Regressors':CandidateRegressors, 'Fixed Regressors':FixedRegressors, 'data':data, 'Has Bias':includeBias}
    CompiledModel = {'Candidate Regressors':CandidateRegressors, 'Fixed Regressors':FixedRegressors, 'Has Bias':includeBias, 'Candidate Polynomials':polys, "Fixed Coefficients":fixed_coefficients}

    return CompiledModel


# TODO: Move fixed coefficients to compile. Then can structure in correct way such that other functions can use it
def train(CompiledModel, TrainingInputs, TrainingTargets, stop_criteria = 'PSE', Fin = 4, Fout = 4, k_lim = 10, force_k_lim = False):
    '''Function to identify a model using stepwise regression 

    :param CompiledModel: Dictionary containing the necessary polynomial regressors for the model. Equivalent to the output of function <compile>. 
    :param TrainingInputs: Unused but remains for compatibility with SysID.Model object . 
    :param TrainingTargets: Pandas DataFrame of the target data used for model identification. Shape is [N x 1] where N = the number of observations (samples). 
    :param fixed_coefficients: Optional dictionary mapping fixed regressor to a coefficient value. This coefficient value will be used for the regressor in the model. Leave empty if coefficient is free.
    :param stop_criteria: Dictates what stopping criteria to use. Default is 'PSE' (Predict Square Error)
        :Options are: 
            'PSE' (Predict square error),
            'R2' (Coefficient of determination),
            'F0' (F-test),
            None (stops after iteration k_lim unless termination condition is met)
    :param Fin: F-test bound for adding a regressor to model structure, default = 4 (See Klein 2006)
    :param Fout: F-test bound for removing a regressor from model structure, default = 4 (See Klein 2006)
    :param k_lim: Upper bound for iterations if stop_criteria is not given (i.e. is None), default = 10. 
    :param force_k_lim: Boolean which enforces a strict upper bound of <param k_lim>, regardless of the chosen stopping condition. Default is False.
    :return: Dictionary, called out, containing the identified model and some additional information in the SysID.Model standard format. 
        - out['Model']: Dictionary containing the model elements. 
            > out['Model']['A']: Final regressor matrix
            > out['Model']['Parameters']: Polynomial coefficients
            > out['Model']['Regressors']: Regressor IDs corresponding to Mapping
            > out['Model']['Error variance']: Variance of modelling errors w.r.t training data
        - out['Additional']: Dictionary of any additional information
            > out['Additional']['Log']: Log indicating what occurred at each step 
    '''

    # Function to check if the regressor scheduled for addition passes the F-test
    # Inputs:
    #   A = Current regressor matrix
    #   IncomingRegressor = Regressor scheduled for addition to model
    #   z = Model targets (measurement data)
    #   Fin = Test statistic
    # Outputs:
    #   F0 = F statistic
    #   F0>Fin = Boolean indicating if the scheduled regressor significantly 
    #       improves model performance under Fin. 
    # def _checkFin(A, IncomingRegressor, z, Fin, hasBias = True):
    def _checkFin(A, IncomingRegressor, z, Fin, hasBias = True, coeff_mapping = {}):
        [na, pa] = A.shape
        
        # Add incoming regressor to A matrix
        Ax = np.matrix(np.hstack((A, IncomingRegressor)))
        # Perform OLS with new A matrix
        # [params0, pred0] = _OLS(Ax, z, hasBias)
        [params0, pred0] = _OLS_FLEX(Ax, z, hasBias = hasBias, fixed_mapping = coeff_mapping)

        SS0 = params0.T * Ax.T * z - na*np.nanmean(z)**2
        
        # # Perform OLS with old A matrix
        # [params1, pred1] = _OLS(A, z, hasBias)

        s2 = np.sum(np.square(z-pred0))/(na-pa-1)      

        # Determine F statistic between Ax and A
        # SS1 = params1.T*A.T*z - na*np.nanmean(z)**2  
        # F0 = (SS0 - SS1)/s2
        F0 = SS0/s2

        return [F0, F0>Fin]



    # Function to check if any of the current regressors should be removed from the model structure
    # Inputs:
    #   A = Current regressor matrix
    #   z = Model targets (measurement data)
    #   Fout = Test statistic
    #   pf = Number of fixed parameters
    # Outputs
    #   RemoveIdx = Column index in A of regressor to be removed. If no regressors should be removed
    #       then RemoveIdx = None
    #   F0 = F statistic
    def _CheckFout(A, z, Fout, pf, hasBias = True, coeff_mapping = {}):
        [na, pa] = A.shape

        # Obtain current model performance
        # [params0, pred0] = _OLS(A, z, hasBias = hasBias)
        [params0, pred0] = _OLS_FLEX(A, z, hasBias = hasBias, fixed_mapping = coeff_mapping)
        SS0 = params0.T * A.T * z - na*np.nanmean(z)**2
        s2 = np.sum(np.square(z - pred0))/(na-pa)

        # Obtain model performance for each regressor removed
        # Pre-allocate array to store F statistic for each of the subsequent regressor-removed models
        F0_lst = np.ones((pa,1)) * Fout
        for j in np.arange(pf,pa, 1):
            _A = np.delete(A, j, 1)
            # [params1, pred1] = _OLS(_A, z, hasBias = hasBias)
            [params1, pred1] = _OLS_FLEX(_A, z, hasBias = hasBias, fixed_mapping = coeff_mapping)
            SS1 = params1.T * _A.T * z - na*np.nanmean(z)**2

            F0_lst[j] = (SS0-SS1)/s2

        # Find index of minimum F0 provided that it is below Fout
        jj = np.where((F0_lst < Fout) & (F0_lst == min(F0_lst)))[0]

        # If there is a regressor for which the model performance 
        # does not change significantly when it is removed, then
        # flag this regressor, otherwise return None. 
        if jj.shape[0] != 0:
            RemoveIdx = jj[0]
        else:
            RemoveIdx = None

        return [RemoveIdx, F0]


    
    # Function to update values in CriterionDict. CriterionDict keeps track of the stopping criterion 
    # of the current and previous steps. 
    # Inputs:
    #   CriterionDict = Dictionary containing stoping criteria values for current and previous steps
    #   which = 'new' or 'old' which correspond to the current and previous steps respectively 
    #   valuesDict = Dictionary of the stoping criteria values to be added to CriterionDict
    # Outputs:
    #   CriterionDict = Modified CriterionDict with updated values from valuesDict
    def _UpdateCriteria(CriterionDict, which, valuesDict):
        for key in CriterionDict.keys():
            subDict = CriterionDict[key]
            subDict.update({which:valuesDict[key]})
            CriterionDict.update({key:subDict})
        return CriterionDict


    
    # Function to check PSE stopping criterion, if selected. 
    # Inputs:
    #   CriterionDict = Dictionary containing stoping criteria values for current and previous steps
    #   A = Current regressor matrix
    #   parameters = Current parameters
    #   CurrentRegressors = Dictionary of the regressor IDs and associated indices in A
    #   targets = Model targets (measurement data)
    #   PSEThreshold = Threshold of PSE for acceptable performance, regardless if improvements can be made
    # Outputs:
    #   Stop = Boolean indicating if algorithm should be stopped
    #   [A, parameters, CurrentRegressors] = Either equivalent to the variables in the current step or the
    #       previous step depending on if the stopping criteria is met or not. 
    def _checkPSE(CriterionDict, A, parameters, CurrentRegressors, targets, PSEThreshold, RemoveLog, hasBias = True, coeff_mapping = {}):
        Stop = False
        # Check if PSE has increased. If so, take [A, parameters, CurrentRegressors] of previous step
        # if CriterionDict['PSE']['new'] >= CriterionDict['PSE']['old']*0.99:
        if CriterionDict['PSE']['new'] >= CriterionDict['PSE']['old']:
            A = CriterionDict['A']['old']
            # [parameters, predictions] = _OLS(A, targets, hasBias)
            [parameters, predictions] = _OLS_FLEX(A, targets, hasBias = hasBias, fixed_mapping = coeff_mapping)
            CurrentRegressors.pop(max(CurrentRegressors, key = CurrentRegressors.get))
            # If there was a regressor removed in the current step, we need to identify it and re-add it to
            # the CurrentRegressors in the correct index. 
            if list(RemoveLog.values())[-1] is not None:
                Candidate = list(RemoveLog.values())[-1][0]
                ToRemove = list(RemoveLog.values())[-1][1]
                offset = 0
                # Create a surrogate dictionary since it will change size over the iterations due to the 
                # addition of the missing regressor. Also, we need to build the dictionary from scratch
                # again since the index matters (i.e. point of addition)
                Surrogate_Regressors = {}
                for key, value in CurrentRegressors.items():
                    # Intercept the index where the removed regressor was originally located
                    if value == ToRemove:
                        # Add removed regressor
                        Surrogate_Regressors.update({Candidate:ToRemove})
                        # Offset all following regressors by 1
                        offset = 1
                        Surrogate_Regressors.update({key:value+offset})
                    else:
                        Surrogate_Regressors.update({key:value+offset})
                CurrentRegressors = Surrogate_Regressors
            Stop = True
            print('[ INFO ] PSE increases. Stopping algorithm to prevent overfitting.')
        
        # Check if PSE is below acceptable threshold, if so, take current [A, parameters, CurrentRegressors]
        elif CriterionDict['PSE']['new'] < PSEThreshold:
            A = CriterionDict['A']['new']
            Stop = True
            print('[ INFO ] Current PSE is below threshold of {}. Stopping algorithm to prevent overfitting.'.format(PSEThreshold))

        return Stop, [A, parameters, CurrentRegressors]



    # Function to check coefficient of determination stopping criterion, if selected.
    # Inputs:
    #   CriterionDict = Dictionary containing stoping criteria values for current and previous steps
    #   A = Current regressor matrix
    #   parameters = Current parameters
    #   CurrentRegressors = Dictionary of the regressor IDs and associated indices in A
    #   targets = Model targets (measurement data)
    #   *args = additional arguments
    # Outputs:
    #   Stop = Boolean indicating if algorithm should be stopped
    #   [A, parameters, CurrentRegressors] = Either equivalent to the variables in the current step
    def _checkR2(CriterionDict, A, parameters, CurrentRegressors, targets, *args, **kwargs):
        Stop = False
        if CriterionDict['R2']['new'] <= CriterionDict['R2']['old']*1.005:
            A = CriterionDict['A']['new']
            Stop = True
            print('[ INFO ] R2 shows no significant growth. Stopping algorithm.')
        
        return Stop, [A, parameters, CurrentRegressors]


    
    # Function to check F test stopping criterion, if selected. 
    # Inputs:
    #   CriterionDict = Dictionary containing stoping criteria values for current and previous steps
    #   A = Current regressor matrix
    #   parameters = Current parameters
    #   CurrentRegressors = Dictionary of the regressor IDs and associated indices in A
    #   targets = Model targets (measurement data)
    #   *args = additional arguments
    # Outputs:
    #   Stop = Boolean indicating if algorithm should be stopped
    #   [A, parameters, CurrentRegressors] = Either equivalent to the variables of the previous step
    def _checkF0(CriterionDict, A, parameters, CurrentRegressors, targets, *args, hasBias = True, coeff_mapping = {}):
        Stop = False
        if CriterionDict['F0']['new'] <= CriterionDict['F0']['old']:
            A = CriterionDict['A']['old']
            # [parameters, predictions] = _OLS(A, targets, hasBias = hasBias)
            [parameters, predictions] = _OLS_FLEX(A, targets, hasBias = hasBias, fixed_mapping = coeff_mapping)
            CurrentRegressors.pop(max(CurrentRegressors, key = CurrentRegressors.get))
            Stop = True 
            print('[ INFO ] F0 has reached its maximum value. Stopping algorithm.')

        return Stop, [A, parameters, CurrentRegressors]



    #[----------------------STEPWISE REGRESSION----------------------]
    # Check shape of TrainingTargets
    if len(TrainingTargets.shape) <= 1:
        TrainingTargets = np.matrix(TrainingTargets).T
    # Extract regressors
    FixedRegressors = CompiledModel['Fixed Regressors'].copy()
    CandidateRegressors = CompiledModel['Candidate Regressors'].copy()
    hasBias = CompiledModel['Has Bias']
    # Check if FixedRegressors is empty
    if not bool(FixedRegressors):
        raise ValueError('[ ERROR ] Fixed Regressors is empty. Stepwise regression assumes that there is at least one fixed regressor for initialization.')
    
    # Build RegressorMatrix (i.e. A) from FixedRegressors
    # Pre-allocate array
    RegressorMatrix = np.matrix(np.ones((max(TrainingTargets.shape),len(FixedRegressors.keys()))))
    for col, key in enumerate(FixedRegressors.keys()):
        # Fill in FixedRegressors in RegressorMatrix
        RegressorMatrix[:, col] = FixedRegressors[key]

    # Create empty log of added and removed candidate regressors
    Added_Candidates = {}
    Add_Log = {}
    Remove_Log = {}
    terminationLog = {}
    
    # Initialize stepwise regression algorithm
    [N, p] = RegressorMatrix.shape
    # Number of fixed parameters
    pf = p
    # [params, pred] = _OLS(RegressorMatrix, TrainingTargets, hasBias = hasBias)
    fixed_coefficients = CompiledModel['Fixed Coefficients']
    # offset = 1 if hasBias else 0
    offset = 0
    # coeff_mapping = {_c + offset:{'coeff':fc, 'regressor':r} for _c, (r, fc) in enumerate(fixed_coefficients.items())}
    coeff_mapping = {}
    for _c, r in enumerate(CompiledModel['Fixed Regressors']):
        if r in fixed_coefficients.keys():
            coeff_mapping.update({_c + offset:{
                'coeff':fixed_coefficients[r],
                'regressor':r
            }})
    [params, pred] = _OLS_FLEX(RegressorMatrix, TrainingTargets, hasBias = hasBias, fixed_mapping = coeff_mapping)
    PSE = _PSE(pred, TrainingTargets, p)
    R2 = _CoeffOfDetermination_R2(pred, TrainingTargets)
    # Special handle for when fixed regressors other than the bias vector are added. 
    if p != 1:
        F0 = (N - p)/(p-1)*(R2/(1-R2))
    else:
        F0 = 0

    print('\n[-------------------------------------------------]')
    print('[ INFO ] Initial values')
    print('[ INFO ] Selected regressor: {}'.format(None))
    print('[ INFO ] Removed regressor: {}'.format(None))
    print('[ INFO ] Predict square error: {}'.format(float(PSE.__array__()[0][0])))
    print('[ INFO ] Coefficient of Determination (R2): {}'.format(float(R2.__array__()[0][0])))

    # Initialize CriterionDict. CriterionDict keeps track of the stopping criterion 
    # of the current and previous steps. 
    CriterionDict = {'PSE':{'old':PSE,'new':None},
                        'R2':{'old':R2,'new':None},
                        'F0':{'old':F0,'new':None},
                        'A':{'old':RegressorMatrix.copy(),'new':None}}

    # Define PSE threshold based on initial PSE
    PSE_thresh = 0.001*PSE

    # Skip candidate selection if k_lim == 0 or candidate regressors are empty
    if not CompiledModel['Candidate Regressors']:
        print('[ WARNING ] Candidate regressors appear empty. Continuing with fixed regressor model identification.')
        k_lim = 0
    if k_lim == 0:
        selecting = False
        Final_RegressorMatrix, Final_Parameters = RegressorMatrix, params
    else:
        # Begin at step, k = 1
        k = 1
        selecting = True
        
    while selecting:

        e = TrainingTargets - pred

        # ---------FORWARD STEP---------
        # Determine correlation of candidates in candidate pool with error, e
        Candidates_r = {}
        for candidate, values in CandidateRegressors.items():
            # [params_C, pred_C] = _OLS(RegressorMatrix, values, hasBias = hasBias)
            [params_C, pred_C] = _OLS_FLEX(RegressorMatrix, values, hasBias = hasBias, fixed_mapping = coeff_mapping)
            e_C = values - pred_C
            Sjj_C = np.sum(np.square((e_C - np.nanmean(e_C))))
            Szz_C = np.sum(np.square((e - np.nanmean(e))))
            # Catch for repeated regressors
            if not Sjj_C == 0 and not Szz_C == 0 and not np.isnan(Sjj_C) and not np.isnan(Szz_C):
                r_C = abs(np.sum((e_C - np.nanmean(e_C)).T*(e - np.nanmean(e)))/np.sqrt(Sjj_C*Szz_C))
                Candidates_r.update({candidate:r_C})

        # Find regressor with maximum correlation   
        Candidate_IN = max(Candidates_r, key=Candidates_r.get)
        
        # Schedule Candidate_IN for addition to RegressorMatrix
        ToAdd = _checkFin(RegressorMatrix, CandidateRegressors[Candidate_IN], TrainingTargets, Fin, hasBias = hasBias, coeff_mapping=coeff_mapping)[1]

        # Add Candidate_IN if _checkFin is successful (i.e. Candidate_IN significantly improves 
        # model performance)
        if ToAdd:
            RegressorMatrix = np.hstack((RegressorMatrix, CandidateRegressors[Candidate_IN]))
            # Add regressor name and index of addition in RegressorMatrix to log
            Added_Candidates.update({Candidate_IN:RegressorMatrix.shape[1]-1})
            Add_Log.update({k:Candidate_IN})
            # Remove regressor from remaining candidate pool
            CandidateRegressors.pop(Candidate_IN)
        # If _checkFin fails, then terminate algorithm as no regressors result in improved 
        # performance
        else:
            selecting = False
            Add_Log.update({k:None})
            Stop = True
            Final_RegressorMatrix, Final_Parameters = RegressorMatrix, params
            print('[ INFO ] No qualified candidates found.')
            # Add the added candidates to fixed regressors
            for key in Added_Candidates.keys():
                FixedRegressors.update({key:RegressorMatrix[:, Added_Candidates[key]]})
            break

        
        # ---------BACKWARD STEP---------
        # Check if any of the current regressors have been made redundant
        [ToRemove, F0_out] = _CheckFout(RegressorMatrix, TrainingTargets, Fout, pf, hasBias=hasBias, coeff_mapping=coeff_mapping)

        # If current regressors are all essential for maintaining model accuracy
        if ToRemove is None:
            Remove_Log.update({k:None})
            Candidate_OUT = None
        # Remove redundant regressor, if there is one scheduled for removal
        else:
            # Identify the regressor ID of the to-be-removed regressor
            Candidate_OUT = [key for key, value in Added_Candidates.items() if value == ToRemove][0]
            CandidateRegressors.update({Candidate_OUT:RegressorMatrix[:, ToRemove]})
            # Remove regressor
            RegressorMatrix = np.delete(RegressorMatrix, ToRemove, 1)
            Added_Candidates.pop(Candidate_OUT)
            # Adjust indices of remaining regressors in model
            for idx, key in enumerate(Added_Candidates):
                val = idx + len(FixedRegressors)
                Added_Candidates.update({key:val})
            Remove_Log.update({k:[Candidate_OUT, ToRemove]})

            
        # Check general termination condition: if added regressor is the same as removed regressor
        if Candidate_IN == Candidate_OUT:
            # [params, pred] = _OLS(RegressorMatrix, TrainingTargets, hasBias = hasBias)
            [params, pred] = _OLS_FLEX(RegressorMatrix, TrainingTargets, hasBias = hasBias, fixed_mapping = coeff_mapping)
            Final_RegressorMatrix, Final_Parameters = RegressorMatrix, params
            print('[ INFO ] Added regressor is the same as removed for step {}. Terminating...'.format(k))
            selecting = False
            # Add the added candidates to fixed regressors
            for key in Added_Candidates.keys():
                FixedRegressors.update({key:RegressorMatrix[:, Added_Candidates[key]]})
            break 
        
        # Calculate metrics for current model structure
        # [params, pred] = _OLS(RegressorMatrix, TrainingTargets, hasBias = hasBias)
        [params, pred] = _OLS_FLEX(RegressorMatrix, TrainingTargets, hasBias = hasBias, fixed_mapping = coeff_mapping)
        p = RegressorMatrix.shape[1]
        PSE = _PSE(pred, TrainingTargets, p)
        R2 = _CoeffOfDetermination_R2(pred, TrainingTargets)
        rmsr = np.sqrt(1/N*(np.square((pred - TrainingTargets))))/(np.max(TrainingTargets) - np.min(TrainingTargets))
        F0 = (N - p)/(p-1)*(R2/(1-R2))

        # Update log and criterion dict
        terminationLog.update({k:{'PSE':PSE, 'R2':R2, 'rmsr':rmsr, 'F0':F0, 'A':RegressorMatrix}})
        CriterionDict = _UpdateCriteria(CriterionDict.copy(), 'new', terminationLog[k])
        
        # Display some results
        print('\n[-------------------------------------------------]')
        print('[ INFO ] Current step: {}'.format(k))
        print('[ INFO ] Bias: {}'.format(params[0]))
        print('[ INFO ] Selected regressor: {}'.format(Candidate_IN))
        print('[ INFO ] Removed regressor: {}'.format(Candidate_OUT))
        print('[ INFO ] Predict square error: {}'.format(float(PSE.__array__()[0][0]),))
        print('[ INFO ] Coefficient of Determination (R2): {}'.format(float(R2.__array__()[0][0])))

        # Check additional stopping critera
        checks = {'PSE': _checkPSE,
                    'R2': _checkR2,
                    'F0': _checkF0}
        # If user selected one of the known stopping criteria
        if stop_criteria in checks.keys():
            Stop, [Final_RegressorMatrix, Final_Parameters, Added_Candidates] = checks[stop_criteria](CriterionDict, RegressorMatrix, params, Added_Candidates, TrainingTargets, PSE_thresh, Remove_Log, hasBias=hasBias, coeff_mapping=coeff_mapping)
            if force_k_lim and not Stop:
                if k >= k_lim:
                    Stop = True
                    Final_RegressorMatrix, Final_Parameters = RegressorMatrix, params
                    print('[ INFO ] Current step has reached upper limit (k_lim = {}). Terminating selection process.'.format(k_lim))
        # Otherwise default to step cap (k_lim) to avoid overfitting. 
        else:
            if k >= k_lim:
                Stop = True
                Final_RegressorMatrix, Final_Parameters = RegressorMatrix, params
                print('[ INFO ] Current step has reached upper limit (k_lim = {}). Terminating selection process.'.format(k_lim))
            else:
                Stop = False

        # Check if there are any regressors left
        if len(CandidateRegressors.keys()) == 0:
            print('[ INFO ] No more candidate regressors, stopping algorithm.')
            selecting = False
            Final_RegressorMatrix, Final_Parameters = RegressorMatrix, params
            Stop = True

        # If any (chosen) stop criteria have been met
        if Stop:
            # Add the added candidates to fixed regressors
            for key in Added_Candidates.keys():
                FixedRegressors.update({key:RegressorMatrix[:, Added_Candidates[key]]})

            selecting = False
        # If no stop criteria have been met, move to next step
        else:
            CriterionDict = _UpdateCriteria(CriterionDict.copy(), 'old', terminationLog[k])
            k += 1

    # Compile model output    
    error_est = Final_RegressorMatrix * Final_Parameters - TrainingTargets
    sigma2_est = 1/(N-2)*np.sum(np.square(error_est))
    inv_XtX = np.linalg.inv(np.dot(np.transpose(Final_RegressorMatrix), Final_RegressorMatrix))
    out = {'Model':{'A':Final_RegressorMatrix, 'Parameters':Final_Parameters, 'Regressors':list(FixedRegressors.keys()), 'Has Bias':hasBias, '_sigma2':sigma2_est, '_inv(XtX)':inv_XtX},
            'Additional':{'Log':{'Add Log':Add_Log, 'Remove Log':Remove_Log, 'Info Log':terminationLog}}}

    return out



def predict(Model, x):
    '''Function to make predictions on input data using the model identified through stepwise regression
    
    :param x: Pandas DataFrame containing the input data upon which predictions should be made. Shape [N x M] with N = number of observations and M = number of independent variables.
    :param Model: Dictionary containing model values. Equivalent to out['Model'] of function <train>. 
    :return pred: numpy.matrix of model predictions based on x, shape [N x 1].  
    :return var: variance of errors associated with the predictions, based on the training data. 
    '''    
    A = _BuildRegressorMatrix(Model['Model']['Regressors'], x, hasBias=Model['Model']['Has Bias'])
    pred = A*Model['Model']['Parameters']
    s2 = Model['Model']['_sigma2']
    inv_XtX = Model['Model']['_inv(XtX)']
    var = pred.copy()*0
    AT = np.transpose(A)
    for i in range(len(var)):
        var[i] = s2 + s2*np.dot(np.dot(A[i, :], inv_XtX), AT[:, i])
    # var = s2*(np.identity(len(A)) + np.dot(np.dot(A, inv_XtX), np.transpose(A)))
    return pred, var



def evaluate(Model, inputs, target, showPlots = True):
    '''Function to validate the chosen model through an analysis of the model residuals and parameter covariances
    
    :param Model: Dictionary containing model values, in SysID.Model format. Equivalent to output of function <train>.
    :param inputs: Pandas DataFrame containing the input data for evaluation. Shape [N x M] with N = number of observations and M = number of independent variables.
    :param target: Targets (measurement data) to evaluate against. Shape [N x 1], N = number of observations
    :param showPlots:  Dictates if plots should be shown or not. Possible values are: True, False, and 'plot'.Default is True, plots will be drawn and shown. False means plots will not be drawn or shown. 'plot' means plots will be drawn but not shown. Users can call matplotlib.pyplot.show() when ready to show drawn plots. 
    :return: Dictionary containing:
        output['Residual Error'] = Model residual errors,
        output['Autocorrelation'] = Residual error autocorrelation,
        output['Index'] = Sample number of output['Residual Error'],
        output['COV'] = Parameter covariances,
        output['Coefficient Variance'] = Variance of a parameter (i.e. diag(COV))
    '''    
    # Obtain model prediction
    pred, _ = predict(Model, inputs)

    # Check shapes of target and pred
    if target.shape != pred.shape:
        try:
            target = np.array(target).reshape(pred.shape)
        except ValueError:
            raise ValueError('input shapes of pred {} and target {} are not the same.'.format(pred.shape, target.shape))

    # Model residuals
    N = max(target.shape)
    index = np.arange(0, N, 1)
    error = np.array((target - pred))
    e_mean = np.nanmean(error)
    e_sig = np.nanstd(error)

    # Error Autocorrelation
    confidenceBounds = 1.96/np.sqrt(N)
    n_index = -1*np.array(sorted(index, reverse=True))[:-1]
    lag = np.hstack((n_index, index))
    autocorrelation = np.correlate((error.reshape(index.shape) - np.nanmean(error))/(e_sig*N), (error.reshape(index.shape) - np.nanmean(error))/e_sig, 'full')


    # Model parameter covariance
    A = Model['Model']['A']
    # Using: sigma = e.T * e / (N - p)
    mse = np.matrix(error).T * np.matrix(error) / (N - min(A.shape))
    COV = float(mse.__array__()[0][0]) * np.linalg.inv((A.T * A))
    coeffVariance = np.diag(COV)
    coeffIndex = np.arange(1, min(A.shape) + 1, 1)

    output = {'Residual Error':error, 
                'Autocorrelation':autocorrelation, 
                'Index':index, 
                'COV':COV, 
                'Coefficient Variance':coeffVariance}


    if showPlots is not False:
        from matplotlib.ticker import AutoMinorLocator

        fig = plt.figure('Residual error between targets and predictions')
        ax = fig.add_subplot(111)
        # plt.title('Residual error between targets and predictions')
        ax.plot(index, error.reshape(index.shape), color = 'royalblue', label = 'Residual error')
        ax.plot(index, np.ones(index.shape)*e_mean, color='k', linestyle='--', label = 'Mean = {}'.format(str(round(e_mean, 7))))
        ax.plot(index, np.ones(index.shape)*e_sig + e_mean, color = 'tab:red', linestyle='--', label = r'$1-\sigma$ bounds')
        ax.plot(index, -1*np.ones(index.shape)*e_sig + e_mean, color = 'tab:red', linestyle='--')
        ax.set_xlabel(r'$\mathbf{Sample} \quad [-]$')
        ax.set_ylabel(r'$\mathbf{Error} \quad [-]$')
        ax.grid()
        ax.tick_params(which='both', direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(loc='upper right')

        fig2 = plt.figure('Residual error autocorrelation')
        ax2 = fig2.add_subplot(111)
        # plt.title('Residual error autocorrelation')
        ax2.plot(lag, autocorrelation, color = 'royalblue', label = 'Residual error')
        # ax2.plot(lag, confidenceBounds*np.ones(lag.shape), color = 'tab:red', linestyle = '--', label='95% confidence bounds')
        # ax2.plot(lag, -confidenceBounds*np.ones(lag.shape), color='tab:red', linestyle = '--')
        ax2.set_xlabel(r'$\mathbf{Lag} \quad [-]$')
        ax2.set_ylabel(r'$\mathbf{Autocorrelation} \quad [-]$')
        ax2.tick_params(which='both', direction='in')
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.grid()
        ax2.legend(loc='upper right')

        fig3 = plt.figure('Coefficient variance')
        ax3 = fig3.add_subplot(111)
        # plt.title('Coefficient variance')
        # ax3.bar(coeffIndex, np.array(Model['Model']['Parameters']).reshape(coeffIndex.shape), color='royalblue', label='Polynomial coefficients')
        ax3.bar(coeffIndex, np.abs(np.array(Model['Model']['Parameters']).reshape(coeffIndex.shape)), color='royalblue', label='Polynomial coefficients')
        ax3.bar(coeffIndex, np.array(coeffVariance).reshape(coeffIndex.shape), color='orangered', label='Coefficient variance')
        ax3.grid()
        ax3.set_xlabel(r'$\mathbf{Regressor}$')
        ax3.set_ylabel(r'$\mathbf{Coefficient \quad Magnitude} \quad [-]$')
        ax3.tick_params(which='both', direction='in')
        ax3.xaxis.set_minor_locator(AutoMinorLocator())
        ax3.yaxis.set_minor_locator(AutoMinorLocator())
        # plt.xticks(ticks=coeffIndex, labels=list(Model['Regressors'].keys()))
        ax3.set_xticks(ticks=coeffIndex)
        ax3.set_xticklabels(labels=Model['Model']['Regressors'])
        for tick in ax3.get_xticklabels():
            tick.set_rotation(80)

        if showPlots is True:
            plt.show()

    return output



def summary(Status, Model):
    '''Function to print a summary of the model to the terminal

    :param Status: State of the model, inherited from SysID class. 
    :param Model: Model corresponding to the state, inherited from SysID class.
    :return None:
    '''
    if Status.lower() == 'compiled':
        # Fixed regressors
        print('{:{fill}^65}'.format('Fixed Regressors', fill='-'))
        if len(Model['Fixed Regressors']) > 0:
            for r in Model['Fixed Regressors'].keys():
                print('{:<65}'.format(r))
        else:
            print('{:>65}'.format('None'))
        # Candidate regressors
        print('{:{fill}^65}'.format('Candidate Regressors', fill='-'))
        for i, p in enumerate(Model['Candidate Polynomials']):
            print('{:{fill}^65}'.format('Polynomial {}'.format(i), fill='.'))
            dummy = '{:<15} ' + '{},'*len(p['vars'])
            print(dummy[:-1].format('Variables:', *p['vars']))
            print('{:<15} {:<39}'.format('Degree:', p['degree']))
            dummy = '{:<15} ' + '{},'*len(p['sets'])
            print(dummy[:-1].format('Sets:', *p['sets']))
        print('-'*65)
        print('{:<25} {:>39}'.format('Total number of terms', len(Model['Candidate Regressors'])))
        print('_'*65)
    elif Status.lower() == 'trained':
        print('{:{fill}^65}'.format('Chosen Regressors', fill='-'))
        print('{:<54} {:>10}'.format('Regressor', 'Value'))
        print('-'*65)
        for i, r in enumerate(Model['Model']['Regressors']):
            # print('{:<65}'.format(r))
            print('{:<54} {:.3e}'.format(r.replace(' ', ''), float(Model['Model']['Parameters'][i].__array__()[0][0])))
        print('_'*65)
        print('{:<25} {:>39}'.format('Total number of terms', len(Model['Model']['Regressors'])))
        print('_'*65)
    else:
        print('{:^65}'.format('Model not compiled!'))
    return None



def save(path, model):
    '''Function to save (trained) stepwise regression model 
    
    :param path: Save directory
    :param model: TrainedModel to save
    :return: None
    '''
    # Create sub-directory to save model to
    saveDir = os.path.join(path, 'stepwise_regression')
    if not os.path.isdir(saveDir):
        os.mkdir(saveDir)

    with open(os.path.join(saveDir, 'model.pkl'), 'wb') as f:
        pkl.dump(model, f)
        f.close()

    return None



def load(path):
    '''Function to load trained stepwise regression model
    
    :param path: Model directory
    :return: TrainedModel in SysID.Model format
    '''
    modelDir = os.path.join(path, 'stepwise_regression')
    files = [f for f in os.listdir(modelDir) if os.path.isfile(os.path.join(modelDir, f)) and f.endswith('.pkl')]
    if len(files) > 1:
        print('[ WARNING ] Multiple (model) .pkl files found in {} when I only expected one. Checking which ones contain the model, and taking the first which satisfies this.'.format(path))
    
    i = 0
    lookingForModel = True
    model = None
    while lookingForModel:
        if i == len(files):
            lookingForModel = False
            raise FileNotFoundError('[ ERROR ] Could not find appropriate model file.')
        else:
            f = files[i]
            with open(os.path.join(modelDir, f), 'rb') as ff:
                model = pkl.load(ff)
                ff.close()
            try:
                if 'Model' in model.keys():
                    lookingForModel = False
            except AttributeError:
                pass
            i += 1 

    return model



def reduceModel(polyModel, inputs, targets, covarianceThreshold = 0.1):
    '''Function to reduce model by removing high covariance regressor terms, if any
    
    :param polyModel: SysID.Model object of the trained polynomial model
    :param inputs: Pandas DataFrame (Shape NxM where N = number observations, M = number input variables) of input data used to evaluate model performance
    :param targets: Pandas DataFrame of associated targets for inputs
    :param covarianceThreshold: Float < 1 which denotes the maximum allowable covariance, as a percentage of the associated coefficient magnitude. Default is 0.1 (i.e. 10%)
    :return: Reduced model.  
    '''
    Model = polyModel.TrainedModel['Model']

    pred, _ = polyModel.predict(inputs)
    error = np.array(targets - np.array(pred).reshape(targets.shape)).reshape(-1, 1)
    N = len(targets)
    A = Model['A']

    # Using: sigma = e.T * e / (N - p)
    mse = np.matrix(error).T * np.matrix(error) / (N - min(A.shape))
    COV = float(mse) * np.linalg.inv((A.T * A))
    coeffVariance = np.diag(COV)
    coeffIndex = np.arange(1, min(A.shape) + 1, 1)
    highCovariances = [c for c in (coeffIndex-1) if coeffVariance[c] >= np.abs(covarianceThreshold*Model['Parameters'][c])]

    if len(highCovariances):
        cutoff = highCovariances[0]
        reducedModel = trimModel(polyModel, cutoff)
    else:
        reducedModel = polyModel.TrainedModel['Model']

    return reducedModel



def trimModel(polyModel, cutoff):
    '''Function to remove regressors added after a specified index
    
    :param polyModel: SysID.Model object of the trained polynomial model
    :param cutoff: Integer index afterwhich regressors should be removed, removal includes cutoff. 
    :return: Reduced model
    '''
    x_train = polyModel.x_train
    y_train = polyModel.y_train
    _regressors = polyModel.TrainedModel['Model']['Regressors'][:cutoff]
    A = _BuildRegressorMatrix(_regressors, x_train, hasBias = polyModel.TrainedModel['Model']['Has Bias'])
    # [params, _] = _OLS(A, np.array(y_train).reshape(-1, 1), hasBias = polyModel.TrainedModel['Model']['Has Bias'])
    [params, _] = _OLS_FLEX(A, np.array(y_train).reshape(-1, 1), hasBias = polyModel.TrainedModel['Model']['Has Bias'], fixed_mapping=polyModel.CompiledModel['Fixed Coefficients'])
    reducedModel = {}
    for k, v in polyModel.TrainedModel['Model'].items():
        reducedModel.update({k:v})
    reducedModel['A'] = A
    reducedModel['Parameters'] = params
    reducedModel['Regressors'] = _regressors
    reducedModel['_inv(XtX)'] = np.linalg.inv(np.dot(np.transpose(A), A))
    error_est = A * params - np.array(y_train).reshape(-1, 1)
    sigma2_est = 1/(len(y_train)-2)*np.sum(np.square(error_est))
    reducedModel['_sigma2'] = sigma2_est
    return reducedModel



def dropRegressor(polyModel, index, retrain = True):
    '''Function to drop a regressor, at index i, from a polynomial model

    :param polyModel: SysID.Model object of the trained polynomial model
    :param index: Integer index corresponding to the regressor to be removed.
    :param retrain: Boolean to indicate if model should be re-fit to the training data using OLS. Default is True. 
    :return: Model with regressor dropped
    '''
    x_train = polyModel.x_train
    y_train = polyModel.y_train
    _regressors = polyModel.TrainedModel['Model']['Regressors'][:index] + polyModel.TrainedModel['Model']['Regressors'][(index+1):]
    if retrain:
        A = _BuildRegressorMatrix(_regressors, x_train, hasBias = polyModel.TrainedModel['Model']['Has Bias'])
        # [params, _] = _OLS(A, np.array(y_train).reshape(-1, 1), hasBias = polyModel.TrainedModel['Model']['Has Bias'])
        [params, _] = _OLS_FLEX(A, np.array(y_train).reshape(-1, 1), hasBias = polyModel.TrainedModel['Model']['Has Bias'], fixed_mapping = polyModel.CompiledModel['Fixed Coefficients'])
        reducedModel = {}
        for k, v in polyModel.TrainedModel['Model'].items():
            reducedModel.update({k:v})
        reducedModel['A'] = A
        reducedModel['Parameters'] = params
        reducedModel['Regressors'] = _regressors
        reducedModel['_inv(XtX)'] = np.linalg.inv(np.dot(np.transpose(A), A))
        error_est = A * params - np.array(y_train).reshape(-1, 1)
        sigma2_est = 1/(len(y_train)-2)*np.sum(np.square(error_est))
        reducedModel['_sigma2'] = sigma2_est
    else:
        A = _BuildRegressorMatrix(_regressors, x_train, hasBias = polyModel.TrainedModel['Model']['Has Bias'])
        keepIdxs = np.delete(np.arange(0, len(polyModel.TrainedModel['Model']['Regressors'])), index)
        params = np.matrix(np.array(polyModel.TrainedModel['Model']['Parameters']).reshape(-1)[keepIdxs]).T
        reducedModel = {}
        for k, v in polyModel.TrainedModel['Model'].items():
            reducedModel.update({k:v})
        reducedModel['A'] = A
        reducedModel['Parameters'] = params
        reducedModel['Regressors'] = _regressors
        reducedModel['_inv(XtX)'] = np.linalg.inv(np.dot(np.transpose(A), A))
        error_est = A * params - np.array(y_train).reshape(-1, 1)
        sigma2_est = 1/(len(y_train)-2)*np.sum(np.square(error_est))
        reducedModel['_sigma2'] = sigma2_est
    return reducedModel



def copy(model):
    '''Function to make a copy of a polynomial model
    
    :param model: SysID.Model.TrainedModel object of the trained polynomial model
    :return: Copy of model 
    '''
    modelCopy = {}
    for i in model.keys():
        subDict = {}
        for j in model[i].keys():
            subDict.update({j:model[i][j]})
        modelCopy.update({i:subDict})
    return modelCopy


def plotRegressorContributions(polyModel, inputs = None, x = None, returnFig = False, normalizer = None, colors = None, legendLoc = 'best'):
    '''Function to plot the individual regressor contributions in the context of the final polynomial model structure and parameter values.
    Also returns the regressor contributions in terms of RMSE and R2.

    :param polyModel: SysID.Model object of the trained polynomial model
    :param inputs: Pandas DataFrame (Shape NxM where N = number observations, M = number input variables) of input data for which these regressor contributions should be plotted. 
        Default = None (Using training data from polyModel.x_train, if available)
    :param x: Array-like to use as x-axis in plot. Default = None (i.e. x-axis represents index)
    :param returnFig: Boolean to indicate if figure object should be returned or not. Default = False
    :param normalizer: Array-like of normalizing factor for data, if used. Default = None (i.e. unused)
    :param colors: List-like of colors to be used for plotting. Colors will be cycled through so repetitions will occur if len(colors) < number of regressors. Default = None (colors will be inferred from a cmap)
    :param legendLoc: String indicating location of legend. Locations have to be compatible with the equivalent parameter in the matplotlib library.
    
    :return: Figure object (if returnFig = True) in a dictionary accessible with key 'fig'.
    '''
    if inputs is None:
        inputs = polyModel.x_train
    outputs = {}
    if x is None:
        x = np.arange(0, len(inputs))
    regressors = polyModel.TrainedModel['Model']['Regressors']
    if colors is None:
        # cmap = clrs.Colormap('rainbow', N = len(polyModel.TrainedModel['Model']['Regressors']))
        cNorm = clrs.Normalize(vmin=0, vmax=len(polyModel.TrainedModel['Model']['Regressors']))
        # scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap='rainbow')
        colors = cycle([scalarMap.to_rgba(i) for i in range(len(polyModel.TrainedModel['Model']['Regressors']))])
    else:
        colors = cycle(colors)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, r in enumerate(regressors[::-1]):
        if i != 0:
            surrogateModel = {}
            for k, v in polyModel.TrainedModel['Model'].items():
                surrogateModel.update({k:v})
            surrogateModel['A'] = surrogateModel['A'][:, :-i]
            surrogateModel['Parameters'] = surrogateModel['Parameters'][:-i]
            surrogateModel['Regressors'] = surrogateModel['Regressors'][:-i]
            surrogateModel['_inv(XtX)'] = surrogateModel['_inv(XtX)'][:-i, :-i]
            pred, _ = predict({'Model':surrogateModel}, inputs)
        else: 
            pred, _ = polyModel.predict(inputs)
        if normalizer is not None:
            pred = np.array(pred).reshape(-1)*np.array(normalizer).reshape(-1)
        ax.plot(x, pred, color = next(colors), label = '+ {}'.format(r))
    handles, labels = ax.get_legend_handles_labels()
    newHandles = []
    for h, l in zip(handles[::-1], labels[::-1]):
        newHandles.append(mpatches.Patch(color = h._color, label = l))
    ax.legend(handles=newHandles, loc=legendLoc)
    ax.set_xlabel(r'$\mathbf{Sample}$ [-]', fontsize = 14)
    ax.set_ylabel(r'$\mathbf{Regressor}$ $\mathbf{contribution}$', fontsize = 14)
    if returnFig:
        outputs.update({'fig':fig})
    else:
        plt.show()
    return outputs