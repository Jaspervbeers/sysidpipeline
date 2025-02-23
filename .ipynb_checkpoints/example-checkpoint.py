'''
Example script describing how to use the SysID library. 

Here, we aim to approximate the function: F(t) = 0.732 + sin(t) + 0.64 * (cos(pi * t))^(2) + 0.32 * cos(pi * t) * sin(t) - 0.25 * (sin(t))^(2) + white_noise
'''
import SysID
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
from Utility import Utility
import os

'''
Definitions
'''
def addWhiteNoise(x, seed, mu = 0, sigma = 0.05):
    # Set seed in isolated variable
    rng = np.random.RandomState(seed)
    # create white noise signal
    white_noise = rng.normal(mu, sigma, x.shape)
    return x + white_noise


def getF(t):
    F = 0.732 + np.sin(t) + 0.64*(np.cos(np.pi*t))**(2) + 0.32*np.cos(np.pi*t)*np.sin(t) - 0.25 * (np.sin(t))**(2)
    return F



'''
Defining the target signals and variables
'''
tStart = 0
tEnd = 20
dt = 0.005
t = np.arange(tStart, tEnd, dt)

x1_clean = np.sin(t)
x2_clean = np.cos(np.pi*t)
x1 = addWhiteNoise(x1_clean, 1111)
x2 = addWhiteNoise(x2_clean, 2222)
F_clean = getF(t)
F_noisy = addWhiteNoise(F_clean, 3333)

Data = pd.DataFrame(np.vstack((x1, x2)).T, columns=['x1', 'x2'])

predictionIntervalConfidenceLevel = 0.95



'''
Polynomial model

Regressor selection:
    x1 = sin(t) + white_noise
    x2 = cos(np.pi * t) + white_noise
So, the most appropriate polynomial approximation of F(t) is:
    y =  0.732 + x1 + 0.64 * (x2)^(2) + 0.32 * x2 * x1 - 0.25 * (x1)^(2)
'''
# Initialize model
PolyModel = SysID.Model('Stepwise_Regression')

# Define candidate model structure
polyCandidates = [
    {'vars':['x1', 'x2'],
     'degree':3,
     'sets':[1]}
]
fixedRegressors = []

# Compile model
PolyModel.compile(Data, polyCandidates, fixedRegressors, includeBias = True)
# Train model
PolyModel.train(Data, F_noisy, stop_criteria = 'PSE')
# Make predictions
polyPred, polyPredVar = PolyModel.predict(Data)
# Get polynomial (95%) prediction interval bounds
PolyPI_lower, PolyPI_upper = Utility.buildIntervalBounds(predictionIntervalConfidenceLevel, polyPred, polyPredVar)

# Print identified model; we see that the selected regressors match the analytical solution with similar coefficients to the expected solution. 
PolyModel.summary()

# Save polynomial model 
print('[ INFO ] Saving polynomial model')
savePath = os.path.join(os.getcwd(), 'exampleData', 'examplePolyModel')
PolyModel.save(savePath)


# Plot results
fig = plt.figure()
ax = fig.add_subplot(111)
ax.fill_between(t, np.array(PolyPI_lower).reshape(-1), np.array(PolyPI_upper).reshape(-1), color = 'firebrick', alpha = 0.5)
ax.plot(t, polyPred, label = 'Polynomial prediction', color='firebrick')
ax.plot(t, F_clean, label='Targets (noise-free)', linestyle = '--', color = 'k')
ax.plot(t, F_noisy, label='Targets (noisy)', linestyle = '-', color = 'grey', alpha = 0.7)
ax.set_xlabel(r'$\mathbf{Time} \quad [s]$', fontsize = 16)
ax.set_ylabel(r'$\mathbf{F} \quad [A.U.]$', fontsize = 16)
ax.legend()

plt.show()



'''
ANN Model
'''
# Initialize model
NeuralNetModel = SysID.Model('ANN')

# Compile (single layer) ANN model. modelOutput = Point means that the prediction intervals (PIs) will be estimated through the bootstrap method (i.e. external ANN estimates PIs.)
NeuralNetModel.compile(Data, F_noisy, num_ensembles = 10, ANN_type = 'FNN', num_hidden_layers = 1, num_hidden_neurons = 10, hidden_activation = 'relu', modelOutput = 'point')
# Train ANN model
NeuralNetModel.train(Data, F_noisy, applyNormalization = True, epochs = 10, bootstrapPIs = True)
# Make predictions
NeuralNetPred, NeuralNetPredVar = NeuralNetModel.predict(Data)
# Get polynomial (95%) prediction interval bounds
NeuralNetPI_lower, NeuralNetPI_upper = Utility.buildIntervalBounds(predictionIntervalConfidenceLevel, NeuralNetPred, NeuralNetPredVar)


# Plot results
fig = plt.figure()
ax = fig.add_subplot(111)
ax.fill_between(t, np.array(NeuralNetPI_lower).reshape(-1), np.array(NeuralNetPI_upper).reshape(-1), color = 'royalblue', alpha = 0.5)
ax.plot(t, NeuralNetPred, label = 'ANN prediction', color='royalblue')
ax.plot(t, F_clean, label='Targets (noise-free)', linestyle = '--', color = 'k')
ax.plot(t, F_noisy, label='Targets (noisy)', linestyle = '-', color = 'grey', alpha = 0.7)
ax.set_xlabel(r'$\mathbf{Time} \quad [s]$', fontsize = 16)
ax.set_ylabel(r'$\mathbf{F} \quad [A.U.]$', fontsize = 16)
ax.legend()

plt.show()


'''
Comparison between models
'''
# Compare PI and fit performance
PolyPICP, PolyMPIW = Utility.qualityPI(F_noisy, np.array(polyPred).reshape(-1), np.array(polyPredVar).reshape(-1), conf=predictionIntervalConfidenceLevel)
PolyRMSE = PolyModel._RMSE(F_noisy, polyPred)
NeuralNetPICP, NeuralNetMPIW = Utility.qualityPI(F_noisy, np.array(NeuralNetPred).reshape(-1), np.array(NeuralNetPredVar).reshape(-1), conf=predictionIntervalConfidenceLevel)
NeuralNetRMSE = NeuralNetModel._RMSE(F_noisy, NeuralNetPred)

# PICP = Proportion of data contained within estimated prediction intervals
#   Valid models for PICP >= predictionIntervalConfidenceLevel (so PICP >= 95% if confidence level of 0.95 is chosen)
# MPIW = Mean width of the prediction intervals
#   The lower the MPIW, the better (narrow PIs), assuming valid PICP
# RMSE = Root mean squared error
#   The lower the RMSE, the closer the prediction resembles the training targets. 
print('[ INFO ] Performance comparison')
print('\t Polynomial model:')
print('\t\t{:<10} {:.2f}'.format('PICP =', PolyPICP))
print('\t\t{:<10} {:.2f}'.format('MPIW =', PolyMPIW))
print('\t\t{:<10} {:.5e}'.format('RMSE =', PolyRMSE))
print('\t Neural network model:')
print('\t\t{:<10} {:.2f}'.format('PICP =', NeuralNetPICP))
print('\t\t{:<10} {:.2f}'.format('MPIW =', NeuralNetMPIW))
print('\t\t{:<10} {:.5e}'.format('RMSE =', NeuralNetRMSE))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.fill_between(t, np.array(NeuralNetPI_lower).reshape(-1), np.array(NeuralNetPI_upper).reshape(-1), color = 'royalblue', alpha = 0.5)
ax.plot(t, NeuralNetPred, label = 'ANN prediction', color='royalblue')
ax.fill_between(t, np.array(PolyPI_lower).reshape(-1), np.array(PolyPI_upper).reshape(-1), color = 'firebrick', alpha = 0.5)
ax.plot(t, polyPred, label = 'Polynomial prediction', color='firebrick')
ax.plot(t, F_clean, label='Targets (noise-free)', linestyle = '--', color = 'k')
# ax.plot(t, F_noisy, label='Targets (noisy)', linestyle = '-', color = 'grey', alpha = 0.5)
ax.set_xlabel(r'$\mathbf{Time} \quad [s]$', fontsize = 16)
ax.set_ylabel(r'$\mathbf{F} \quad [A.U.]$', fontsize = 16)
ax.legend()

plt.show()