# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 10:57:45 2016

@author: Matthew
"""

# This import block has basically everything you could need for calculating
# and graphing your results. Unused modules are left in the code for reference.
from pyomo.environ import *
from pyomo.dae import *
from pyomo.opt import SolverFactory
import numpy as np
import scipy as sp
from scipy.stats.distributions import chi2
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

# You must import a function, here VdV_model, that solves a pyomo ConcreteModel
# and returns the solved instance and its results. See VdV_model for an example
# of that looks.
from ExTEG_module import TEG_model as model

# Import the PyMPLE class. This has everything you need for plotting and
# Analyzing your model response to profiling.
from PyMPLE import PyMPLE

# This is maybe the only example-specific line of code in here, but the data
# is specific to the problem. Don't use this data on your problem. Or do.
import pandas as pd

#Set up data from file
data=pd.read_csv('ExampleTEG.txt',delimiter='\t') 
data=data[["Time(sec)","Value"]]
data=data.dropna()
data=data.set_index("Time(sec)") #indexing by time
data=data/4 # get data into mm, then replicate top half of TEG
datadiff=np.diff(np.reshape(data.values,data.shape[0])) #find the difference between each point
maxdiff=np.max(np.abs(datadiff)) #getting the largest jump between points (5 sec apart)
indexmax=np.argmax(np.abs(datadiff))*5 #getting the index of this max... multiplying by 5 to get into seconds
#This version is aimed at removing discontinuieties
#plt.close("all")
if maxdiff >= 1.7 and indexmax >= 200:
    print(':(')
    dataold=data
    data=data[:(indexmax)-5]
    #plt.figure(1)
    #plt.plot(data,'k',dataold,'y--')  
else:
    print(':)')
    #plt.figure(1)    
    #plt.plot(data,'k')

# Provide some initialization values for your pyomo model that the class can 
# also use after the initial pyomo solution.
solver = 'ipopt'# your pyomo-approved solver

############################################################################
#Model Jam
# Reactions:
# P <-> Pa
# Pa -> T
# T -> L
# P: Platelets
# Pa: Activated platelet
# T: Thrombus
# L: Lysis
#Rate constants
k1f = 0.005
k1b=0.0
k2=0.04
k3=0.00004

#Initial Conditions:
Pa0 = 0.0;
T0 = 0.0;
L0 = 0.0;
P0 = np.max(data.Value);


p0 = [k1f, k2, k3, P0] # pyomo will need an initial condition vector
#                                 might as well provide that here
pkey = ['k1f', 'k2', 'k3', 'P0'] # Your parameter names.

pfix = [0,0,0,0] # This is the vector of parameters that you don't want to
#                    profile. Ask, "Do I want to exclude this from profiling?"
#                    If yes, then enter 1 at its index location to exclude it.
nfinels = 300#len(data) # Tell pyomo (and the class) how many finite elements you want.

mstates = ['P','Pa','T','L'] # This is the names of your states for plotting trajectories

bds = [(1.0e-7,1.0e0),(1.0e-7,1.0e0),(1.0e-7,1.0e0),(1,70)] # Your model probably has bounds.
#                                              You should add them here.
# Solver your model to get an initial point for profiling
inst, mres = model(data, solver, nfinels, pfix, p0, pkey, bound=bds)
print('I finished the first optimization step!')
# Load your model solutions.
inst.solutions.load_from(mres)
# Get the reference objective value for profiling.
obj = value(inst.obj)

# This list comprehension auto-generates your initial parameter vector for profiling
popt = [value(getattr(inst,pkey[i])) for i in range(len(pkey))]

# Generate an instance of the PyMPLE class
PL_instance = PyMPLE(popt, model, obj, pkey, bds, mstates, True)

# This an exomple of how to generate a vector of the relative step-sizes you
# want to take when profiling each parameter.
sfvec = [0.001 for i in range(len(popt))]
nsteps = 1000 # the max number of steps to take when profiling.

# Perform the profiling with this command. Returns the upper and lower bounds
fin = PL_instance.get_CI(pfix,data,nsteps,sfvec,solver,nfinels)

# Plot your profile likelihoods that were obtained from the get_CI method
PL_Fig = PL_instance.plot_PL()

# You can also use the ebarplots method to plot your parameters as a bar with CIs
PL_instance.ebarplots()

# And you can also plot each of the profiled state trajectories
PL_instance.plot_trajectories(mstates)
