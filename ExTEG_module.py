# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:13:00 2015

@author: Matthew
"""

#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# This import header should probably be used in every model

from pyomo.environ import *
from pyomo.dae import *
from pyomo.opt import SolverFactory
from pyomo.dae.plugins.finitedifference import Finite_Difference_Transformation
#from pyomo.dae.plugins.colloc import Collocation_Discretization_Transformation
import math
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import interactive
interactive(True)

'''
You should set up your function to look something like this. It should have 
all of the arguments below, but you can add whatever else you want.

data: is your data. It can be a list or ndarray, but the values have to be
floats if you're going to use a sum-squared error objective.

solver: any AMPL solver that is compatible with pyomo

nfinels: the number of finite elements that you will use for either finite
difference or collocation on finite elements methods.

p0: the initial parameter vector that pyomo is going to operate on.

pkey: the list of name strings for your parameters

bound: a tuple or list of tuples (prefered) containing the bounds of your model
parameters. NOTE: this is not the profile bounds; it is the pyomo search-space
bounds. If you just want to search the positive reals, you can use the default
format given in the example function declaration below.

cp: is the number of collocation points that you will use if you elect to 
employ collocation on finite elements to integrate your system.

'''
def TEG_model(data, solver, nfinels, pfix, p0, pkey, bound=(0.0,None), cp=3):
    # You should include this if and only if you are too lazy to pass your own
    # bounds list.
    if type(bound)==tuple:
        mbounds = [bound for i in range(len(pfix))]
    elif type(bound)==list:
        mbounds = bound
    #Dependance on activated platelets
    n=2;
    #Initial Conditions:
    Pa0 = 0.0;
    T0 = 0.0;
    L0 = 0.0;
#    P0 = np.max(data.Value);
    k1b=0.0;
    # Initiliaze model as a pyomo ConcreteModel(). 
    # create the model
    model = ConcreteModel()
    model.t=ContinuousSet(initialize=data.index.values)
    model.time=Set(initialize=data.index.values, within=model.t, ordered=True)
    for i in range(len(pfix)):
    # This for loop has to be present in your model in order for the get_CI()
    # method to operate for a general set of parameters specified in the run file.
    # It accomplishes the following assignments for the VdV example case when
    # profiling only FV and kAB.
        # m.FV  = Var(initialize=3/7)# True Value 4/7
        # m.Caf = Param(initialize=10)
        # m.kAB = Var(initialize=3/6)# True Value 5/6
        # m.kBC = Param(initialize=5/3)
        # m.kAD = Param(initialize=1/6)
        
        if pfix[i]==0:
            dVar = Var(initialize=float(p0[i]),bounds=mbounds[i])
            setattr(model, pkey[i], dVar)
            del(dVar)
        else:
            dPar = Param(initialize=float(p0[i]))
            setattr(model, pkey[i], dPar)
            del(dPar)
    #Rate Constants and Variables (Parameters)
#    model.k1f=Var(initialize=k1f,bounds=(1.0e-7,1.0e0))					
#    model.k2=Var(initialize=k2,bounds=(1.0e-7,1.0e0))					
#    model.k3=Var(initialize=k3,bounds=(1.0e-7,1.0e-1))					
#    model.Platelet=Var(initialize=P0,bounds=(1.0,70))					
    
    # Concentration Variables, Define States
    # Platelets
    model.P = Var(model.t, within=NonNegativeReals, initialize=(value(model.P0)))
    model.dpdt = DerivativeVar(model.P,wrt=model.t, initialize=(0.0))
    
    # Activated Platelets
    model.Pa = Var(model.t, within=NonNegativeReals, initialize=(Pa0))
    model.dpadt = DerivativeVar(model.Pa,wrt=model.t, initialize=(0.0))
    
    
    # T: thrombus
    model.T = Var(model.t, within=NonNegativeReals, initialize=(T0))
    model.dTdt = DerivativeVar(model.T,wrt=model.t, initialize=(0.0))
    
    # Lysis
    model.L = Var(model.t, within=NonNegativeReals, initialize=(L0))
    model.dLdt =DerivativeVar(model.L,wrt=model.t, initialize=(0.0))
    
    #Initial Conditions
    def _init_conditions(model):
        yield model.P[0] == model.P0
        yield model.Pa[0] == 0.0
        yield model.T[0] == 0.0
        yield model.L[0] == 0.0
    model.init_conditions = ConstraintList(rule=_init_conditions)
    
    #Define Equations
    ## ODEs are defined in the return statement
    # p
    def dpdt_rule(m, t):
      return m.dpdt[t] ==  -m.k1f*(m.P[t])+k1b*m.Pa[t]
    model.dpdt_1_con = Constraint(model.t, rule=dpdt_rule)
    
    # pa
    def dpadt_rule(m, t):
      return m.dpadt[t] ==  m.k1f*(m.P[t])-k1b*m.Pa[t]-m.k2*(m.Pa[t]**n)
    model.dpadt_1_con = Constraint(model.t, rule=dpadt_rule)
    
    # T
    def dTdt_rule(m, t):
      return m.dTdt[t] ==  m.k2*(m.Pa[t]**n)-m.k3*m.T[t]
    model.dTdt_1_con = Constraint(model.t, rule=dTdt_rule)
    
    # L
    def dLdt_rule(m, t):
      return m.dLdt[t] ==  m.k3*m.T[t]
    model.dLdt_1_con = Constraint(model.t, rule=dLdt_rule)
    
    
    # define the objective function
    def _obj(m):
        SSE=sum([(data.Value.loc[t]-m.T[t])**2 for t in model.time])
        return SSE
    model.obj = Objective(sense=minimize,rule=_obj)
    
#    TFD=TransformationFactory("dae.finite_difference")
#    TFD.apply_to(model,nfe=len(model.t),wrt=model.t,scheme="BACKWARD")
#    # solve the problem
#    opt = SolverFactory('ipopt')
#    opt.options['linear_solver'] = "ma97"
#    opt.options['tol'] = 1e-6
#    #model.preprocess()
#    results = opt.solve(model, keepfiles=False, tee=False)
#    #model.load(results)
#    model.solutions.load_from(results)
    TFD=TransformationFactory("dae.finite_difference")
    TFD.apply_to(model,nfe=len(model.t),wrt=model.t,scheme="BACKWARD")
    # solve the problem
    opt = SolverFactory(solver)
    opt.options['linear_solver'] = "ma97"
    opt.options['tol'] = 1e-5
    opt.options['max_iter'] = 6000
    #model.preprocess()
    results = opt.solve(model, keepfiles=False, tee=False)
    #model.load(results)
    model.solutions.load_from(results)
    
#    discretize = Finite_Difference_Transformation()
#    disc = discretize.apply_to(model,nfe=nfinels,wrt=model.t,scheme='BACKWARD')
#    # You could also discretize using collocation on finite elements
#    #discretize2 = Collocation_Discretization_Transformation()
#    #disc = discretize2.apply(m,nfe=200,ncp=3,wrt=m.t)
#    
#    # Solve your model
#    opt=SolverFactory(solver)
#    
#    # Get the results of the solution to be returned to the get_CI() method
#    results = opt.solve(disc,tee=False)
   
    return model, results
    #disc.u.pprint()