# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 13:26:43 2016

@author: Pressly
"""
from pyomo.environ import *
import math
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import interactive
interactive(True)
from pyomo.dae import *
import pandas as pd
from PyMPLE import PyMPLE
from time import time

#Set up data from file
data=pd.read_csv('ExampleTEG.txt',delimiter='\t') 
data=data[["Time(sec)","Value"]]
data=data.dropna()
data=data.set_index("Time(sec)") #indexing by time
data=data/4 # get data into mm, then replicate top half of TEG


datadiff=np.diff(np.reshape(data.values,data.shape[0])) #find the difference between each point
maxdiff=np.max(np.abs(datadiff)) #getting the largest jump between points (5 sec apart)
indexmax=np.argmax(np.abs(datadiff))*5 #getting the index of this max... multiplying by 5 to get into seconds

#Do you care about Max amplitude Michelle?
#check here then
#MA=np.max(data.values) #gives MA of the TEG
#rmaxMA=maxdiff/MA

#This version is aimed at removing discontinuieties
#so the maxdiff is here to say if there is a jump in mm greater than 2.5 then
#this data might be compromised. Furthermore, 200 index says if this jump
#happens before 200; then this is important information: 
#(found 200 by taking the max range of K at 138 and the max range of R 44 seconds.. added gives 182... rounded to 200)


plt.close("all")
if maxdiff >= 1.7 and indexmax >= 200:
    print(':(')
    dataold=data
    data=data[:(indexmax)-5]
    plt.figure(1)
    plt.plot(data,'k',dataold,'y--')  
else:
    print(':)')
    plt.figure(1)    
    plt.plot(data,'k')

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

#Dependance on activated platelets
n=2

t0 = time()
# create the model
model = ConcreteModel()
model.t=ContinuousSet(initialize=data.index.values)

#Rate Constants and Variables (Parameters)
model.k1f=Var(initialize=k1f,bounds=(1.0e-7,1.0e0))					
model.k2=Var(initialize=k2,bounds=(1.0e-7,1.0e0))					
model.k3=Var(initialize=k3,bounds=(1.0e-7,1.0e-1))					
model.Platelet=Var(initialize=P0,bounds=(1.0,70))					



		



# Concentration Variables, Define States
# Platelets
#model.plower = Param(initialize=0.0)
#def _ptest(model):
#    return None
#model.pupper = Param(initialize=model.Platelet*2,mutable=True)
##model.p = Var(model.t, bounds=(model.plower,model.pupper), initialize=(model.Platelet))
model.p = Var(model.t, within=NonNegativeReals, initialize=(model.Platelet))
model.dpdt = DerivativeVar(model.p,wrt=model.t, initialize=(0.0))

# Activated Platelets
#model.palower = Param(initialize=0.0)
#def _patest(model):
#    return None
#model.paupper = Param(initialize=model.Platelet*2,mutable=True)
##model.pa = Var(model.t, bounds=(model.palower,model.paupper), initialize=(Pa0))
model.pa = Var(model.t, within=NonNegativeReals, initialize=(Pa0))
model.dpadt = DerivativeVar(model.pa,wrt=model.t, initialize=(0.0))


# T: thrombus
#model.Tlower = Param(initialize=0.0)
#def _Ttest(model):
#    return None
#model.Tupper = Param(initialize=model.Platelet*2,mutable=True)
##model.T = Var(model.t, bounds=(model.Tlower,model.Tupper), initialize=(T0))
model.T = Var(model.t, within=NonNegativeReals, initialize=(T0))
model.dTdt = DerivativeVar(model.T,wrt=model.t, initialize=(0.0))

# Lysis
#model.Llower = Param(initialize=0.0)
#def _Ltest(model):
#    return None
#model.Lupper = Param(initialize=model.Platelet*2,mutable=True)
##model.L = Var(model.t, bounds=(model.Llower,model.Lupper), initialize=(L0))
model.L = Var(model.t, within=NonNegativeReals, initialize=(L0))
model.dLdt =DerivativeVar(model.L,wrt=model.t, initialize=(0.0))

#Initial Conditions
def _init_conditions(model):
    yield model.p[0] == model.Platelet
    yield model.pa[0] == 0.0
    yield model.T[0] == 0.0
    yield model.L[0] == 0.0
model.init_conditions = ConstraintList(rule=_init_conditions)

#Define Equations
## ODEs are defined in the return statement
# p
def dpdt_rule(m, t):
  return m.dpdt[t] ==  -m.k1f*(m.p[t])+k1b*m.pa[t]
model.dpdt_1_con = Constraint(model.t, rule=dpdt_rule)

# pa
def dpadt_rule(m, t):
  return m.dpadt[t] ==  m.k1f*(m.p[t])-k1b*m.pa[t]-m.k2*(m.pa[t]**n)
model.dpadt_1_con = Constraint(model.t, rule=dpadt_rule)

# T
def dTdt_rule(m, t):
  return m.dTdt[t] ==  m.k2*(m.pa[t]**n)-m.k3*m.T[t]
model.dTdt_1_con = Constraint(model.t, rule=dTdt_rule)

# L
def dLdt_rule(m, t):
  return m.dLdt[t] ==  m.k3*m.T[t]
model.dLdt_1_con = Constraint(model.t, rule=dLdt_rule)


# define the objective function
def obj(m):
    SSE=sum([(data.Value.loc[t]-m.T[t])**2 for t in data.index.values])
    return SSE

model.obj = Objective(rule= obj)
#TFD=TransformationFactory("dae.collocation")
#TFD.apply_to(model,nfe=len(model.t),cp=3,wrt=model.t,scheme="LAGRANGE-RADAU")
TFD=TransformationFactory("dae.finite_difference")
TFD.apply_to(model,nfe=len(model.t),wrt=model.t,scheme="BACKWARD")
# solve the problem
opt = SolverFactory('ipopt')
opt.options['linear_solver'] = "ma97"
opt.options['tol'] = 1e-6
#model.preprocess()
results = opt.solve(model, keepfiles=False, tee=True)
#model.load(results)
model.solutions.load_from(results)

t1 = time()
pl_inst = PyMPLE(model, ['k1f', 'k2', 'k3', 'Platelet'])

t2 = time()
pl_inst.load_json('pl_inst.json')
pl_inst.plot_PL()
# pl_inst.get_CI()
# t3 = time()
