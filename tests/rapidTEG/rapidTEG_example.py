import os
import sys

import numpy as np
import pandas as pd
from pyomo.dae import ContinuousSet, DerivativeVar
import pyomo.environ as penv

sys.path.append(os.path.abspath("../../"))
from plepy import PLEpy

def rapidTEG():
  pwd = os.getcwd()
  fpath = os.path.dirname(__file__)
  os.chdir(fpath)
  ##### Shout out to Michelle for lending me your Pyomo model :) #####

  # Set up data from file
  data = pd.read_csv('ExampleTEG.txt', delimiter='\t')
  data = data[["Time(sec)", "Value"]]
  data = data.dropna()
  data = data.set_index("Time(sec)") # indexing by time
  data = data/4 # get data into mm, then replicate top half of TEG
  os.chdir(pwd)

  # Data processing
  datadiff = np.diff(np.reshape(data.values, data.shape[0]))
  maxdiff = np.max(np.abs(datadiff))
  indexmax = np.argmax(np.abs(datadiff))*5

  if maxdiff >= 1.7 and indexmax >= 200:
      print(':(')
      dataold = data
      data = data[:(indexmax)-5]
  else:
      print(':)')

  #####################################################################
  # Model Jam
  # Reactions:
  # P <-> Pa
  # Pa -> T
  # T -> L
  # P: Platelets
  # Pa: Activated platelet
  # T: Thrombus
  # L: Lysis

  # Rate constants
  k1f = 5.0
  k2 = 4.0
  k3 = 4.0

  # Initial Conditions:
  Pa0 = 0.0
  T0 = 0.0
  L0 = 0.0
  P0 = np.max(data.Value)

  # Dependance on activated platelets
  n = 2

  # Create the model in Pyomo
  model = penv.ConcreteModel()
  model.t = ContinuousSet(initialize=data.index.values)
  model.time = penv.Set(initialize=data.index.values, within=model.t,
                        ordered=True)

  # Rate Constants and Variables (Parameters)
  model.k1f = penv.Var(initialize=k1f, bounds=(1.0e-4, 1.0e3))
  model.k2 = penv.Var(initialize=k2, bounds=(1.0e-5, 100.))
  model.k3 = penv.Var(initialize=k3, bounds=(1.0e-2, 1.0e5))
  model.Platelet = penv.Var(initialize=P0, bounds=(1.0, 70))


  # Concentration Variables, Define States
  # Platelets
  model.p = penv.Var(model.t, within=penv.NonNegativeReals,
                     initialize=(model.Platelet))
  model.dpdt = DerivativeVar(model.p, wrt=model.t, initialize=(0.0))

  # Activated Platelets
  model.pa = penv.Var(model.t, within=penv.NonNegativeReals, initialize=(Pa0))
  model.dpadt = DerivativeVar(model.pa, wrt=model.t, initialize=(0.0))


  # Thrombus
  model.T = penv.Var(model.t, within=penv.NonNegativeReals, initialize=(T0))
  model.dTdt = DerivativeVar(model.T, wrt=model.t, initialize=(0.0))

  # Lysis
  model.L = penv.Var(model.t, within=penv.NonNegativeReals, initialize=(L0))
  model.dLdt =DerivativeVar(model.L, wrt=model.t, initialize=(0.0))

  #Initial Conditions
  def _init_conditions(model):
      yield model.p[0] == model.Platelet
      yield model.pa[0] == 0.0
      yield model.T[0] == 0.0
      yield model.L[0] == 0.0
  model.init_conditions = penv.ConstraintList(rule=_init_conditions)

  #Define Equations
  # ODEs are defined in the return statement
  # p
  def dpdt_rule(m, t):
    return m.dpdt[t] ==  -m.k1f*(1e-3)*(m.p[t])
  model.dpdt_1_con = penv.Constraint(model.t, rule=dpdt_rule)

  # pa
  def dpadt_rule(m, t):
    return m.dpadt[t] ==  m.k1f*(1e-3)*(m.p[t]) - m.k2*(1e-2)*(m.pa[t]**n)
  model.dpadt_1_con = penv.Constraint(model.t, rule=dpadt_rule)

  # T
  def dTdt_rule(m, t):
    return m.dTdt[t] ==  m.k2*(1e-2)*(m.pa[t]**n) - m.k3*(1e-5)*m.T[t]
  model.dTdt_1_con = penv.Constraint(model.t, rule=dTdt_rule)

  # L
  def dLdt_rule(m, t):
    return m.dLdt[t] ==  m.k3*(1e-5)*m.T[t]
  model.dLdt_1_con = penv.Constraint(model.t, rule=dLdt_rule)


  # Define the objective function
  def obj(m):
      SSE = sum([(data.Value.loc[t] - m.T[t])**2 for t in model.time])
      return SSE
  model.obj = penv.Objective(rule= obj)
  # To use collocation method uncomment:
  # TFD = ransformationFactory("dae.collocation")
  # TFD.apply_to(model, nfe=len(model.t), cp=3, wrt=model.t,
  #              scheme="LAGRANGE-RADAU")

  # To use finite difference method uncomment:
  TFD = penv.TransformationFactory("dae.finite_difference")
  TFD.apply_to(model, nfe=len(model.t), wrt=model.t, scheme="BACKWARD")

  # Solve the problem
  opt = penv.SolverFactory('ipopt')
  opt.options['linear_solver'] = "ma97"
  opt.options['tol'] = 1e-5
  results = opt.solve(model, keepfiles=False, tee=False)
  model.solutions.load_from(results)

  #####################################################################

  # Create instance of PLEpy
  pl_inst = PLEpy(model, ['k1f', 'k2', 'k3', 'Platelet'])

  # Get profile likelihood estimates and (if they exist) confidence
  # intervals
  pl_inst.get_clims()
  pl_inst.get_PL()

  # Save results to JSON file
  # pl_inst.to_json('rapidTEG_solutions.json')

  # Load results from JSON file
  # pl_inst.load_json('rapidTEG_solutions.json')

  # Plot profile likelihood
  pl_inst.plot_PL(join=True)

if __name__ == '__main__':
    rapidTEG()