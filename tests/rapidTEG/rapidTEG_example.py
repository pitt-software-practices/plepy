import numpy as np
import os
import pandas as pd
import sys
sys.path.append(os.path.abspath("../../"))
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.environ import *
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
  model = ConcreteModel()
  model.t = ContinuousSet(initialize=data.index.values)
  model.time = Set(initialize=data.index.values, within=model.t, ordered=True)

  # Rate Constants and Variables (Parameters)
  model.k1f = Var(initialize=k1f, bounds=(1.0e-4, 1.0e3))
  model.k2 = Var(initialize=k2, bounds=(1.0e-5, 100.))
  model.k3 = Var(initialize=k3, bounds=(1.0e-2, 1.0e5))
  model.Platelet = Var(initialize=P0, bounds=(1.0, 70))


  # Concentration Variables, Define States
  # Platelets
  model.p = Var(model.t, within=NonNegativeReals, initialize=(model.Platelet))
  model.dpdt = DerivativeVar(model.p, wrt=model.t, initialize=(0.0))

  # Activated Platelets
  model.pa = Var(model.t, within=NonNegativeReals, initialize=(Pa0))
  model.dpadt = DerivativeVar(model.pa, wrt=model.t, initialize=(0.0))


  # Thrombus
  model.T = Var(model.t, within=NonNegativeReals, initialize=(T0))
  model.dTdt = DerivativeVar(model.T, wrt=model.t, initialize=(0.0))

  # Lysis
  model.L = Var(model.t, within=NonNegativeReals, initialize=(L0))
  model.dLdt =DerivativeVar(model.L, wrt=model.t, initialize=(0.0))

  #Initial Conditions
  def _init_conditions(model):
      yield model.p[0] == model.Platelet
      yield model.pa[0] == 0.0
      yield model.T[0] == 0.0
      yield model.L[0] == 0.0
  model.init_conditions = ConstraintList(rule=_init_conditions)

  #Define Equations
  # ODEs are defined in the return statement
  # p
  def dpdt_rule(m, t):
    return m.dpdt[t] ==  -m.k1f*(1e-3)*(m.p[t])
  model.dpdt_1_con = Constraint(model.t, rule=dpdt_rule)

  # pa
  def dpadt_rule(m, t):
    return m.dpadt[t] ==  m.k1f*(1e-3)*(m.p[t]) - m.k2*(1e-2)*(m.pa[t]**n)
  model.dpadt_1_con = Constraint(model.t, rule=dpadt_rule)

  # T
  def dTdt_rule(m, t):
    return m.dTdt[t] ==  m.k2*(1e-2)*(m.pa[t]**n) - m.k3*(1e-5)*m.T[t]
  model.dTdt_1_con = Constraint(model.t, rule=dTdt_rule)

  # L
  def dLdt_rule(m, t):
    return m.dLdt[t] ==  m.k3*(1e-5)*m.T[t]
  model.dLdt_1_con = Constraint(model.t, rule=dLdt_rule)


  # Define the objective function
  def obj(m):
      SSE = sum([(data.Value.loc[t] - m.T[t])**2 for t in model.time])
      return SSE
  model.obj = Objective(rule= obj)
  # To use collocation method uncomment:
  # TFD = ransformationFactory("dae.collocation")
  # TFD.apply_to(model, nfe=len(model.t), cp=3, wrt=model.t,
  #              scheme="LAGRANGE-RADAU")

  # To use finite difference method uncomment:
  TFD = TransformationFactory("dae.finite_difference")
  TFD.apply_to(model, nfe=len(model.t), wrt=model.t, scheme="BACKWARD")

  # Solve the problem
  opt = SolverFactory('ipopt')
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