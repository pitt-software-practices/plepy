"""
TODO: ADD FEATURE TO ENABLE USE OF UNBOUNDED VARIABLES
Note: This does not work with the current version of PLEpy, to be fixed
in future versions

Uses a calculated "cross-talk" matrix (converts 3D counts to 2D
activity for each 3D and 2D shell) to fit first-order rate coefficients
and initial activity in 3D shells using simulated 2D planar imaging
data. Each 3D shell only moves inward.

Model:
dA5/dt = -k5*A5
dA4/dt = k5*A5 - k4*A4
dA3/dt = k4*A4 - k3*A3
dA2/dt = k3*A3 - k2*A2
dA1/dt = k2*A2 - k1*A1

where k1-k5 are the rate coefficients and k1 > k2 > k3 > k4 > k5
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from pyomo.environ import *
from pyomo.dae import *

sys.path.append(os.path.abspath("../../"))
from plepy import PLEpy

pwd = os.getcwd()
fpath = os.path.dirname(__file__)
os.chdir(fpath)

# Import 2D data
ydata = np.load('toy2D_data_exp3.npz')['arr_0']
ytotal = ydata.sum(axis=1)
tdata = list(range(0, 81, 2))

# Import cross-talk matrix
crssfile = loadmat('shelltoy_crsstlk_dist.mat')
ctalk = crssfile['crsstlk']
ictalk = np.linalg.inv(ctalk)
iydata = np.dot(ictalk, ydata.T).T
iydata[1:, :] = (iydata[1:, :] + iydata[:-1, :])/2

# Actual data (for comparison)
datafile = loadmat('shelltoydata_exp3.mat')
data3d = datafile['a']

# Initial guesses
k0 = [5., 5., 1., 0.75, 0.5]    # [p1, p2, p3, p4, k5]
a0 = np.dot(ictalk, ydata[0, :].T) # [A1, A2, a3, A4, A5]'

da0dt = [k0[i+1]*a0[i+1] - k0[i]*a0[i] for i in range(4)]
da0dt.append(-k0[4]*a0[4])
da0dt = [1e-2*a for a in da0dt]

# Create dynamic model
model = ConcreteModel()

# Define parameters
model.t = ContinuousSet(bounds=(0, 81), initialize=range(81))
# Rate coefficients are fit as sum of previous rate coefficient and
# corresponding "p" parameter.
# k4 = k5 + p4, k3 = k4 + p3, etc.
model.p1 = Var(initialize=k0[0], bounds=(1e-3, 100.))
model.p2 = Var(initialize=k0[1], bounds=(1e-3, 100.))
model.p3 = Var(initialize=k0[2], bounds=(1e-3, 100.))
model.p4 = Var(initialize=k0[3], bounds=(1e-3, 100.))
model.k5 = Var(initialize=k0[4], bounds=(1e-3, 100.))

# Define 3D shell states
model.A1 = Var(model.t, initialize=a0[0], within=NonNegativeReals)
model.A2 = Var(model.t, initialize=a0[1], within=NonNegativeReals)
model.A3 = Var(model.t, initialize=a0[2], within=NonNegativeReals)
model.A4 = Var(model.t, initialize=a0[3], within=NonNegativeReals)
model.A5 = Var(model.t, initialize=a0[4], within=NonNegativeReals)

# Initialize derivatives
model.dA1dt = DerivativeVar(model.A1, wrt=model.t, initialize=da0dt[0])
model.dA2dt = DerivativeVar(model.A2, wrt=model.t, initialize=da0dt[1])
model.dA3dt = DerivativeVar(model.A3, wrt=model.t, initialize=da0dt[2])
model.dA4dt = DerivativeVar(model.A4, wrt=model.t, initialize=da0dt[3])
model.dA5dt = DerivativeVar(model.A5, wrt=model.t, initialize=da0dt[4])

# System dynamics
def _dA1dt(m, t):
    k4 = m.k5 + m.p4
    k3 = k4 + m.p3
    k2 = k3 + m.p2
    k1 = k2 + m.p1

    return m.dA1dt[t] == 1e-2*(k2*m.A2[t] - k1*m.A1[t])
model.dA1dt_ode = Constraint(model.t, rule=_dA1dt)

def _dA2dt(m, t):
    k4 = m.k5 + m.p4
    k3 = k4 + m.p3
    k2 = k3 + m.p2

    return m.dA1dt[t] == 1e-2*(k3*m.A3[t] - k2*m.A2[t])
model.dA2dt_ode = Constraint(model.t, rule=_dA2dt)

def _dA3dt(m, t):
    k4 = m.k5 + m.p4
    k3 = k4 + m.p3

    return m.dA3dt[t] == 1e-2*(k4*m.A4[t] - k3*m.A3[t])
model.dA3dt_ode = Constraint(model.t, rule=_dA3dt)

def _dA4dt(m, t):
    k4 = m.k5 + m.p4

    return m.dA4dt[t] == 1e-2*(m.k5*m.A5[t] - k4*m.A4[t])
model.dA4dt_ode = Constraint(model.t, rule=_dA4dt)

def _dA5dt(m, t):
    return m.dA5dt[t] == 1e-2*(- m.k5*m.A5[t])
model.dA5dt_ode = Constraint(model.t, rule=_dA5dt)

# Objective function (SSE)
def _obj(m):
    a3D = np.array([[m.A1[t], m.A2[t], m.A3[t], m.A4[t], m.A5[t]]
                    for t in tdata]).T
    a2D = np.dot(ctalk, a3D).T

    # err = (ydata - a2D)**2
    err = (iydata - a3D.T)**2
    return sum(sum(err))
model.obj = Objective(rule=_obj)

# Set-up solver
TFD=TransformationFactory("dae.finite_difference")
TFD.apply_to(model, nfe=2*len(model.t), wrt=model.t, scheme="BACKWARD")
solver = SolverFactory('ipopt')
solver.options['linear_solver'] = 'ma97'    # academic solver
solver.options['tol'] = 1e-6
solver.options['max_iter'] = 6000

results = solver.solve(model, keepfiles=False, tee=True)
model.solutions.load_from(results)

# Plot results
sns.set(context='talk')
plt.figure()
ccycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.plot(tdata, data3d[:, 0], ls='None', marker='o', color=ccycle[0])
plt.plot(tdata, data3d[:, 1], ls='None', marker='o', color=ccycle[1])
plt.plot(tdata, data3d[:, 2], ls='None', marker='o', color=ccycle[2])
plt.plot(tdata, data3d[:, 3], ls='None', marker='o', color=ccycle[3])
plt.plot(tdata, data3d[:, 4], ls='None', marker='o', color=ccycle[4])
# plt.plot(tdata, iydata[:, 0], label='Shell 1', color=ccycle[0])
# plt.plot(tdata, iydata[:, 1], label='Shell 2', color=ccycle[1])
# plt.plot(tdata, iydata[:, 2], label='Shell 3', color=ccycle[2])
# plt.plot(tdata, iydata[:, 3], label='Shell 4', color=ccycle[3])
# plt.plot(tdata, iydata[:, 4], label='Shell 5', color=ccycle[4])
plt.plot(model.t, model.A1[:](), label='Shell 1', color=ccycle[0])
plt.plot(model.t, model.A2[:](), label='Shell 2', color=ccycle[1])
plt.plot(model.t, model.A3[:](), label='Shell 3', color=ccycle[2])
plt.plot(model.t, model.A4[:](), label='Shell 4', color=ccycle[3])
plt.plot(model.t, model.A5[:](), label='Shell 5', color=ccycle[4])
plt.xlabel('Time (min)')
plt.ylabel('Activity (counts)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.tight_layout()
plt.show()

# Initialize PLEpy object
ps = [model.p1(), model.p2(), model.p3(), model.p4(), model.k5()]
ps.reverse()
ks = np.cumsum(ps)
A0s = [model.A1[0](), model.A2[0](), model.A3[0](), model.A4[0](),
       model.A5[0]()]

PLobj = PLEpy(model,
              ['p1', 'p2', 'p3', 'p4', 'k5', 'A1', 'A2', 'A3', 'A4', 'A5'],
              indices={'t0': [0]})
PLobj.set_index('A1', 't0')
PLobj.set_index('A2', 't0')
PLobj.set_index('A3', 't0')
PLobj.set_index('A4', 't0')
PLobj.set_index('A5', 't0')

# Get confidence limits using binary search (currently won't work
# because initial activity is unbounded)
PLobj.get_clims(['A1', 'A2', 'A3', 'A4', 'A5'])
# Generate profile likelihood curves
PLobj.get_PL(['A1', 'A2', 'A3', 'A4', 'A5'])
PLobj.plot_PL(pnames=['A1', 'A2', 'A3', 'A4', 'A5'], join=True, jmax=5)

os.chdir(pwd)
