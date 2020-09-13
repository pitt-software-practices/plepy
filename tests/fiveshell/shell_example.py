# Problem Set-up
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath("../../"))
from scipy.io import loadmat
from plepy import PLEpy
from pyomo.environ import *
from pyomo.dae import *

pwd = os.getcwd()
fpath = os.path.dirname(__file__)
os.chdir(fpath)
## Import data
tydata = pd.read_json('5ShellData.json')
tydata = tydata.sort_values('t')
tdata = tydata['t'] - 2
y0data = np.array(tydata.iloc[0:2].mean(axis=0).drop('t'))
ydata = np.array(tydata.drop('t', axis=1))
C = np.load('fiveshell_Cmatrix_take3.npy')
M = np.load('fiveshell_Mmatrix_take3.npy')
Cdiag = C + M
x0 = np.load('fiveshell_x0_take3.npy')
V = np.load('fiveshell_V.npy').flatten()

## Initial parameter guesses
# rate coefficients
k0 = [100., 0.176, 0.176, 0.0645, 4.23e-4]
# fraction functional
f_func = [0.420, 0.492, 0.351, 0.582, 1.]
# initial counts in functional regions
x0_func = np.array([x0[i]*f_func[i] for i in range(5)])

## Initial derivatives
# dxdt = Ax
A0 = np.zeros_like(C)
for i in range(4):
    # rate out of i
    A0[i, i] = -(1e-3)*k0[i]*V[i]
    if i != 0:
        # rate in to i-1
        A0[i-1, i] = (1e-3)*k0[i]*V[i]/V[i-1]
dxdt = np.dot(A0, x0_func)


# Create dynamic model
model = ConcreteModel()

## Define parameters/constants
# time
model.t = ContinuousSet(bounds=(0, 81), initialize=range(81))
# shell
model.i = RangeSet(0, 4)
# rate coefficients
model.k = Var(model.i, bounds=(1e-5, 100.))
for i in model.i:
    model.k[i] = k0[i]

## Define states
# activity in non-functional region of lungs
model.x_nf = Param(model.i, within=NonNegativeReals, mutable=True)
for i in model.i:
        model.x_nf[i] = x0[i][0] - x0_func[i][0]
# activity in functional region of lungs
model.x_func = Var(model.i, model.t, within=NonNegativeReals)
for i in model.i:
        for t in model.t:
            model.x_func[i, t] = x0_func[i][0]

# Initialize derivatives
model.dxdt = DerivativeVar(model.x_func, wrt=model.t)
for i in model.i:
        for t in model.t:
            model.dxdt[i, t] = dxdt[i][0]

# Initial conditions
def _init_cond(m):
    for i in model.i:
        yield m.x_func[i, 0] == x0_func[i][0]
model.init_cond = ConstraintList(rule=_init_cond)

# Increasing ki constraint
def _incr_k(m):
    for i in range(4):
        yield m.k[i] >= m.k[i+1]
model.incr_k = ConstraintList(rule=_incr_k)

# System dynamics
def _dxdt(m, i, t):
    if i != 4:
        return m.dxdt[i, t] == (-1e-3*m.k[i]*V[i]*m.x_func[i, t]
                                + 1e-3*m.k[i+1]*V[i+1]/V[i]*m.x_func[i+1, t])
    else:
        return m.dxdt[i, t] == -1e-3*m.k[i]*V[i]*m.x_func[i, t]
model.dxdt_ode = Constraint(model.i, model.t, rule=_dxdt)


## Objective function
def _obj(m):
    err = 0.
    for t in range(len(tdata)):
        x = np.array([[m.x_nf[i] + m.x_func[i, tdata[t]]] for i in m.i])
        yhat = np.dot(C, x)
        yobs = np.fliplr(np.array([ydata[t, :]])).T
        err += sum([(yhat[i][0] - yobs[i][0])**2 for i in model.i])
    return err
model.obj = Objective(rule=_obj)

# Set-up solver
TFD=TransformationFactory("dae.finite_difference")
TFD.apply_to(model, nfe=2*len(model.t), wrt=model.t, scheme="BACKWARD")
solver = SolverFactory('ipopt')
solver.options['linear_solver'] = 'ma97'
solver.options['tol'] = 1e-6
solver.options['max_iter'] = 6000

results = solver.solve(model, keepfiles=False, tee=True)
model.solutions.load_from(results)

# Create PLEpy object
PLobj = PLEpy(model, ['k'], indices={'i': [0, 1, 2, 3, 4]})
PLobj.set_index('k', 'i')
PLobj.get_clims()
PLobj.get_PL()
PLobj.to_json('shell_example.json')
# PLobj.load_json('shell_example.json')

# Plot profile likelihood (and make pretty)
figs, axs = PLobj.plot_PL(join=True, jmax=5, disp='None')
for i, ax in enumerate(axs[0][1, :]):
    if i != 4:
        ax.set_xlim([None, 20])
    ax.xaxis.label.set_size(14)
    ax.xaxis.label.set_weight('bold')
axs[0][0, 0].yaxis.label.set_size(14)
axs[0][0, 0].yaxis.label.set_weight('bold')
axs[0][1, 0].yaxis.label.set_size(14)
axs[0][1, 0].yaxis.label.set_weight('bold')
figs[0].show()

os.chdir(pwd)
