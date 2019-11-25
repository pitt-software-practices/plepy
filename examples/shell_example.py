# %% Problem Set-up
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
sys.path.append("../")
from scipy.io import loadmat
from tmpPLEpy import *
from pyomo.environ import *
from pyomo.dae import *

# Import data
tydata = pd.read_json('5ShellData.json')
tydata = tydata.sort_values('t')
tdata = tydata['t'] - 2
y0data = np.array(tydata.iloc[0:2].mean(axis=0).drop('t'))
ydata = np.array(tydata.drop('t', axis=1))
# C = np.load('fiveshell_Cmatrix.npz')['arr_0']
C = np.load('fiveshell_Cmatrix_take3.npy')
M = np.load('fiveshell_Mmatrix_take3.npy')
Cdiag = C + M
# x0 = np.flipud(np.load('fiveshell_x0.npy'))
x0 = np.load('fiveshell_x0_take3.npy')
V = np.load('fiveshell_V.npy').flatten()

# Initial parameter guesses
k0 = [100., 0.176, 0.176, 0.0645, 4.23e-4]
k0_ub = [100., 100., 100., 100., 0.00715]
k0_lb = [1e-4, 1e-4, 1e-4, 2.56e-3, 1e-4]
# k0[3:] = k0_ub[3:]
f_func = [0.420, 0.492, 0.351, 0.582, 1.]
# f_func = 5*[1.]
f_lb = [1., 1., 0.428, 0.256, 1.]
f_ub = [1., 1., 0.335, 0.256, 1.]
x0_func = np.array([x0[i]*f_func[i] for i in range(5)])

# Initial derivatives
A0 = np.zeros_like(C)
for i in range(4):
    A0[i, i] = -(1e-3)*k0[i]*V[i]
    if i != 0:
        A0[i-1, i] = (1e-3)*k0[i]*V[i]/V[i-1]
dxdt = np.dot(A0, x0_func)


# %% Create dynamic model
model = ConcreteModel()

# Define parameters/constants
model.t = ContinuousSet(bounds=(0, 81), initialize=range(81))
model.i = RangeSet(0, 4)
model.k = Var(model.i, bounds=(1e-4, 100.))
for i in model.i:
    model.k[i] = k0[i]
    # model.k[i].setlb(k0_lb[i])
    # model.k[i].setub(k0_ub[i])
    # if i != 0:
    #     model.k[i].fix()

# Define states
model.x_nf = Param(model.i, within=NonNegativeReals, mutable=True)
for i in model.i:
        model.x_nf[i] = x0[i][0] - x0_func[i][0]
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

# Increasing ki
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


def _obj(m):
    err = 0.
    for t in range(len(tdata)):
        x = np.array([[m.x_nf[i] + m.x_func[i, tdata[t]]] for i in m.i])
        yhat = np.dot(C, x)
        yobs = np.fliplr(np.array([ydata[t, :]])).T
        err += sum([(yhat[i][0] - yobs[i][0])**2 for i in model.i])
        # err += (yhat[0][0] - yobs[0][0])**2
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


PLobj = PLEpy(model, ['k'], indices={'i': [0, 1, 2, 3, 4]})
PLobj.set_index('k', 'i')
# PLobj.get_clims()
PLobj.clevel = 17.8
# PLobj.get_PL()
# PLobj.to_json('shell_example.json')
PLobj.load_json('shell_example.json')
figs, axs = PLobj.plot_PL(join=True, jmax=5)
# plot_PL(PLobj.PLdict, clevel=PLobj.clevel)
