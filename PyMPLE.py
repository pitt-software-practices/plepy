import numpy as np
import scipy as sp
from pyomo.environ import *
from pyomo.dae import *
from scipy.stats.distributions import chi2
from numpy import copy
import json


class PyMPLE:

    def __init__(self, model, pnames, solver='ipopt', solver_kwds={},
                 tee=False, dae=None, dae_kwds={}, presolve=False):
        # Define solver & options
        solver_opts = {
            'linear_solver': 'ma97',
            'tol': 1e-6
        }
        solver_opts = {**solver_opts, **solver_kwds}
        opt = SolverFactory(solver)
        opt.options = solver_opts
        self.solver = opt

        self.m = model
        # Discretize and solve model if necessary
        if dae and presolve:
            if not isinstance(dae, str):
                raise TypeError
            tfd = TransformationFactory("dae." + dae)
            tfd.apply_to(self.m, **dae_kwds)
        if presolve:
            r = self.solver.solve(self.m)
            self.m.solutions.load_from(r)

        # Gather parameters to be profiled, their optimized values, and bounds
        # list of names of parameters to be profiled
        self.pnames = pnames
        m_items = self.m.component_objects()
        m_obj = list(filter(lambda x: isinstance(x, Objective), m_items))[0]
        self.obj = value(m_obj)    # original objective value
        pprofile = {p: self.m.find_component(p) for p in self.pnames}
        # list of Pyomo Variable objects to be profiled
        self.plist = pprofile
        # list of optimal parameter values
        self.popt = {p: value(self.plist[p]) for p in self.pnames}
        pbounds = {p: self.plist[p].bounds for p in self.pnames}
        # list of parameter bounds
        self.pbounds = pbounds

    def step_CI(self, pname, pop=False, dr='up', stepfrac=0.01):

        # for stepping towards upper bound
        if dr == 'up':
            if self.pbounds[pname][1]:
                bound = self.pbounds[pname][1]
            else:
                bound = float('Inf')
            dB = 'UB'
            drer = 'upper'
            bd_eps = 1.0e-5
        # for stepping towards lower bound
        else:
            if self.pbounds[pname][0]:
                bound = self.pbounds[pname][0]
            else:
                bound = 1e-10
            dB = 'LB'
            drer = 'lower'
            stepfrac = -stepfrac
            bd_eps = -1.0e-5

        states_dict = dict()
        _var_dict = dict()
        _obj_dict = dict()

        def_SF = float(stepfrac)  # default stepfrac
        ctol = self.ctol
        _obj_CI = value(self.obj)

        i = 0
        err = 0.0
        pstep = 0.0
        df = 1.0
        etol = chi2.isf(self.alpha, df)
        pardr = float(self.popt[pname])
        nextdr = self.popt[pname] - bd_eps
        if dr == 'up':
            bdreach = nextdr > bound
        else:
            bdreach = nextdr < bound

        while i < ctol and err <= etol and not bdreach:
            pstep = pstep + stepfrac*self.popt[pname]    # stepsize
            pardr = self.popt[pname] + pstep     # take step
            self.plist[pname].set_value(pardr)
            iname = '_'.join([pname, 'inst_%s' % (dr), str(i)])
            # ^seems like kind of a long name
            try:
                riter = self.solver.solve(self.m)
                self.m.solutions.load_from(riter)
            except Exception as e:
                z = e
                print(z)
                i = ctol
                continue
            """
            if pop:
                states_dict[iname] = [[value(getattr(iterinst, a)[b, c])
                                          for b in iterinst.t
                                          for c in iterinst.N]
                                         for a in self.states]
            else:
                states_dict[iname] = [[value(getattr(iterinst, a)[b])
                                          for b in iterinst.t]
                                         for a in self.states]
            """

            err = 2*(np.log(value(self.m.obj)) - np.log(_obj_CI))
            _var_dict[iname] = value(getattr(self.m, pname))
            _obj_dict[iname] = value(self.m.obj)

            # adjust step size if convergence slow
            if i > 0:
                prname = '_'.join([pname, 'inst_%s' % (dr), str(i-1)])
                d = np.abs((np.log(_obj_dict[prname])
                            - np.log(_obj_dict[iname])))
                d /= np.log(_obj_dict[prname])*stepfrac
            else:
                d = err

            if d <= 0.01:  # if obj change too small, increase stepsize
                pstr = ['Stepsize increased from', str(stepfrac), 'to',
                        str(1.05*stepfrac), 'with previous p value:',
                        str(pardr)]
                print(' '.join(pstr))
                stepfrac = 1.05*stepfrac
            else:
                stepfrac = def_SF

            # print iteration info
            pstr = ['finished %s iteration' % (dB), pname, str(i),
                    'with error:', str(err), 'and parameter change:',
                    str(pstep)]
            print(' '.join(pstr))
            if err > etol:
                print('Reached %s CI!' % (drer))
                return pardr, states_dict, _var_dict, _obj_dict
            elif i == ctol-1:
                print('Maximum steps taken!')
                if dr == 'up':
                    return np.inf, states_dict, _var_dict, _obj_dict
                else:
                    return -np.inf, states_dict, _var_dict, _obj_dict

            nextdr += pstep + stepfrac*self.popt[pname]
            if dr == 'up':
                bdreach = nextdr > bound
            else:
                bdreach = nextdr < bound
            if bdreach:
                print('Reached parameter %s bound!' % (drer))
                return pardr, states_dict, _var_dict, _obj_dict
            i += 1

    def get_CI(self, maxSteps=100, alpha=0.05, stepfrac=0.01):

        # Get Confidence Intervals
        self.ctol = maxSteps
        self.alpha = alpha

        states_dict = dict()

        parub = dict(self.popt)
        parlb = dict(self.popt)
        _var_dict = dict()
        _obj_dict = dict()

        _obj_CI = value(self.obj)

        # Initialize parameters
        for pname in self.pnames:
            # manually change parameter of interest
            self.plist[pname].fix()

            # step to upper limit
            parub[pname], upstates, upvars, upobj = self.step_CI(
                pname, dr='up', stepfrac=stepfrac
            )
            states_dict = {**states_dict, **upstates}
            _var_dict = {**_var_dict, **upvars}
            _obj_dict = {**_obj_dict, **upobj}

            # step to lower limit
            self.plist[pname].set_value(self.popt[pname])
            parlb[pname], dnstates, dnvars, dnobj = self.step_CI(
                pname, dr='down', stepfrac=stepfrac
            )
            states_dict = {**states_dict, **dnstates}
            _var_dict = {**_var_dict, **dnvars}
            _obj_dict = {**_obj_dict, **dnobj}
            
            # reset variable
            self.plist[pname].set_value(self.popt[pname])
            self.plist[pname].unfix()

        # assign profile likelihood bounds to PyMPLE object
        self.parub = parub
        self.parlb = parlb
        self.var_dict = _var_dict
        self.obj_dict = _obj_dict
        return {'Lower Bound': parlb, 'Upper Bound': parub}

    def ebarplots(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        nPars = len(self.pnames)
        sns.set(style='whitegrid')
        plt.figure(figsize=(11,5))
        nrow = np.floor(nPars/3)
        ncol = np.ceil(nPars/nrow)
        # for future look into making collection of PyMPLE objects ->
        # can put parameter bar plots from different subgroups on single plot
        for i, pname in enumerate(self.pnames):
            ax = plt.subplot(nrow, ncol, i+1)
            ax.bar(1, self.popt[pname], 1, color='blue')
            pub = self.parub[pname] - self.popt[pname]
            plb = self.popt[pname] - self.parlb[pname]
            errs = [[plb], [pub]]
            ax.errorbar(x=1.5, y=self.popt[pname], yerr=errs, color='black')
            plt.ylabel(pname + ' Value')
            plt.xlabel(pname)

        plt.tight_layout()
        plt.show()

    def plot_PL(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        nPars = len(self.pnames)
        sns.set(style='whitegrid')
        PL_fig = plt.figure(figsize=(11, 6))
        nrow = np.floor(nPars/3)
        if nrow < 1:
            nrow = 1
        ncol = np.ceil(nPars/nrow)

        for i, pname in enumerate(self.pnames):
            pkeys = sorted(filter(lambda x: x.split('_')[0] == pname,
                                  self.var_dict.keys()))
            pl = [self.var_dict[key] for key in pkeys]
            pl.append(self.popt[pname])
            ob = [np.log(self.obj_dict[key]) for key in pkeys]
            ob.append(np.log(self.obj))
            ob = [x for y, x in sorted(zip(pl, ob))]
            pl = sorted(pl)

            ax = plt.subplot(nrow, ncol, i+1)
            ax.plot(pl, ob)
            chibd = np.log(self.obj) + chi2.isf(self.alpha, 1)/2
            ax.plot(self.popt[pname], np.log(self.obj), marker='o')
            ax.plot([pl[0], pl[-1]], [chibd, chibd])
            plt.xlabel(pname +' Value')
            plt.ylabel('Objective Value')
        plt.tight_layout()
        plt.show()
        return PL_fig
        
    # def plot_trajectories(self, states):
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
        
    #     nrow = np.floor(len(states)/2)
    #     if nrow < 1:
    #         nrow = 1
    #     ncol = np.ceil(len(states)/nrow)
    #     sns.set(style='whitegrid')
    #     traj_Fig = plt.figure(figsize=(11, 10))
    #     for k in self.state_traj:
    #         j = 1
    #         for i in range(len(states)):
    #             ax = plt.subplot(nrow, ncol, j)
    #             j += 1
    #             ax.plot(self.times, self.state_traj[k][i])
    #             plt.title(states[i])
    #             plt.xlabel('Time')
    #             plt.ylabel(states[i] + ' Value')
    #     plt.tight_layout()
    #     plt.show()
    #     return traj_Fig
    
    # def pop(self, pname, lb=True, ub=True):
    #     CI_dict = dict()
    #     for i in range(len(pname)):
    #         plb = self.parlb[i]
    #         pub = self.parub[i]
    #         CI_dict[pname[i]] = (plb,pub)
    #     return CI_dict

    def to_json(self, filename):
        atts = ['alpha', 'parub', 'parlb', 'var_dict', 'obj_dict']
        sv_dict = {}
        for att in atts:
            sv_dict[att] = getattr(self, att)
        with open(filename, 'w') as f:
            json.dump(sv_dict, f)
    
    def load_json(self, filename):
        with open(filename, 'r') as f:
            sv_dict = json.load(f)
        for att in sv_dict.keys():
            setattr(self, att, sv_dict[att])