import numpy as np
import scipy as sp
from pyomo.environ import *
from pyomo.dae import *
from scipy.stats.distributions import chi2
from numpy import copy


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
        self.obj = m_obj    # original objective value
        pprofile = {p: self.m.find_component(p) for p in self.pnames}
        # list of Pyomo Variable objects to be profiled
        self.plist = pprofile
        # list of optimal parameter values
        self.popt = {p: value(self.plist) for p in self.pnames}
        pbounds = {p: self.plist[p].bounds for p in self.pnames}
        # list of parameter bounds
        self.pbounds = pbounds

    def step_CI(self, pname, pop=False, dr='up', stepfrac=0.01, alpha=0.05):
        # for stepping towards upper bound
        if dr == 'up':
            if self.pbounds[pname][1]:
                bound = self.pbounds[pname][1]
            else:
                bound = float('Inf')
            dB = 'UB'
            drer = 'upper'
            bdcrit = 'nextdr > bound'
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
            bdcrit = 'nextdr < bound'
            bd_eps = -1.0e-5

        states_dict = dict()
        _var_dict = dict()
        _obj_dict = dict()

        def_SF = copy(stepfrac)  # default stepfrac
        ctol = self.ctol
        _obj_CI = self.obj

        i = 0
        err = 0.0
        pstep = 0.0
        df = 1.0
        etol = chi2.isf(alpha, df)
        pardr = copy(self.popt)
        nextdr = self.popt[pname] - bd_eps
        bdreach = eval(bdcrit)

        while i < ctol and err <= etol and bdreach:
            pstep = pstep + stepfrac*self.popt[pname]    # stepsize
            pardr = self.popt[pname] + pstep     # take step
            self.plist[pname].set_value(pardr)
            iname = '_'.join([pname, 'inst_%s' % (dr), str(i)])
            # ^seems like kind of a long name
            try:
                riter = self.solver.solve(self.m)
                self.m.solutions.load_from(riter)
            except ValueError as e:
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
                return np.inf, states_dict, _var_dict, _obj_dict

            nextdr += pstep + stepfrac*self.popt[pname]
            bdreach = eval(bdcrit)
            if bdreach:
                print('Reached parameter %s bound!' % (drer))
                return pardr, states_dict, _var_dict, _obj_dict
            i += 1

    def get_CI(self, maxSteps=100, **kwds):

        # Get Confidence Intervals
        self.ctol = maxSteps

        states_dict = dict()

        parub = dict(self.popt)
        parlb = dict(self.popt)
        _var_dict = dict()
        _obj_dict = dict()

        _obj_CI = self.obj

        # Initialize parameters
        for pname in self.pnames:
            # manually change parameter of interest
            self.plist[pname].fix()

            # step to upper limit
            parub[pname], upstates, upvars, upobj = self.step_CI(
                pname, dr='up', **kwds
            )
            states_dict = {**states_dict, **upstates}
            _var_dict = {**_var_dict, **upvars}
            _obj_dict = {**_obj_dict, **upobj}

            # step to lower limit
            self.plist[pname].set_value(self.popt[pname])
            parlb[pname], dnstates, dnvars, dnobj = self.step_CI(
                pname, dr='down', **kwds
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
        
    def ebarplots(self,):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        nPars = len(self.popt)
        sns.set(style='whitegrid')
        plt.figure(figsize=(11,5))
        nrow = np.floor(nPars/3)
        ncol = np.ceil(nPars/nrow)
        for i in range(nPars):
            ax = plt.subplot(nrow, ncol, i+1)
            ax.bar(1, self.popt[i], 1, color='blue')
            pub = self.parub[i] - self.popt[i]
            plb = self.popt[i] - self.parlb[i]
            errs = [[plb], [pub]]
            ax.errorbar(x=1.5, y=self.popt[i], yerr=errs, color='black')
            plt.ylabel(self.pkey[i] + ' Value')
            plt.xlabel(self.pkey[i])
            
        plt.tight_layout()
        plt.show()
        
    def plot_PL(self,):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        nPars = len(self.pfix) - np.count_nonzero(self.pfix)
        sns.set(style='whitegrid')
        PL_fig = plt.figure(figsize=(11, 6))
        nrow = np.floor(nPars/3)
        if nrow < 1:
            nrow = 1
        ncol = np.ceil(nPars/nrow)
        # ndat, npat = np.shape(self.data)
        j=1
        for i, notfixed in enumerate(self.pfix):
            if notfixed == 0:
                dp = 0.0
                dob = 0.0
                k = 0
                PLub = []
                OBub = []
                PLlb = []
                OBlb = []
                while dp < self.parub[i] and k < self.ctol:
                    kname = '_'.join([self.pkey[i], 'inst_up', str(k)])
                    dp = self.var_dict[kname][i]
                    dob = log(self.obj_dict[kname])
                    PLub.append(dp)
                    OBub.append(2*dob)
                    k += 1
                k = 0
                while dp > self.parlb[i] and k < self.ctol:
                    kname = '_'.join([self.pkey[i], 'inst_down', str(k)])
                    dp = self.var_dict[kname][i]
                    dob = log(self.obj_dict[kname])
                    PLlb = np.append(dp, PLlb)
                    OBlb = np.append(2*dob, OBlb)
                    k += 1
            else:
                continue
            
            PL = np.append(PLlb, PLub)
            OB = np.append(OBlb, OBub)
            ax = plt.subplot(nrow, ncol, j)
            j += 1
            ax.plot(PL, OB)
            chibd = OBub[0]+chi2.isf(self.alpha, 1)
            ax.plot(PLub[0], OBub[0], marker='o')
            ax.plot([PLlb[0],PLub[-1]],[chibd,chibd])
            plt.xlabel(self.pkey[i]+' Value')
            plt.ylabel('Objective Value')
        plt.tight_layout()
        plt.show()
        return PL_fig
        
    def plot_trajectories(self, states):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        nrow = np.floor(len(states)/2)
        if nrow < 1:
            nrow = 1
        ncol = np.ceil(len(states)/nrow)
        sns.set(style='whitegrid')
        traj_Fig = plt.figure(figsize=(11, 10))
        for k in self.state_traj:
            j = 1
            for i in range(len(states)):
                ax = plt.subplot(nrow, ncol, j)
                j += 1
                ax.plot(self.times, self.state_traj[k][i])
                plt.title(states[i])
                plt.xlabel('Time')
                plt.ylabel(states[i] + ' Value')
        plt.tight_layout()
        plt.show()
        return traj_Fig
    
    def pop(self, pname, lb=True, ub=True):
        CI_dict = dict()
        for i in range(len(pname)):
            plb = self.parlb[i]
            pub = self.parub[i]
            CI_dict[pname[i]] = (plb,pub)
        return CI_dict