# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 13:36:36 2016

@author: Matthew Markovetz

This can remain umodified if you so choose. As long as you structure your model
module and run-file in the same format as provided in the example file, this
wonderful piece of code need never see the light of day.
"""

import numpy as np
import scipy as sp
from pyomo.environ import *
from scipy.stats.distributions import chi2
from numpy import copy


class PyMPLE:

    def __init__(self, params, model, objective, pnames, bounds, states,
                 hard=False):
        self.popt = params
        self.m = model
        self.obj = objective
        self.pkey = pnames
        self.bounds = bounds
        self.hard_bounds = hard
        self.states = states

    def step_CI(self, pfixCI, data, pindex, bound, pop=False, dr='up',
                stepfrac=0.01, solver='ipopt', nfe=150, alpha=0.05):
        # for stepping towards upper bound
        if dr == 'up':
            dB = 'UB'
            drer = 'upper'
            bdcrit = 'nextdr > bound'
            bd_eps = 1.0e-5
        # for stepping towards lower bound
        else:
            dB = 'LB'
            drer = 'lower'
            stepfrac = -stepfrac
            bdcrit = 'nextdr < bound'
            bd_eps = -1.0e-5

        states_dict = dict()
        _var_dict = dict()
        _obj_dict = dict()
        def_SF = copy(stepfrac)  # default stepfrac
        parkey = self.pkey
        ctol = self.ctol
        _obj_CI = self.obj
        j = pindex

        i = 0
        err = 0.0
        pstep = 0.0
        df = 1.0
        etol = chi2.isf(alpha, df)
        pardr = copy(self.popt)
        nextdr = self.popt[j] - bd_eps
        bdreach = eval(bdcrit)

        while i < ctol and err <= etol and bdreach:
            pstep = pstep + stepfrac[j]*self.popt[j]    # stepsize
            pardr[j] = self.popt[j] + pstep     # take step
            itername = '_'.join([parkey[j], 'inst_%s' % (dr), str(i)])
            # ^seems like kind of a long name
            try:
                iterinst, _ = self.m(data, solver, nfe, pfixCI, pardr,
                                     self.pkey, bound=self.bounds)
            except ValueError as e:
                z = e
                print(z)
                i = ctol
                continue
            if pop:
                states_dict[itername] = [[value(getattr(iterinst, a)[b, c])
                                          for b in iterinst.t
                                          for c in iterinst.N]
                                         for a in self.states]
            else:
                states_dict[itername] = [[value(getattr(iterinst, a)[b])
                                          for b in iterinst.t]
                                         for a in self.states]

            err = 2*(log(value(iterinst.obj)) - log(_obj_CI))
            _var_dict[itername] = [value(getattr(iterinst, parkey[a]))
                                   for a in range(len(parkey))]
            _obj_dict[itername] = value(iterinst.obj)

            # adjust step size if convergence slow
            if i > 0:
                prname = '_'.join([parkey[j], 'inst_%s' % (dr), str(i-1)])
                d = np.abs((log(_obj_dict[prname]) - log(_obj_dict[itername])))
                d = d/log(_obj_dict[prname])/stepfrac[j]
            else:
                d = err

            if d <= 0.01:  # if obj change too small, increase stepsize
                pstr = ['Stepsize increased from', str(stepfrac[j]), 'to',
                        str(1.05*stepfrac[j]), 'with previous p value:',
                        str(pardr[j])]
                print(' '.join(pstr))
                stepfrac[j] = 1.05*stepfrac[j]
            else:
                stepfrac[j] = stepfrac[j] + def_SF[j]

            # print iteration info
            pstr = ['finished %s iteration' % (dB), parkey[j], str(i),
                    'with error:', str(err), 'and parameter change:',
                    str(pstep)]
            print(' '.join(pstr))
            if err > etol:
                print('Reached %s CI!' % (drer))
                return pardr[j], states_dict, _var_dict, _obj_dict
            elif i == ctol-1:
                return np.inf, states_dict, _var_dict, _obj_dict

            nextdr = self.popt[j] + pstep + stepfrac[j]*self.popt[j]
            bdreach = eval(bdcrit)
            if bdreach:
                print('Reached parameter %s bound!' % (drer))
                return pardr[j], states_dict, _var_dict, _obj_dict
            i += 1

    def get_CI(self, pfixed, data, maxSteps=100, stepfrac=0.01, solver='ipopt',
               nfe=150, alpha=0.05):

        # Get Confidence Intervals
        ctol = maxSteps
        self.ctol = ctol

        states_dict = dict()

        parub = copy(self.popt)
        parlb = copy(self.popt)
        _var_dict = dict()
        _obj_dict = dict()

        def_SF = copy(stepfrac)
        pfixCI = copy(pfixed)

        _obj_CI = self.obj

        # Initialize parameters
        for j in range(len(pfixed)):
            if self.hard_bounds:
                lower_bound = self.bounds[j][0]
                upper_bound = self.bounds[j][1]
            else:
                lower_bound = 1e-10
                upper_bound = float('Inf')

            if pfixed[j] == 0:
                pfixCI[j] = 1   # manually change parameter of interest
                # step to upper limit
                parub[j], upstates, upvars, upobj = self.step_CI(
                    pfixCI, data, j, upper_bound, dr='up', stepfrac=stepfrac,
                    solver=solver, nfe=nfe, alpha=alpha
                )
                states_dict = {**states_dict, **upstates}
                _var_dict = {**_var_dict, **upvars}
                _obj_dict = {**_obj_dict, **upobj}
                # step to lower limit
                stepfrac[j] = def_SF[j] # reset stepfrac
                parlb[j], dnstates, dnvars, dnobj = self.step_CI(
                    pfixCI, data, j, lower_bound, dr='down', stepfrac=stepfrac,
                    solver=solver, nfe=nfe, alpha=alpha
                )
                states_dict = {**states_dict, **dnstates}
                _var_dict = {**_var_dict, **dnvars}
                _obj_dict = {**_obj_dict, **dnobj}

                pfixCI[j] = pfixed[j] # change back to variable
            else:
                continue
        iterinst, _ = self.m(data, solver, nfe, pfixed, self.popt, self.pkey,
                             bound=self.bounds)

        self.parub = parub
        self.parlb = parlb
        self.var_dict = _var_dict
        self.obj_dict = _obj_dict
        self.pfix = pfixed
        self.data = data
        self.alpha = alpha
        self.state_traj = states_dict
        self.times = iterinst.t
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