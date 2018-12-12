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

    def __init__(self, params, model, objective, pnames, bounds, states, hard=False):
        self.popt = params
        self.m = model
        self.obj = objective
        self.pkey = pnames
        self.bounds = bounds
        self.hard_bounds = hard
        self.states = states
    
    def get_CI(self, pfixed, data, maxSteps=100, stepfrac=0.01, solver='ipopt', nfe=150, alpha=0.05):
        
        # Get Confidence Intervals
        ctol = maxSteps
        self.ctol=ctol
        
        states_dict = dict()
#        resultses = dict()
        
        parup = copy(self.popt)
        pardn = copy(self.popt)
        parub = copy(self.popt)
        parlb = copy(self.popt)
        parkey = self.pkey
        upkey = []
        dnkey = []
        _var_dict = dict()
        _obj_dict = dict()
        
        def_SF = copy(stepfrac)
        pfixCI = copy(pfixed)
        
        _obj_CI = self.obj
        ndat, npat = np.shape(data)
        
        i=0
        err = 0.0
        pstep = 0.0
        bound_eps = 1.0e-5
        
        for j in range(len(pfixed)):
            if self.hard_bounds:
                lower_bound = self.bounds[j][0]
                upper_bound = self.bounds[j][1]
            else:
                lower_bound = 0.0
                upper_bound = float('Inf')
                        
            if pfixed[j]==0:    
                pfixCI[j] = 1
                df = 1.0#len(pfixed)-sum(pfixed)
                etol = chi2.isf(alpha,df)
                parup = copy(self.popt)
                next_up = self.popt[j] - bound_eps
                while i<ctol and err<=etol and next_up<upper_bound:
                    pstep = pstep + stepfrac[j]*self.popt[j]
                    parub[j] = self.popt[j] + pstep
                    parup[j] = self.popt[j] + pstep
                    
                    itername = '_'.join([parkey[j],'inst_up',str(i)])
                    try:
                        iterinst, iterres = self.m(data, solver, nfe, pfixed, pfixCI, parup, self.pkey,bound=self.bounds)
                    except ValueError as e:
                        z = e
                        print(z)
                        i = ctol
                        continue
                        
#                    instances[itername] = iterinst
#                    resultses[itername] = iterres
                    states_dict[itername] = [[value(getattr(iterinst,i)[j]) for j in iterinst.t] for i in self.states]
                    err = 2*(log(value(iterinst.obj)) - log(_obj_CI))#*2*ndat*npat
                    _var_dict[itername] = [value(getattr(iterinst,parkey[i])) for i in range(len(parkey))]
                    _obj_dict[itername] = value(iterinst.obj)
                    
                    #adjust step size if convergence slow
                    if i>0:
                        prevname = '_'.join([parkey[j],'inst_up',str(i-1)])
                        d = np.abs((log(_obj_dict[prevname]) - log(_obj_dict[itername])))/log(_obj_dict[prevname])/stepfrac[j]
                    else:
                        d = err
                        
                    if d<=0.01:
                        print(' '.join(['Stepsize increased from',str(stepfrac[j]),'to',str(1.05*stepfrac[j]), 'with previous p value: ',str(parup[j])]))
                        stepfrac[j] = 1.05*stepfrac[j]
                    else:
                        stepfrac[j] = stepfrac[j]+def_SF[j]
                    
                    print(' '.join(['finished UB iteration',parkey[j],str(i), 'with error: ',str(err), 'and parameter change: ', str(pstep)]))
                    if err>etol:
                        upkey.append(itername)
                        print('Reached upper CI!')
                    elif i==ctol-1:
                        parub[j] = np.inf
                        
                    next_up = self.popt[j] + pstep + stepfrac[j]*self.popt[j]
                    if next_up > upper_bound:
                        print('Reached parameter upper bound!')
                    i+=1
                i=0
                err=0.0
                d=0.0
                pstep=0.0
                pardn = copy(self.popt)
                stepfrac[j] = def_SF[j]
                dneps=1e-10
                next_down = self.popt[j] + bound_eps
                
                while i<ctol and err<=etol and next_down>lower_bound:                    
                    pstep = pstep - stepfrac[j]*self.popt[j]
                    parlb[j] = self.popt[j] + pstep
                    pardn[j] = self.popt[j] + pstep
                    
                    if pardn[j]<dneps:
                        pardn[j]=dneps
                        parlb[j]=dneps
                    itername = '_'.join([parkey[j],'inst_down',str(i)])
                    try:
                        iterinst, iterres = self.m(data, solver, nfe, pfixed, pfixCI, pardn, self.pkey,bound=self.bounds)
                    except ValueError as e:
                        z = e
                        print(z)
                        i = ctol
                        continue
#                    instances[itername] = iterinst
#                    resultses[itername] = iterres
                    states_dict[itername] = [[value(getattr(iterinst,i)[j]) for j in iterinst.t] for i in self.states]
                    err = 2*(log(value(iterinst.obj)) - log(_obj_CI))#*2*ndat*npat
                    _var_dict[itername] = [value(getattr(iterinst,parkey[i])) for i in range(len(parkey))]
                    _obj_dict[itername] = value(iterinst.obj)
                    
                    #adjust step size if convergence slow
                    if i>0:
                        prevname = '_'.join([parkey[j],'inst_down',str(i-1)])
                        d = np.abs((log(_obj_dict[prevname]) - log(_obj_dict[itername])))/log(_obj_dict[prevname])/stepfrac[j]
                    else:
                        d = err
                    
                    if d<=0.01:
                        print(' '.join(['Stepsize increased from',str(stepfrac[j]),'to',str(1.05*stepfrac[j]), 'with previous p value: ',str(pardn[j])]))
                        stepfrac[j] = 1.05*stepfrac[j]
                    else:
                        stepfrac[j] = stepfrac[j]+def_SF[j]
                        
                    print(' '.join(['finished LB iteration',parkey[j],str(i), 'with error: ',str(err), 'and parameter change: ', str(pstep)]))
                    if err>etol:
                        dnkey.append(itername)
                        print('Reached lower CI!')
                    elif i==ctol-1:
                        parlb[j] = -np.inf
                        
                    next_down = self.popt[j] + pstep - stepfrac[j]*self.popt[j]
                    if next_down < lower_bound:
                        print('Reached parameter lower bound!')
                    i+=1
                i=0
                err=0.0
                pstep=0.0
                pfixCI[j] = pfixed[j]
                parup = [value(getattr(iterinst,parkey[i])) for i in range(len(parkey))]
                pardn = [value(getattr(iterinst,parkey[i])) for i in range(len(parkey))]
            else:
                continue
            
        self.parub = parub
        self.parlb = parlb
        self.var_dict = _var_dict
        self.obj_dict = _obj_dict
        self.pfix = pfixed
        self.data = data
        self.alpha = alpha
        self.state_traj = states_dict
        self.times = iterinst.t
#        self.model_instances = instances
#        self.model_resultses = resultses
        return {'Lower Bound': parlb, 'Upper Bound': parub}
        
    def ebarplots(self,):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        nPars = len(self.popt)
        sns.set(style='whitegrid')
        plt.figure(figsize=(21,12))
        nrow = np.floor(nPars/3)
        ncol = np.ceil(nPars/nrow)
        for i in range(nPars):
            ax = plt.subplot(nrow,ncol, i+1)
            ax.bar(1,self.popt[i],1,color='blue')
            pub = self.parub[i]-self.popt[i]
            plb = self.popt[i]-self.parlb[i]
            errs = [[plb],[pub]]
            ax.errorbar(x=1.5,y=self.popt[i],yerr=errs,color='black')
            plt.ylabel(self.pkey[i]+' Value')
            plt.xlabel(self.pkey[i])
            
        plt.tight_layout()
        plt.show()
        
    def plot_PL(self,):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        nPars = len(self.pfix) - np.count_nonzero(self.pfix)
        sns.set(style='whitegrid')
        PL_fig = plt.figure(figsize=(21,12))
        nrow = np.floor(nPars/3)
        if nrow<1:
            nrow=1
        ncol = np.ceil(nPars/nrow)
        ndat, npat = np.shape(self.data)
        j=1
        for i,notfixed in enumerate(self.pfix):
            if notfixed==0:
                dp=0.0
                dob=0.0
                k=0
                PLub = []
                OBub = []
                PLlb = []
                OBlb = []
                while dp<self.parub[i] and k<self.ctol:
                    kname = '_'.join([self.pkey[i],'inst_up',str(k)])
                    dp = self.var_dict[kname][i]
                    dob= log(self.obj_dict[kname])
                    PLub.append(dp)
                    OBub.append(2*dob)
                    k=k+1
                k=0
                while dp>self.parlb[i] and k<self.ctol:
                    kname = '_'.join([self.pkey[i],'inst_down',str(k)])
                    dp = self.var_dict[kname][i]
                    dob= log(self.obj_dict[kname])
                    PLlb=np.append(dp,PLlb)
                    OBlb=np.append(2*dob,OBlb)
                    k=k+1
            else:
                continue
            
            PL = np.append(PLlb,PLub)
            OB = np.append(OBlb,OBub)
            ax = plt.subplot(nrow,ncol, j)
            j=j+1
            ax.plot(PL,OB)
            chibd = OBub[0]+chi2.isf(self.alpha,1)
            ax.plot(PLub[0],OBub[0],marker='o')
            ax.plot([PLlb[0],PLub[-1]],[chibd,chibd])
            plt.xlabel(self.pkey[i]+' Value')
            plt.ylabel('Objective Value')
#        plt.tight_layout()
        plt.show()
        return PL_fig
        
    def plot_trajectories(self,states):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        nrow = np.floor(len(states)/2)
        if nrow<1:
            nrow=1
        ncol = np.ceil(len(states)/nrow)
        sns.set(style='whitegrid')
        traj_Fig = plt.figure(figsize=(21,12))
        for k in self.state_traj:
            j=1
            for i in range(len(states)):
                ax = plt.subplot(nrow,ncol,j)
                j=j+1
                ax.plot(self.times,self.state_traj[k][i])
                plt.title(states[i])
                plt.xlabel('Time')
                plt.ylabel(states[i]+' Value')
#        plt.tight_layout()
        plt.show()
        return traj_Fig
    
    def pop(self, pname, lb=True, ub=True):
        CI_dict = dict()
        for i in range(len(pname)):
            plb = self.parlb[i]
            pub = self.parub[i]
            CI_dict[pname[i]] = (plb,pub)
        return CI_dict