import json
import numbers
import numpy as np
import pandas as pd
from numpy import copy
from scipy.stats.distributions import chi2
from sigfig import sigfig
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.dae import *
from pyomo.environ import *


def sflag(results):
            # determine solver status for iteration & assign flag
            stat = results.solver.status
            tcond = results.solver.termination_condition
            if ((stat == SolverStatus.ok) and
                    (tcond == TerminationCondition.optimal)):
                flag = 0
            elif (tcond == TerminationCondition.infeasible):
                flag = 1
            elif (tcond == TerminationCondition.maxIterations):
                flag = 2
            else:
                flag = 3
            return flag


class PLEpy:

    def __init__(self, model, pnames: list, solver='ipopt', solver_kwds={},
                 tee=False, dae=None, dae_kwds={}, presolve=False,
                 multistart=None, multi_kwds={}):
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
        # determine which variables are indexed
        self.pindexed = {p: self.plist[p].is_indexed() for p in self.pnames}
        # make empty dictionaries for optimal parameters and their bounds
        self.popt = {}
        self.pbounds = {}
        self.pidx = {}
        for p in self.pnames:
            # for indexed parameters...
            if self.pindexed[p]:
                # get index
                self.pidx[p] = list(self.plist[p].iterkeys())
                # get optimal solution
                self.popt[p] = self.plist[p].get_values()
                # get parameter bounds
                self.pbounds[p] = {k: self.plist[p][k].bounds
                                   for k in self.pidx[p]}
            # for unindexed parameters...
            else:
                # get optimal solution
                self.popt[p] = value(self.plist[p])
                # get parameter bounds
                self.pbounds[p] = self.plist[p].bounds
        # Create/validate multistart dictionary
        if multistart:
            for p in self.pnames:
                # Check if multistart specified for each parameter
                if p in sorted(multistart.keys()):
                    # Check specified correctly for indexed variables
                    if self.pindexed[p]:
                        pmulti = multistart[p]
                        if type(pmulti) == dict:
                            for idx in self.pidx[p]:
                                if idx in sorted(pmulti.keys()):
                                    if type(pmulti[idx]) == list:
                                        continue
                                    elif isinstance(pmulti, numbers.Number):
                                        pmulti[idx] = [pmulti[idx]]
                                    else:
                                        print('Warning! Invalid multistart!')
                                        pmulti[idx] = [self.popt[p][idx]]
                                else:
                                    pmulti[idx] = [self.popt[p][idx]]
                        else:
                            print('Warning! Invalid multistart!')
                            multistart[p] = {idx: [self.popt[p][idx]]
                                             for idx in self.pidx[p]}
                    # Check specified correctly for unindexed variables
                    else:
                        if type(multistart[p]) == list:
                            continue
                        elif isinstance(multistart[p], numbers.Number):
                            multistart[p] = [multistart[p]]
                        else:
                            print('Warning! Invalid multistart!')
                            multistart[p] = [self.popt[p]]
                # If not specified, or specified incorrectly, set values to
                # optimal solution
                else:
                    if self.pindexed[p]:
                        multistart[p] = {idx: [self.popt[p][idx]]
                                         for idx in self.pidx[p]}
                    else:
                        multistart[p] = [self.popt[p]]
            self.multistart = multistart
            self.multi_opts = {
                'max_iter': 50,
                'best_n': 3
            }
            self.multi_opts = {**self.multi_opts, **multi_kwds}
        else:
            self.multistart = False

    def getval(self, pname: str):
        if self.pindexed[pname]:
            return self.plist[pname].get_values()
        else:
            return value(self.plist[pname])

    def setval(self, pname: str, val):
        if self.pindexed[pname]:
            self.plist[pname].set_values(val)
        else:
            self.plist[pname].set_value(val)

    def get_PL(self, pnames='all', n: int=20, min_step: float=1e-3,
               dtol: float=0.2):
        """Once bounds are found, calculate likelihood profiles for each
        parameter

        Args
        ----
        pnames: list or str
            name(s) of parameters to generate likelihood profiles for, or 'all'
            to generate profiles for all model parameters, by default 'all'
        
        Keywords
        --------
        n : int, optional
            minimum number of discretization points between optimum and each
            parameter bound, by default 20
        min_step : float, optional
            minimum allowable difference between two discretization points,
            by default 1e-3
        dtol : float, optional
            maximum error change between two points, by default 0.2
        """

        def inner_loop(xopt, xb, direct=1) -> dict:
            pdict = {}
            if direct:
                print('Going up...')
                x0 = np.linspace(xopt, xb, n+2, endpoint=True)
            else:
                print('Going down...')
                x0 = np.linspace(xb, xopt, n+2, endpoint=True)
            print('x0:', x0)
            # evaluate objective at each discretization point
            for x in x0:
                xdict = {} 
                rx = self.m_eval(pname, x)
                xdict['flag'] = sflag(rx)
                self.m.solutions.load_from(rx)
                xdict['obj'] = np.log(value(self.m.obj))
                # store values of other parameters at each point
                for p in self.pnames:
                    xdict[p] = self.getval(p)
                pdict[x] = xdict
            if direct:
                x_out = x0[1:]
                x_in = x0[:-1]
            else:
                x_out = x0[:-1]
                x_in = x0[1:]
            # calculate magnitude of step sizes
            dx = x_out - x_in
            y0 = np.array([pdict[x]['obj'] for x in x0])
            print('y0:', y0)
            if direct:
                y_out = y0[1:]
                y_in = y0[:-1]
            else:
                y_out = y0[:-1]
                y_in = y0[1:]
            # calculate magnitude of objective value changes between each step
            dy = y_out - y_in
            # pull indices where objective value change is greater than
            # threshold value (dtol) and step size is greater than minimum
            ierr = [(i > dtol and j > min_step)
                             for i, j in zip(dy, dx)]
            print('ierr:', ierr)
            itr = 0
            # For intervals of large change (above dtol), calculate values at
            # midpoint. Repeat until no large changes or minimum step size
            # reached.
            while len(ierr) != 0:
                print('iter: %i' % (itr))
                x_oerr = np.array([j for i, j in zip(ierr, x_out) if i])
                x_ierr = np.array([j for i, j in zip(ierr, x_in) if i])
                x_mid = 0.5*(x_oerr + x_ierr)
                for x in x_mid:
                    xdict = {} 
                    rx = self.m_eval(pname, x)
                    xdict['flag'] = sflag(rx)
                    self.m.solutions.load_from(rx)
                    xdict['obj'] = np.log(value(self.m.obj))
                    for p in self.pnames:
                        xdict[p] = self.getval(p)
                    pdict[x] = xdict
                # get all parameter values involved in intervals of interest
                x0 = np.array(sorted(set([*x_oerr, *x_mid, *x_ierr])))
                print('x0:', x0)
                x_out = x0[1:]
                x_in = x0[:-1]
                # calculate magnitude of step sizes
                dx = x_out - x_in
                y0 = np.array([pdict[x]['obj'] for x in x0])
                print('y0:', y0)
                y_out = y0[1:]
                y_in = y0[:-1]
                # calculate magnitude of objective value change between each
                # step
                dy = y_out - y_in
                # pull indices where objective value change is greater than
                # threshold value (dtol) and step size is greater than minimum
                ierr = [(i > dtol and j > min_step)
                                 for i, j in zip(dy, dx)]
                print('ierr:', ierr)
                itr += 1
            return pdict

        if isinstance(pnames, str):
            if pnames == 'all':
                pnames = list(self.pnames)
            else:
                pnames = [pnames]

        # master dictionary for all parameter likelihood profiles
        PLdict = {}
        # generate profiles for parameters indicated
        for pname in pnames:
            print('Profiling %s...' % (pname))
            # make sure upper and lower confidence limits have been specified
            # or solved for using get_clims()
            emsg = ("Parameter confidence limits must be determined prior to "
                    "calculating likelihood profile.\nTry running "
                    ".get_clims() method first.")
            assert self.parlb[pname] is not None, emsg
            assert self.parub[pname] is not None, emsg

            self.plist[pname].fix()
            xopt = self.popt[pname]
            xlb = self.parlb[pname]
            xub = self.parub[pname]
            # discretize each half separately
            parPLup = inner_loop(xopt, xub, direct=1)
            parPLdn = inner_loop(xopt, xlb, direct=0)
            # combine results into parameter profile likelihood
            parPL = {**parPLup, **parPLdn}
            PLdict[pname] = parPL
            self.plist[pname].free()
        self.PLdict = PLdict

    def m_eval(self, pname: str, pardr, idx=None):
        # initialize all parameters at their optimal value (ensure feasibility)
        for p in self.pnames:
            self.setval(p, self.popt[p])
        # if parameter is indexed, set value of parameter at specified index
        # to pardr
        if idx is not None:
            self.plist[pname][idx].set_value(pardr)
        # if parameter is unindexed, set value of parameter to pardr
        else:
            self.plist[pname].set_value(pardr)
        # evalutate model at this point
        return self.solver.solve(self.m)

    def bsearch(self, pname: str, clevel: float, acc: int,
                direct: int=1) -> float:
        """Binary search for confidence limit
        Args
        ----
        pname : str
            parameter name
        clevel: float
            value of log of objective function at confidence limit
        acc: int
            accuracy in terms of the number of significant figures to consider

        Keywords
        --------
        direct : int, optional
            direction to search (0=downwards, 1=upwards), by default 1
        
        Returns
        -------
        float
            value of parameter bound
        """
        # manually change parameter of interest
        self.plist[pname].fix()

        # Initialize values based on direction
        x_out = self.pbounds[pname][direct]
        x_in = self.popt[pname]
        x_mid = x_out
        # for upper CI search
        if direct:
            x_high = x_out
            x_low = x_in
            plc = 'upper'
            puc = 'Upper'
            no_lim = self.pbounds[pname][1]
        # for lower CI search
        else:
            x_high = x_in
            x_low = x_out
            plc = 'lower'
            puc = 'Lower'
            no_lim = self.pbounds[pname][0]
        
        # Print search info
        print(' '*80)
        print('Parameter: {:s}'.format(pname))
        print('Bound: {:s}'.format(puc))
        print(' '*80)

        # check convergence criteria
        ctol = sigfig(x_high, acc) - sigfig(x_low, acc)

        # Find outermost feasible value
        # evaluate at outer bound
        r_mid = self.m_eval(pname, x_mid)
        fcheck = sflag(r_mid)
        self.m.solutions.load_from(r_mid)
        err = np.log(value(self.m.obj))
        # If solution is feasible and the error is less than the value at the
        # confidence limit, there is no CI in that direction. Set to bound.
        if fcheck == 0 and err < clevel:
            pCI = no_lim
            print('No %s CI! Setting to %s bound.' % (plc, plc))
        else:
            fiter = 0
            # If solution is infeasible, find a new value for x_out that is
            # feasible and above the confidence limit threshold.
            while (fcheck == 1 or err < clevel) and ctol > 0.0:
                print('f_iter: %i, x_high: %f, x_low: %f'
                        % (fiter, x_high, x_low))
                # check convergence criteria
                ctol = sigfig(x_high, acc) - sigfig(x_low, acc)
                # evaluate at midpoint
                x_mid = 0.5*(x_high + x_low)
                r_mid = self.m_eval(pname, x_mid)
                fcheck = sflag(r_mid)
                # if infeasible, continue search inward from current midpoint
                if fcheck == 1:
                    x_out = float(x_mid)
                self.m.solutions.load_from(r_mid)
                err = np.log(value(self.m.obj))
                # if feasbile, but not over CL threshold, continue search
                # outward from current midpoint
                if fcheck == 0 and err < clevel:
                    x_in = float(x_mid)
                if direct:
                    x_high = x_out
                    x_low = x_in
                else:
                    x_high = x_in
                    x_low = x_out
                fiter += 1
            # if convergence reached, there is no upper CI
            if ctol == 0.0:
                pCI = no_lim
                print('No %s CI! Setting to %s bound.' % (plc, plc))
            # otherwise, find the upper CI between outermost feasible pt and
            # optimal solution using binary search
            else:
                x_out = float(x_mid)
                if direct:
                    x_high = x_out
                    x_low = x_in
                else:
                    x_high = x_in
                    x_low = x_out
                biter = 0
                # repeat until convergence criteria is met (x_high = x_low)
                while ctol > 0.0:
                    print('b_iter: %i, x_high: %f, x_low: %f'
                            % (biter, x_high, x_low))
                    # check convergence criteria
                    ctol = sigfig(x_high, acc) - sigfig(x_low, acc)
                    # evaluate at midpoint
                    x_mid = 0.5*(x_high + x_low)
                    r_mid = self.m_eval(pname, x_mid)
                    fcheck = sflag(r_mid)
                    self.m.solutions.load_from(r_mid)
                    err = np.log(value(self.m.obj))
                    biter += 1
                    # if midpoint infeasible, continue search inward
                    if fcheck == 1:
                        x_out = float(x_mid)
                    # if midpoint over CL, continue search inward
                    elif err > clevel:
                        x_out = float(x_mid)
                    # if midpoint under CL, continue search outward
                    else:
                        x_in = float(x_mid)
                    if direct:
                        x_high = x_out
                        x_low = x_in
                    else:
                        x_high = x_in
                        x_low = x_out
                pCI = sigfig(x_mid, acc)
                print('%s CI of %f found!' % (puc, pCI))
        # reset parameter
        self.setval(pname, self.popt[pname])
        self.plist[pname].free()
        return pCI

    def get_clims(self, alpha=0.05, acc=3):
        """Get confidence limits of parameters
        Keywords
        --------
        alpha : float, optional
            confidence level, by default 0.05
        acc : int, optional
            accuracy in terms of significant figures, by default 3
        """
        # Define threshold of confidence level
        etol = chi2.isf(alpha, 1)
        obj0 = np.log(self.obj)
        # value to compare against in binary search
        clevel = etol/2 + obj0

        parub = dict(self.popt)
        parlb = dict(self.popt)
        # Get upper & lower confidence limits for unindexed parameters
        for pname in filter(lambda x: not self.pindexed[x], self.pnames):
            parlb[pname] = self.bsearch(pname, clevel, acc, direct=0)
            parub[pname] = self.bsearch(pname, clevel, acc, direct=1)
        # TODO: make compatible with indexed parameters
        self.parub = parub
        self.parlb = parlb
            

    def ebarplots(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        nPars = len(self.pnames)
        sns.set(style='whitegrid')
        plt.figure(figsize=(11,5))
        nrow = np.floor(nPars/3)
        ncol = np.ceil(nPars/nrow)
        # for future look into making collection of PLEpy objects ->
        # can put parameter bar plots from different subgroups on single plot
        for i, pname in enumerate(self.pnames):
            ax = plt.subplot(nrow, ncol, i+1)
            pub = self.parub[pname] - self.popt[pname]
            plb = self.popt[pname] - self.parlb[pname]
            errs = [[plb], [pub]]
            ax.bar(1, self.popt[pname], yerr=errs, tick_label=pname)
            plt.ylabel(pname + ' Value')

        plt.tight_layout()
        plt.show()

    def plot_simplePL(self, show=True, fname=None):
        # Plot just the PLs for each parameter
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = pd.DataFrame(self.var_dict).T
        for c in df.columns:
            if self.pindexed[c]:
                cols = sorted(['_'.join([c, str(i)]) for i in self.pidx[c]])
                df[cols] = df[c].apply(pd.Series).sort_index(axis=1)
                df = df.drop(c, axis=1)
        objs = pd.Series(self.obj_dict, name='objective').apply(np.log)

        nPars = len(df.columns)
        sns.set(style='whitegrid')
        PL_fig = plt.figure(figsize=(11, 6))
        nrow = np.floor(nPars/3)
        if nrow < 1:
            nrow = 1
        ncol = np.ceil(nPars/nrow)

        i = 1
        for pname in df.columns:
            if pname in self.pnames:
                pkeys = sorted(filter(lambda x: x.split('_')[0] == pname,
                                      df.index.values))
            else:
                pkeys = sorted(filter(lambda x: x.split('_')[:2] == 
                                      pname.split('_'), df.index.values))
            pdata = df.loc[pkeys]
            odata = objs.loc[pkeys]
            plt_df = pd.concat((pdata, odata), axis=1, sort=True)
            pdata = pdata.sort_values(pname)
            plt_df = plt_df.sort_values(pname)

            ax0 = plt.subplot(nrow, ncol, i)
            plt_df.plot(pname, 'objective', ax=ax0, legend=False)
            chibd = np.log(self.obj) + chi2.isf(self.alpha, 1)/2
            if pname in self.pnames:
                ax0.plot(self.popt[pname], np.log(self.obj), marker='o')
            else:
                nm, idx = pname.split('_')
                ax0.plot(self.popt[nm][int(idx)], np.log(self.obj), marker='o')
                # come back & change later
            ax0.plot([min(pdata[pname]), max(pdata[pname])], [chibd, chibd])
            plt.xlabel(pname + ' Value')
            plt.ylabel('Objective Value')
            i += 1
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(fname, dpi=600)
        return PL_fig

    def plot_dual(self, sep_index=True, show=True, fname=None):
        # This one works :)
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Get variable data in plotting format
        m = ['.', '^', 'x', 's']
        df = pd.DataFrame(self.var_dict).T
        fs = pd.Series(self.flag_dict, name='flag')
        objs = pd.Series(self.obj_dict, name='objective')
        plt_df = pd.concat((objs, fs), axis=1)
        for c in df.columns:
            if self.pindexed[c]:
                cols = sorted(['_'.join([c, str(i)]) for i in self.pidx[c]])
                df[cols] = df[c].apply(pd.Series).sort_index(axis=1)
                df = df.drop(c, axis=1)

        nPars = len(df.columns)
        sns.set(style='whitegrid')
        pal = sns.color_palette('cubehelix', nPars)
        colors = pal.as_hex()
        clr_dict = dict(zip(df.columns, colors))
        dual_fig = plt.figure(figsize=(11, 6))
        nrow = np.floor(nPars/3)*2
        if nrow < 1:
            nrow = 2
        ncol = np.ceil(2*nPars/nrow)

        i = 1
        for pname in df.columns:
            if pname in self.pnames:
                pkeys = sorted(filter(lambda x: x.split('_')[0] == pname,
                                      df.index.values))
            else:
                pkeys = sorted(filter(lambda x: x.split('_')[:2] == 
                                      pname.split('_'), df.index.values))
            pdata = df.loc[pkeys]
            pdata = pdata.sort_values(pname)
            minrow = pdata.loc[pname + '_up_0']
            ddf = pdata - minrow
            ddf[pname] = pdata[pname]
            pdata = pd.concat((pdata, plt_df.loc[pkeys]), axis=1, sort=True)
            pdata['ln_obj'] = pdata['objective'].apply(np.log)

            ax0 = plt.subplot(nrow, ncol, i)
            for j in range(4):
                ob_df = pdata[pdata.flag == j]
                try:
                    ob_df.plot(pname, 'ln_obj', ax=ax0, ls='None',
                               marker=m[j], color='k', legend=False)
                except TypeError:
                    pass
            chibd = np.log(self.obj) + chi2.isf(self.alpha, 1)/2
            if pname in self.pnames:
                ax0.plot(self.popt[pname], np.log(self.obj), marker='x',
                         markersize=14, color='b')
            else:
                nm, idx = pname.split('_')
                ax0.plot(self.popt[nm][int(idx)], np.log(self.obj), marker='x',
                         markersize=14, color='b') # come back & change later
            ax0.plot([min(pdata[pname]), max(pdata[pname])], [chibd, chibd],
                     color='r')
            plt.xlabel(pname + ' Value')
            plt.ylabel('Objective Value')

            # Plot parameter-parameter relationships
            ax1 = plt.subplot(nrow, ncol, i + ncol)
            cols = list(filter(lambda x: x != pname, df.columns))
            for j in range(4):
                par_df = ddf[pdata.flag == j]
                try:
                    if j == 0:
                        lgnd = True
                    else:
                        lgnd = False
                    par_df.plot(pname, cols, ax=ax1, sharex=ax0, ls='None',
                                marker=m[j],
                                color=[clr_dict.get(x, '#000000')
                                       for x in cols], legend=lgnd)
                except TypeError:
                    pass
            plt.xlabel(pname + ' Value')
            plt.ylabel('Parameter Change')
            if i % ncol:
                i += 1
            else:
                i += ncol + 1
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(fname, dpi=600, close=True)
        return dual_fig

    def to_json(self, filename):
        # save PL data to a json file
        atts = ['alpha', 'parub', 'parlb', 'var_dict', 'obj_dict', 'flag_dict']
        sv_dict = {}
        for att in atts:
            sv_var = getattr(self, att)
            if type(sv_var) == pd.DataFrame:
                sv_var = sv_var.to_dict()
            sv_dict[att] = sv_var
        with open(filename, 'w') as f:
            json.dump(sv_dict, f)

    def load_json(self, filename):
        # load PL data from a json file
        with open(filename, 'r') as f:
            sv_dict = json.load(f)
        for att in sv_dict.keys():
            # if att == 'var_df':
            #     sv_dict[att] = pd.DataFrame.from_dict(sv_dict[att])
            setattr(self, att, sv_dict[att])
