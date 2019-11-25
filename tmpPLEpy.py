import json
import numbers
import numpy as np
import pandas as pd
import copy
from scipy.stats.distributions import chi2
from sigfig import sigfig
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.dae import *
from pyomo.environ import *


def recur_to_json(d: dict) -> dict:
            # recurssively convert dictionaries to compatible forms for JSON
            # serialization (keys must be strings)
            for key in list(d.keys()):
                if isinstance(d[key], dict):
                    d[key] = recur_to_json(d[key])
            d2 = {str(key): d[key] for key in list(d.keys())}
            return d2


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


def plot_PL(PLdict, clevel: float, pnames='all', covar='all', join: bool=False,
            jmax: int=4, disp: str='show', fprefix: str='tmp_fig', **dispkwds):
    """Plot likelihood profiles for specified parameters
    
    Args
    ----
    PLdict : dict
        profile likelihood data generated from PLEpy function, 'get_PL()', has
        format {'pname': {par_val: {keys: 'obj', 'par1', 'par2', etc.}}}.
    clevel: float
        value of objective at confidence threshold
    
    Keywords
    --------
    pnames : list or str, optional
        parameter(s) to generate plots for, if 'all' will plot for all keys in
        outer level of dictionary, by default 'all'
    covar : list or str, optional
        parameter(s) to include covariance plots for, if 'all' will include all
        keys in outer level of dictionary, by default 'all'
    join : bool, optional
        place multiple profile likelihood plots on a single figure, by default 
        False
    jmax : int, optional
        if join=True, the maximum number of plots to put in a single figure, by
        default 4
    disp: str, optional
        how to display generated figures, 'show' will run command plt.show(),
        'save' will save figures using filename prefix specified in fprefix,
        'None' will not display figures and simply return their handles, by
        default 'show'
    fprefix: str, optional
        filename prefix to give figures if disp='save', by default 'tmp_fig'
    **dispkwds: optional
        Keywords to pass to display function (either fig.show() or
        fig.savefig())
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cpal = sns.color_palette("deep")
    # If pnames or covar is a string, convert to appropriate list
    if isinstance(pnames, str):
        if pnames == 'all':
            pnames = list(PLdict.keys())
        else:
            pnames = [pnames]
    if isinstance(covar, str):
        if covar == 'all':
            plkeys = list(PLdict.keys())
            dict1 = PLdict[plkeys[0]]
            d1keys = list(dict1.keys())
            dict2 = dict1[d1keys[0]]
            d2keys = list(dict2.keys())
            if 'obj' in d2keys:
                covar = [k for k in d2keys if k not in ['obj', 'flag']]
            else:
                dict3 = dict2[d2keys[0]]
                d3keys = list(dict3.keys())
                covar = [k for k in d3keys if k not in ['obj', 'flag']]
        else:
            covar = [covar]

    # Determine which parameters (if any) are indexed
    pidx = {}
    dict1 = PLdict[pnames[0]]
    d1keys = list(dict1.keys())
    dict2 = dict1[d1keys[0]]
    d2keys = list(dict2.keys())
    if 'obj' in d2keys:
        for c in covar:
            if isinstance(dict2[c], dict):
                pidx[c] = list(dict2[c].keys())
    else:
        dict3 = dict2[d2keys[0]]
        for c in covar:
            if isinstance(dict3[c], dict):
                pidx[c] = list(dict3[c].keys())

    # Initialize counting scheme for tracking figures/subplots
    npars = len(pnames)
    ncovs = len(covar)
    if len(covar) > len(cpal):
        nreps = np.ceil(len(covar)/len(cpal))
        cpal = nreps*cpal
    cmap = {covar[i]: cpal[i] for i in range(len(covar))}
    for k in list(pidx.keys()):
        klen = len(pidx[k])
        if k in pnames:
            npars += (klen - 1)
        ncovs += (klen - 1)
        r0, g0, b0 = cmap[k]
        cmult = np.linspace(0.1, 1.5, num=klen, endpoint=True)
        cmap[k] = {pidx[k][i]: (min(cmult[i]*r0, 1), min(cmult[i]*g0, 1),
                                min(cmult[i]*b0, 1))
                   for i in range(klen)}
    assert npars != 0
    assert jmax > 0
    if join:
        nfig = int(np.ceil(npars/jmax))
        # if the number of profiled parameters is not divisible by the maximum
        # number of subplot columns (jmax), make the first figure generated
        # contain the remainder
        ncur = npars % jmax
        if not ncur:
            ncur = jmax
    else:
        nfig = npars
        ncur = 1
    # count number of parameters left to plot
    nleft = npars
    # index (b), parameter (c), and figure (d) counters
    b = 0
    c = 0
    d = 0
    figs = {}
    axs = {}
    while nleft > 0:
        print('d: %i' % (d))
        figs[d], axs[d] = plt.subplots(2, ncur, figsize=(3.5*ncur, 9),
                                       sharex='col', sharey='row')
        if ncur == 1:
            axs[d] = np.array([[axs[d][i]] for i in range(2)])
        for i in range(ncur):
            print('b: %i' % (b))
            print('c: %i' % (c))
            key = pnames[c]
            if key in list(pidx.keys()):
                idx = True
                ikey = pidx[key][b]
                x = sorted([float(j) for j in PLdict[key][ikey].keys()])
                xstr = [str(j) for j in x]
                y1 = [PLdict[key][ikey][j]['obj'] for j in xstr]
                b += 1
                if b == len(pidx[key]):
                    b = 0
            else:
                idx = False
                x = sorted([float(j) for j in PLdict[key].keys()])
                xstr = [str(j) for j in x]
                # plot objective value in first row
                y1 = [PLdict[key][j]['obj'] for j in xstr]
            axs[d][0, i].plot(x, y1, ls='None', marker='o')
            axs[d][0, i].plot(x, len(x)*[clevel], color='red')
            # plot other parameter values in second row
            if idx:
                for p in covar:
                    if p in list(pidx.keys()):
                        for a in pidx[p]:
                            if not (p == key and a == ikey):
                                lbl = ''.join([p, '[', str(a), ']'])
                                yi = [PLdict[key][ikey][j][p][a] for j in xstr]
                                axs[d][1, i].plot(x, yi, ls='None', marker='o',
                                                  label=lbl, color=cmap[p][a])
                    else:
                        yi = [PLdict[key][ikey][j][p] for j in xstr]
                        axs[d][1, i].plot(x, yi, ls='None', marker='o',
                                          label=p, color=cmap[p])
                klbl = ''.join([key, '[', str(ikey), ']'])
                axs[d][1, i].set_xlabel(klbl)
            else:
                for p in [p for p in covar if p != key]:
                    if p in list(pidx.keys()):
                        for a in pidx[p]:
                            lbl = ''.join([p, '[', str(a), ']'])
                            yi = [PLdict[key][j][p][a] for j in xstr]
                            axs[d][1, i].plot(x, yi, ls='None', marker='o',
                                              label=lbl, color=cmap[p][a])
                    else:
                        yi = [PLdict[key][j][p] for j in xstr]
                        axs[d][1, i].plot(x, yi, ls='None', marker='o',
                                          label=p, color=cmap[p])
                axs[d][1, i].set_xlabel(key)
            axs[d][1, i].legend(loc='best')
            if b == 0:
                c += 1
        axs[d][0, 0].set_ylabel('Objective Value')
        axs[d][1, 0].set_ylabel('Parameter Values')
        sns.despine(figs[d])
        figs[d].tight_layout()
        # check how many parameters are left to plot
        nleft = nleft - ncur
        # since we already plotted the remainder parameters, nleft should be
        # divisible by jmax now
        if join:
            ncur = jmax
        d += 1
    # display generated plots and/or return their handles
    if disp == 'show':
        for i in range(nfig):
            figs[i].show(**dispkwds)
        return figs, axs
    elif disp == 'save':
        for i in range(nfig):
            figs[i].savefig('_'.join([fprefix, str(i)]), **dispkwds)
        return figs, axs
    else:
        return figs, axs


class PLEpy:

    def __init__(self, model, pnames: list, indices=None, solver='ipopt',
                 solver_kwds={}, tee=False, dae=None, dae_kwds={},
                 presolve=False):
        """Profile Likelihood Estimator object
        
        Args
        ----
        model : Pyomo model
        pnames : list
            names of estimated parameters in model
        
        Keywords
        --------
        indices : dict, optional
            dictionary of indices for estimated parameters of format:
            {'index name': values}, 'index name' does not need to be the name
            of an index in the model, by default None
        solver : str, optional
            name of solver for Pyomo to use, by default 'ipopt'
        solver_kwds : dict, optional

        tee : bool, optional
            print Pyomo iterations at each step, by default False
        dae : discretization method for dae package, optional
            'finite_difference', 'collocation', or None, by default None
        dae_kwds : dict, optional
            keywords for dae package, by default {}
        presolve : bool, optional
            if True, model needs to be solved first, by default False
        """
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
            assert isinstance(dae, str)
            tfd = TransformationFactory("dae." + dae)
            tfd.apply_to(self.m, **dae_kwds)
        if presolve:
            r = self.solver.solve(self.m)
            self.m.solutions.load_from(r)

        # Gather parameters to be profiled, their optimized values, and bounds
        # list of names of parameters to be profiled
        self.pnames = pnames
        self.indices = indices
        m_items = self.m.component_objects()
        m_obj = list(filter(lambda x: isinstance(x, Objective), m_items))[0]
        self.obj = value(m_obj)    # original objective value
        pprofile = {p: self.m.find_component(p) for p in self.pnames}
        # list of Pyomo Variable objects to be profiled
        self.plist = pprofile
        # determine which variables are indexed
        self.pindexed = {p: self.plist[p].is_indexed() for p in self.pnames}
        # make empty dictionaries for optimal parameters and their bounds
        self.pidx = {}
        self.popt = {}
        self.pbounds = {}
        for p in self.pnames:
            # for indexed parameters...
            if not self.pindexed[p]:
                # get optimal solution
                self.popt[p] = value(self.plist[p])
                # get parameter bounds
                self.pbounds[p] = self.plist[p].bounds

    def set_index(self, pname: str, *args):
        import itertools as it

        assert self.pindexed[pname]
        for arg in args:
            assert arg in self.indices.keys()
        # get list of indices in same order as *args
        pindex = list(it.product(*[self.indices[arg] for arg in args]))
        self.pidx[pname] = pindex
        self.popt[pname] = {}
        self.pbounds[pname] = {}
        for k in pindex:
            # get optimal solutions
            self.popt[pname][k] = value(self.plist[pname][k])
            # get parameter bounds
            self.pbounds[pname][k] = self.plist[pname][k].bounds

    def getval(self, pname: str):
        if self.pindexed[pname]:
            return {k: value(self.plist[pname][k]) for k in self.pidx[pname]}
        else:
            return value(self.plist[pname])

    def setval(self, pname: str, val):
        if self.pindexed[pname]:
            self.plist[pname].set_values(val)
        else:
            self.plist[pname].set_value(val)

    def get_PL(self, pnames='all', n: int=20, min_step: float=1e-3,
               dtol: float=0.2, save: bool=False, fname='tmp_PLfile.json'):
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
        save: bool, optional
            if True, will save results to a JSON file, by default False
        fname: str or path, optional
            location to save JSON file (if save=True),
            by default 'tmp_PLfile.json'
        """

        def inner_loop(xopt, xb, direct=1, idx=None) -> dict:
            pdict = {}
            if direct:
                print('Going up...')
                x0 = np.linspace(xopt, xb, n+2, endpoint=True)
            else:
                print('Going down...')
                x0 = np.linspace(xb, xopt, n+2, endpoint=True)
            print('x0:', x0)
            # evaluate objective at each discretization point
            for w, x in enumerate(x0):
                xdict = {}
                if w == 0:
                    for p in self.pnames:
                        self.setval(p, self.popt[p])
                else:
                    for p in self.pnames:
                        prevx = pdict[x0[w-1]][p]
                        self.setval(p, prevx)
                try:
                    rx = self.m_eval(pname, x, idx=idx, reset=False)
                    xdict['flag'] = sflag(rx)
                    self.m.solutions.load_from(rx)
                    xdict['obj'] = np.log(value(self.m.obj))
                    # store values of other parameters at each point
                    for p in self.pnames:
                        xdict[p] = self.getval(p)
                except ValueError:
                    xdict = copy.deepcopy(pdict[x0[w-1]])
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
            dy = np.abs(y_out - y_in)
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
                for w, x in enumerate(x_mid):
                    xdict = {}
                    for p in self.pnames:
                        prevx = pdict[x_ierr[w]][p]
                        self.setval(p, prevx)
                    try:
                        rx = self.m_eval(pname, x, idx=idx, reset=False)
                        xdict['flag'] = sflag(rx)
                        self.m.solutions.load_from(rx)
                        xdict['obj'] = np.log(value(self.m.obj))
                        # store values of other parameters at each point
                        for p in self.pnames:
                            xdict[p] = self.getval(p)
                    except ValueError:
                        xdict = copy.deepcopy(pdict[x_ierr[w]])
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
                dy = np.abs(y_out - y_in)
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

            if self.pindexed[pname]:
                parPL = {}
                for k in self.pidx[pname]:
                    self.plist[pname][k].fix()
                    xopt = self.popt[pname][k]
                    xlb = self.parlb[pname][k]
                    xub = self.parub[pname][k]
                    print('xopt: ', xopt, 'xlb: ', xlb, 'xub: ', xub)
                    kPLup = inner_loop(xopt, xub, direct=1, idx=k)
                    kPLdn = inner_loop(xopt, xlb, direct=0, idx=k)
                    kPL = {**kPLup, **kPLdn}
                    parPL[k] = kPL
                    self.plist[pname][k].free()
                PLdict[pname] = parPL
            else:
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
        if save:
            jdict = recur_to_json(PLdict)
            with open(fname, 'w') as f:
                json.dump(jdict, f)

    def plot_PL(self, **kwds):
        assert isinstance(self.PLdict, dict)
        assert isinstance(self.clevel, float)
        jdict = recur_to_json(self.PLdict)
        figs, axs = plot_PL(jdict, self.clevel, **kwds)
        return figs, axs

    def m_eval(self, pname: str, pardr, idx=None, reset=True):
        # initialize all parameters at their optimal value (ensure feasibility)
        if reset:
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
                direct: int=1, idx=None) -> float:
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
        idx: optional
            for indexed parameters, the value of the index to get the
            confidence limits for
        
        Returns
        -------
        float
            value of parameter bound
        """
        # manually change parameter of interest
        if idx is None:
            self.plist[pname].fix()
            x_out = float(self.pbounds[pname][direct])
            x_in = float(self.popt[pname])
        else:
            self.plist[pname][idx].fix()
            x_out = float(self.pbounds[pname][idx][direct])
            x_in = float(self.popt[pname][idx])

        # Initialize values based on direction
        x_mid = x_out
        # for upper CI search
        if direct:
            x_high = x_out
            x_low = x_in
            plc = 'upper'
            puc = 'Upper'
            no_lim = float(x_out)
        # for lower CI search
        else:
            x_high = x_in
            x_low = x_out
            plc = 'lower'
            puc = 'Lower'
            no_lim = float(x_out)
        
        # Print search info
        print(' '*80)
        print('Parameter: {:s}'.format(pname))
        if idx is not None:
            print('Index: {:s}'.format(str(idx)))
        print('Bound: {:s}'.format(puc))
        print(' '*80)

        # check convergence criteria
        ctol = sigfig(x_high, acc) - sigfig(x_low, acc)

        # Find outermost feasible value
        # evaluate at outer bound
        r_mid = self.m_eval(pname, x_mid, idx=idx)
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
                r_mid = self.m_eval(pname, x_mid, idx)
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
                    r_mid = self.m_eval(pname, x_mid, idx=idx)
                    fcheck = sflag(r_mid)
                    self.m.solutions.load_from(r_mid)
                    err = np.log(value(self.m.obj))
                    print(self.popt[pname])
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
        if idx is None:
            self.plist[pname].free()
        else:
            self.plist[pname][idx].free()
        print(self.popt[pname])
        return pCI

    def get_clims(self, pnames='all', alpha: float=0.05, acc: int=3):
        """Get confidence limits of parameters
        Keywords
        --------
        pnames: list or str, optional
            name of parameter(s) to get confidence limits for, if 'all' will
            find limits for all parameters, by default 'all'
        alpha : float, optional
            confidence level, by default 0.05
        acc : int, optional
            accuracy in terms of significant figures, by default 3
        """
        if isinstance(pnames, str):
            if pnames == 'all':
                pnames = list(self.pnames)
            else:
                pnames = [pnames]

        # Define threshold of confidence level
        etol = chi2.isf(alpha, 1)
        obj0 = np.log(self.obj)
        # value to compare against in binary search
        clevel = etol/2 + obj0

        parub = copy.deepcopy(dict(self.popt))
        parlb = copy.deepcopy(dict(self.popt))
        # Get upper & lower confidence limits for unindexed parameters
        for pname in pnames:
            if self.pindexed[pname]:
                f = 0
                print(f)
                for idx in self.pidx[pname]:
                    parlb[pname][idx] = self.bsearch(pname, clevel, acc,
                                                     direct=0, idx=idx)
                    print(parlb)
                    parub[pname][idx] = self.bsearch(pname, clevel, acc,
                                                     direct=1, idx=idx)
                    print(parub)
                    print(self.popt)
                    f += 1
            else:
                parlb[pname] = self.bsearch(pname, clevel, acc, direct=0)
                parub[pname] = self.bsearch(pname, clevel, acc, direct=1)
        self.clevel = clevel
        self.parub = parub
        self.parlb = parlb

    def to_json(self, filename):

        # save existing attributes
        atts = ['pnames', 'indices', 'obj', 'pindexed', 'pidx', 'popt',
                'pbounds', 'parlb', 'parub', 'clevel', 'PLdict']
        sv_dict = {}
        for att in atts:
            try:
                sv_var = getattr(self, att)
                if isinstance(sv_var, dict):
                    sv_var = recur_to_json(sv_var)
                sv_dict[att] = sv_var
            except AttributeError:
                print("Attribute '%s' does not exist. Skipping." % (att))
        with open(filename, 'w') as f:
            json.dump(sv_dict, f)

    def load_json(self, filename):

        def recur_load_json(d: dict) -> dict:
            from ast import literal_eval
            d2 = {}
            for key in list(d.keys()):
                if isinstance(d[key], dict):
                    d[key] = recur_load_json(d[key])
                try:
                    lkey = literal_eval(key)
                except ValueError:
                    lkey = key
                d2[lkey] = d[key]
            return d2

        # load PL data from a json file
        atts = ['pidx', 'popt', 'pbounds', 'parlb', 'parub', 'clevel',
                'PLdict']
        with open(filename, 'r') as f:
            sv_dict = json.load(f)
        for att in atts:
            try:
                sv_var = sv_dict[att]
                if att == 'pidx':
                    sv_var = {k: [tuple(i) for i in sv_var[k]]
                              for k in sv_var.keys()}
                elif att == 'clevel':
                    pass
                else:
                    sv_var = recur_load_json(sv_var)
                setattr(self, att, sv_var)
            except KeyError:
                print("Attribute '%s' not yet defined." % (att))
