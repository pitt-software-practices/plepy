import json
import copy

import numpy as np
import pyomo.environ as penv

from plepy.helper import recur_to_json


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
            {'index name': values}, 'index name' does not need to be
            the name of an index in the model, by default None
        solver : str, optional
            name of solver for Pyomo to use, by default 'ipopt'
        solver_kwds : dict, optional

        tee : bool, optional
            print Pyomo iterations at each step, by default False
        dae : discretization method for dae package, optional
            'finite_difference', 'collocation', or None, by default
            None
        dae_kwds : dict, optional
            keywords for dae package, by default {}
        presolve : bool, optional
            if True, model needs to be solved first, by default False
        """
        # Define solver & options
        solver_opts = {
            'linear_solver': 'ma27',
            'tol': 1e-6
        }
        solver_opts = {**solver_opts, **solver_kwds}
        opt = penv.SolverFactory(solver)
        opt.options = solver_opts
        self.solver = opt

        self.m = model
        # Discretize and solve model if necessary
        if dae and presolve:
            assert isinstance(dae, str)
            tfd = penv.TransformationFactory("dae." + dae)
            tfd.apply_to(self.m, **dae_kwds)
        if presolve:
            r = self.solver.solve(self.m)
            self.m.solutions.load_from(r)

        # Gather parameters to be profiled, their optimized values, and
        # bounds list of names of parameters to be profiled
        self.pnames = pnames
        self.indices = indices
        m_items = self.m.component_objects()
        obj_ls = list(filter(lambda x: isinstance(x, penv.Objective), m_items))
        m_obj = obj_ls[0]
        self.obj = penv.value(m_obj)    # original objective value
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
                self.popt[p] = penv.value(self.plist[p])
                # get parameter bounds
                self.pbounds[p] = self.plist[p].bounds

    def set_index(self, pname: str, *args):
        """Indicate which index to use for parameter, pname

        Args
        ----
        pname : str
            name of parameter to set index of
        *args : str
            string specifying the name of the index or indices to use
            for parameter
        """
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
            self.popt[pname][k] = penv.value(self.plist[pname][k])
            # get parameter bounds
            self.pbounds[pname][k] = self.plist[pname][k].bounds

    def getval(self, pname: str):
        if self.pindexed[pname]:
            return {k: penv.value(self.plist[pname][k])
                    for k in self.pidx[pname]}
        else:
            return penv.value(self.plist[pname])

    def setval(self, pname: str, val):
        if self.pindexed[pname]:
            self.plist[pname].set_values(val)
        else:
            self.plist[pname].set_value(val)

    def get_clevel(self, alpha: float=0.05):
        # determine confidence threshold value
        from scipy.stats.distributions import chi2
        etol = chi2.isf(alpha, df=1)
        clevel = etol/2 + np.log(self.obj)
        self.clevel = clevel
        return clevel

    def get_PL(self, pnames='all', n: int=20, min_step: float=1e-3,
               dtol: float=0.2, save: bool=False, fname='tmp_PLfile.json'):
        """Once bounds are found, calculate likelihood profiles for
        each parameter

        Args
        ----
        pnames: list or str
            name(s) of parameters to generate likelihood profiles for,
            or 'all' to generate profiles for all model parameters, by
            default 'all'

        Keywords
        --------
        n : int, optional
            minimum number of discretization points between optimum and
            each parameter bound, by default 20
        min_step : float, optional
            minimum allowable difference between two discretization
            points, by default 1e-3
        dtol : float, optional
            maximum error change between two points, by default 0.2
        save: bool, optional
            if True, will save results to a JSON file, by default False
        fname: str or path, optional
            location to save JSON file (if save=True),
            by default 'tmp_PLfile.json'
        """

        def inner_loop(xopt, xb, direct=1, idx=None) -> dict:
            from plepy.helper import sflag

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
                    xdict['obj'] = np.log(penv.value(self.m.obj))
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
            # calculate magnitude of objective value changes between
            # each step
            dy = np.abs(y_out - y_in)
            # pull indices where objective value change is greater than
            # threshold value (dtol) and step size is greater than
            # minimum
            ierr = [(i > dtol and j > min_step)
                             for i, j in zip(dy, dx)]
            print('ierr:', ierr)
            itr = 0
            # For intervals of large change (above dtol), calculate
            # values at midpoint. Repeat until no large changes or
            # minimum step size is reached.
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
                        xdict['obj'] = np.log(penv.value(self.m.obj))
                        # store values of other parameters at each pt
                        for p in self.pnames:
                            xdict[p] = self.getval(p)
                    except ValueError:
                        xdict = copy.deepcopy(pdict[x_ierr[w]])
                    pdict[x] = xdict
                # get parameter values needed to calculate change in
                # error over intervals that have not converged
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
                # calculate magnitude of objective value change between
                # each step
                dy = np.abs(y_out - y_in)
                # pull indices where objective value change is greater
                # than threshold value (dtol) and step size is greater
                # than minimum
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
            # make sure upper and lower confidence limits have been
            # specified or calculated using get_clims()
            emsg = ("Parameter confidence limits must be determined "
                    "prior to calculating likelihood profile.\nTry "
                    "running .get_clims() method first.")
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
        from plepy.helper import plot_PL

        assert isinstance(self.PLdict, dict)
        assert isinstance(self.clevel, float)
        jdict = recur_to_json(self.PLdict)
        figs, axs = plot_PL(jdict, self.clevel, **kwds)
        return figs, axs

    def m_eval(self, pname: str, pardr, idx=None, reset=True):
        # initialize all parameters at their optimal value (to ensure
        # feasibility)
        if reset:
            for p in self.pnames:
                self.setval(p, self.popt[p])
        # if parameter is indexed, set value of parameter at specified
        # index to pardr
        if idx is not None:
            self.plist[pname][idx].set_value(pardr)
        # if parameter is unindexed, set value of parameter to pardr
        else:
            self.plist[pname].set_value(pardr)
        # evalutate model at this point
        return self.solver.solve(self.m)

    def bsearch(self, pname: str, clevel: float, acc: float,
                direct: int=1, idx=None) -> float:
        """Binary search for confidence limit
        Args
        ----
        pname : str
            parameter name
        clevel: float
            value of log of objective function at confidence limit
        acc: float
            maximum fractional difference between binary search bounds
            allowed for convergence

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
        from plepy.helper import sflag

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
        x_range = x_high - x_low
        ctol = x_high*acc

        # Find outermost feasible value
        # evaluate at outer bound
        r_mid = self.m_eval(pname, x_mid, idx=idx)
        fcheck = sflag(r_mid)
        self.m.solutions.load_from(r_mid)
        err = np.log(penv.value(self.m.obj))
        # If solution is feasible and the error is less than the value
        # at the confidence limit, there is no CI in that direction.
        # Set to bound.
        if fcheck == 0 and err < clevel:
            pCI = no_lim
            print('No %s CI! Setting to %s bound.' % (plc, plc))
        else:
            fiter = 0
            # If solution is infeasible, find a new value for x_out
            # that is feasible and above the confidence threshold.
            while (fcheck == 1 or err < clevel) and x_range > ctol:
                print('f_iter: %i, x_high: %f, x_low: %f'
                        % (fiter, x_high, x_low))
                # check convergence criteria
                x_range = x_high - x_low
                ctol = x_high*acc
                # evaluate at midpoint
                x_mid = 0.5*(x_high + x_low)
                r_mid = self.m_eval(pname, x_mid, idx)
                fcheck = sflag(r_mid)
                # if infeasible, continue search inward from current
                # midpoint
                if fcheck == 1:
                    x_out = float(x_mid)
                self.m.solutions.load_from(r_mid)
                err = np.log(penv.value(self.m.obj))
                # if feasbile, but not over CL threshold, continue
                # search outward from current midpoint
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
            if x_range < ctol:
                pCI = no_lim
                print('No %s CI! Setting to %s bound.' % (plc, plc))
            # otherwise, find the upper CI between outermost feasible
            # pt and optimal solution using binary search
            else:
                x_out = float(x_mid)
                if direct:
                    x_high = x_out
                    x_low = x_in
                else:
                    x_high = x_in
                    x_low = x_out
                biter = 0
                # repeat until convergence criteria is met
                # (i.e. x_high = x_low)
                while x_range > ctol:
                    print('b_iter: %i, x_high: %f, x_low: %f'
                            % (biter, x_high, x_low))
                    # check convergence criteria
                    x_range = x_high - x_low
                    ctol = x_high*acc
                    # evaluate at midpoint
                    x_mid = 0.5*(x_high + x_low)
                    r_mid = self.m_eval(pname, x_mid, idx=idx)
                    fcheck = sflag(r_mid)
                    self.m.solutions.load_from(r_mid)
                    err = np.log(penv.value(self.m.obj))
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
                pCI = x_mid
                print('%s CI of %f found!' % (puc, pCI))
        # reset parameter
        self.setval(pname, self.popt[pname])
        if idx is None:
            self.plist[pname].free()
        else:
            self.plist[pname][idx].free()
        print(self.popt[pname])
        return pCI

    def get_clims(self, pnames='all', alpha: float=0.05, acc: float=0.001):
        """Get confidence limits of parameters
        Keywords
        --------
        pnames: list or str, optional
            name of parameter(s) to get confidence limits for, if 'all'
            will find limits for all parameters, by default 'all'
        alpha : float, optional
            confidence level, by default 0.05
        acc : float, optional
            maximum fractional difference between binary search bounds
            allowed for convergence, by default 0.001
        """
        if isinstance(pnames, str):
            if pnames == 'all':
                pnames = list(self.pnames)
            else:
                pnames = [pnames]

        # Define threshold of confidence level
        clevel = self.get_clevel(alpha)

        # create dictionaries for the confidence limits with the same
        # structure as self.popt
        parub = copy.deepcopy(dict(self.popt))
        parlb = copy.deepcopy(dict(self.popt))
        # Get upper & lower confidence limits
        for pname in pnames:
            # for indexed variables
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
            # for unindexed variables
            else:
                parlb[pname] = self.bsearch(pname, clevel, acc, direct=0)
                parub[pname] = self.bsearch(pname, clevel, acc, direct=1)
        self.parub = parub
        self.parlb = parlb

    def to_json(self, filename):
        # save existing attributes to JSON file
        atts = ['pnames', 'indices', 'obj', 'pindexed', 'pidx', 'popt',
                'pbounds', 'parlb', 'parub', 'clevel', 'PLdict']
        sv_dict = {}
        for att in atts:
            # if PLEpy attribute exists, convert it to a JSON
            # compatible form and add it to sv_dict
            try:
                sv_var = getattr(self, att)
                if isinstance(sv_var, dict):
                    sv_var = recur_to_json(sv_var)
                sv_dict[att] = sv_var
            except AttributeError:
                print("Attribute '%s' does not exist. Skipping." % (att))
        # write to JSON file
        with open(filename, 'w') as f:
            json.dump(sv_dict, f)

    def load_json(self, filename):
        from plepy.helper import recur_load_json
        # load PL data from a json file
        atts = ['pidx', 'popt', 'pbounds', 'parlb', 'parub', 'clevel',
                'PLdict']
        with open(filename, 'r') as f:
            sv_dict = json.load(f)
        for att in atts:
            # check for each PLEpy attribute and unserialize it from
            # JSON format
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
