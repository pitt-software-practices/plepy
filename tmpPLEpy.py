import json
import numbers
import numpy as np
import pandas as pd
from numpy import copy
from scipy.stats.distributions import chi2
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.dae import *
from pyomo.environ import *


class PLEpy:

    def __init__(self, model, pnames, solver='ipopt', solver_kwds={},
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

    def getval(self, pname):
        if self.pindexed[pname]:
            return self.plist[pname].get_values()
        else:
            return value(self.plist[pname])

    def setval(self, pname, val):
        if self.pindexed[pname]:
            self.plist[pname].set_values(val)
        else:
            self.plist[pname].set_value(val)

    def step_CI(self, pname, idx=None, pop=False, dr='up', stepfrac=0.01):
        # function for stepping in a single direction

        def pprint(pname, inum, ierr, ipval, istep, iflag=0, ifreq=20):
            # function for iteration printing
            dash = '='*80
            head = ' Iter. | Error | Par. Value | Stepsize | Par. Name | Flag'
            iform = (' {:^5d} | {:^5.3f} | {:>10.4g} | {:>8.3g} | {:<9s} |'
                     ' {:<27d}')
            iprint = iform.format(inum, ierr, ipval, istep, pname, iflag)
            if inum % ifreq == 0:
                print(*[dash, head, dash], sep='\n')
                print(iprint)
            else:
                print(iprint)

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

        def _multistep(pname, pardr, idx=None):

            def get_initial_guesses(pname, pardr, idx=None):
                import itertools as it

                # Create temporary multistart dict, replacing parameter being
                # profiled with value of next step
                tmpstart = dict(self.multistart)
                if idx is not None:
                    tmpstart[pname][idx] = [pardr]
                else:
                    tmpstart[pname] = [pardr]
                # Get product of initial guesses within indexed variables
                for p in self.pnames:
                    if self.pindexed[p]:
                        k0 = [tmpstart[p][i] for i in sorted(self.pidx[p])]
                        k = list(it.product(*k0))
                        tmpstart[p] = k
                # Get product of intial guesses
                k0 = [tmpstart[i] for i in sorted(self.pnames)]
                k = list(it.product(*k0))
                # Convert back to dictionary with each entry as a dictionary of
                # initial guesses for each parameter
                mstarts = {i: {x: y for x, y in zip(sorted(self.pnames), k[i])}
                           for i in range(len(k))}
                # Expand entries for indexed variables
                for p in self.pnames:
                    if self.pindexed[p]:
                        keys = sorted(self.pidx[p])
                        for i in list(mstarts.keys()):
                            m = {keys[j]: mstarts[i][p][j]
                                 for j in range(len(keys))}
                            mstarts[i][p] = m
                return mstarts

            if idx is not None:
                init_guesses = get_initial_guesses(pname, pardr, idx)
            else:
                init_guesses = get_initial_guesses(pname, pardr)

            if self.multi_opts['max_iter']:
                # Do multistart max iterations for each inital guess
                objN = {}
                if 'max_iter' in self.solver.options.keys():
                    orig_iters = int(self.solver.options['max_iter'])
                else:
                    orig_iters = 3000
                self.solver.options['max_iter'] = self.multi_opts['max_iter']
                for i in list(init_guesses.keys()):
                    ig = init_guesses[i]
                    for p in self.pnames:
                        self.setval(p, ig[p])
                    results = self.solver.solve(self.m)
                    self.m.solutions.load_from(results)
                    objN[i] = value(self.m.obj)
                sort_i = sorted(objN, key=lambda k: objN[k])

                # Continue solving best 3 candidates
                self.solver.options['max_iter'] = orig_iters
                results = []
                objs = []
                topN = sort_i[:self.multi_opts['best_n']]
                for i in topN:
                    ig = init_guesses[i]
                    for p in self.pnames:
                        self.setval(p, ig[p])
                    res_i = self.solver.solve(self.m)
                    results.append(res_i)
                    self.m.solutions.load_from(res_i)
                    objs.append(value(self.m.obj))
                results = sorted(results, key=lambda x: objs[results.index(x)])
            else:
                results = []
                objs = []
                for i in list(init_guesses.keys()):
                    ig = init_guesses[i]
                    for p in self.pnames:
                        self.setval(p, ig[p])
                    res_i = self.solver.solve(self.m)
                    results.append(res_i)
                    self.m.solutions.load_from(res_i)
                    objs.append(value(self.m.obj))
                results = sorted(results, key=lambda x: objs[results.index(x)])
            return results[0]

        def _singlestep(pname, pardr, idx=None):
            for p in self.pnames:
                self.setval(p, self.popt[p])
            if idx is not None:
                self.plist[pname][idx].set_value(pardr)
            else:
                self.plist[pname].set_value(pardr)

            results = self.solver.solve(self.m)
            return results

        # for stepping towards upper bound
        if dr == 'up':
            # for indexed variables...
            if idx is not None:
                if self.pbounds[pname][idx][1]:
                    bound = self.pbounds[pname][idx][1]
                else:
                    bound = float('Inf')
            # for unindexed variables
            elif self.pbounds[pname][1]:
                bound = self.pbounds[pname][1]
            else:
                bound = float('Inf')
            dB = 'UB'
            drer = 'upper'
            bd_eps = 1.0e-5

        # for stepping towards lower bound
        else:
            # for indexed variables...
            if idx is not None:
                if self.pbounds[pname][idx][0]:
                    bound = self.pbounds[pname][idx][0]
                else:
                    bound = 1e-10
            # for unindexed variables...
            elif self.pbounds[pname][0]:
                bound = self.pbounds[pname][0]
            else:
                bound = 1e-10
            dB = 'LB'
            drer = 'lower'
            stepfrac = -stepfrac
            bd_eps = -1.0e-5

        states_dict = {}    # currently does nothing
        vdict = {}  # dictionary for all parameter values at each step
        obj_dict = {}  # dictionary for objective values at each step
        flag_dict = {}  # dictionary for solver flag at each step

        def_SF = float(stepfrac)  # default stepfrac
        ctol = self.ctol    # max number of steps
        _obj_CI = value(self.obj)   # original objective value

        i = 0
        err = 0.0
        pstep = 0.0
        df = 1.0
        etol = chi2.isf(self.alpha, df)     # error tolerance for given alpha
        if idx is not None:
            popt = self.popt[pname][idx]    # parameter value at optimum
            prtname = '_'.join([pname, str(idx)])   # printed parameter name
        else:
            popt = self.popt[pname]
            prtname = str(pname)
        pardr = float(popt)
        nextdr = pardr - bd_eps
        if dr == 'up':
            bdreach = nextdr > bound
        else:
            bdreach = nextdr < bound

        while i < ctol and err <= etol and not bdreach:
            pstep = pstep + stepfrac*popt    # stepsize
            pardr = popt + pstep     # next parameter value
            iname = '_'.join([prtname, dr, str(i)])
            try:
                if self.multistart:
                    if idx is None:
                        riter = _multistep(pname, pardr)
                    else:
                        riter = _multistep(pname, pardr, idx=idx)
                else:
                    if idx is None:
                        riter = _singlestep(pname, pardr)
                    else:
                        riter = _singlestep(pname, pardr, idx=idx)
                self.m.solutions.load_from(riter)
                iflag = sflag(riter)

                err = 2*(np.log(value(self.m.obj)) - np.log(_obj_CI))
                vdict[iname] = {k: self.getval(k) for k in self.pnames}
                obj_dict[iname] = value(self.m.obj)
                flag_dict[iname] = iflag

                # adjust step size if convergence slow
                if i > 0:
                    prname = '_'.join([prtname, dr, str(i-1)])
                    d = np.abs((np.log(obj_dict[prname])
                                - np.log(obj_dict[iname])))
                    d /= np.abs(np.log(obj_dict[prname]))*stepfrac
                else:
                    d = err

                if d <= 0.01:  # if obj change too small, increase stepsize
                    stepfrac = 1.05*stepfrac
                else:
                    stepfrac = def_SF

                # print iteration info
                pprint(prtname, i, err, pardr, stepfrac*popt, iflag)
                if err > etol:
                    print('Reached %s CI!' % (drer))
                    print('{:s} = {:.4g}'.format(dB, pardr))
                    return pardr, states_dict, vdict, obj_dict, flag_dict
                elif i == ctol-1:
                    print('Maximum steps taken!')
                    if dr == 'up':
                        return np.inf, states_dict, vdict, obj_dict, flag_dict
                    else:
                        return -np.inf, states_dict, vdict, obj_dict, flag_dict

                nextdr += popt*stepfrac
                if dr == 'up':
                    bdreach = nextdr > bound
                else:
                    bdreach = nextdr < bound

                if bdreach:
                    print('Reached parameter %s bound!' % (drer))
                    print('{:s} = {:.4g}'.format(dB, pardr))
                    return pardr, states_dict, vdict, obj_dict, flag_dict
                i += 1
            except Exception as e:
                z = e
                print(z)
                if i > 0:
                    prname = '_'.join([prtname, dr, str(i-1)])
                    iname = '_'.join([prtname, dr, str(i)])
                    if idx is not None:
                        pardr = vdict[prname][pname][idx]
                    else:
                        pardr = vdict[prname][pname]
                    states_dict.pop(iname, None)
                    vdict.pop(iname, None)
                    obj_dict.pop(iname, None)
                else:
                    pardr = popt
                i = ctol
                print('Error occured!')
                print('{:s} set to {:.4g}'.format(dB, pardr))
                return pardr, states_dict, vdict, obj_dict, flag_dict

    def get_PL(self, n=20, min_step=1e-3, dtol=0.2):
    
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

        # Get upper & lower bounds for unindexed parameters
        for pname in filter(lambda x: not self.pindexed[x], self.pnames):
            # manually change parameter of interest
            self.plist[pname].fix()

            # search for upper bound
            print(' '*80)
            print('Parameter: {:s}'.format(pname))
            print('Bound: Upper')
            print(' '*80)
            if idx is None:
                pmax = self.pbounds[pname][1]
            else:
                pmax = self.pbounds[pname][idx][1]
            

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
