import json
import numpy as np
import pandas as pd
from numpy import copy
from scipy.stats.distributions import chi2
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.dae import *
from pyomo.environ import *


class PLEpy:

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
        _obj_dict = {}  # dictionary for objective values at each step
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
            # reset params to optimal solutions
            for p in self.pnames:
                self.setval(p, self.popt[p])
            # set value of parameter to be profiled
            if idx is not None:
                self.plist[pname][idx].set_value(pardr)
            else:
                self.plist[pname].set_value(pardr)
            iname = '_'.join([prtname, dr, str(i)])
            try:
                riter = self.solver.solve(self.m)
                self.m.solutions.load_from(riter)
                iflag = sflag(riter)

                err = 2*(np.log(value(self.m.obj)) - np.log(_obj_CI))
                vdict[iname] = {k: self.getval(k) for k in self.pnames}
                _obj_dict[iname] = value(self.m.obj)
                flag_dict[iname] = iflag

                # adjust step size if convergence slow
                if i > 0:
                    prname = '_'.join([prtname, dr, str(i-1)])
                    d = np.abs((np.log(_obj_dict[prname])
                                - np.log(_obj_dict[iname])))
                    d /= np.abs(np.log(_obj_dict[prname]))*stepfrac
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
                    # v_dict = pd.DataFrame.from_dict(vdict, orient='index')
                    return pardr, states_dict, vdict, _obj_dict, flag_dict
                elif i == ctol-1:
                    print('Maximum steps taken!')
                    if dr == 'up':
                        # v_dict = pd.DataFrame.from_dict(vdict, orient='index')
                        return np.inf, states_dict, vdict, _obj_dict, flag_dict
                    else:
                        # v_dict = pd.DataFrame.from_dict(vdict, orient='index')
                        return -np.inf, states_dict, vdict, _obj_dict, flag_dict

                nextdr += popt*stepfrac
                if dr == 'up':
                    bdreach = nextdr > bound
                else:
                    bdreach = nextdr < bound

                if bdreach:
                    print('Reached parameter %s bound!' % (drer))
                    print('{:s} = {:.4g}'.format(dB, pardr))
                    # v_dict = pd.DataFrame.from_dict(vdict, orient='index')
                    return pardr, states_dict, vdict, _obj_dict, flag_dict
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
                    _obj_dict.pop(iname, None)
                else:
                    pardr = popt
                i = ctol
                print('Error occured!')
                print('{:s} set to {:.4g}'.format(dB, pardr))
                # v_dict = pd.DataFrame.from_dict(vdict, orient='index')
                return pardr, states_dict, vdict, _obj_dict, flag_dict

    def get_CI(self, maxSteps=100, alpha=0.05, stepfrac=0.01):

        # Get Confidence Intervals
        self.ctol = maxSteps
        self.alpha = alpha

        states_dict = dict()    # currently does nothing

        parub = dict(self.popt)
        parlb = dict(self.popt)
        # _var_df = pd.DataFrame(columns=self.pnames)
        _var_df = dict()
        _obj_dict = dict()
        _flag_dict = dict()

        _obj_CI = value(self.obj)

        # Get CIs for unindexed parameters
        for pname in filter(lambda x: not self.pindexed[x], self.pnames):
            # manually change parameter of interest
            self.plist[pname].fix()

            # step to upper limit
            print(' '*80)
            print('Parameter: {:s}'.format(pname))
            print('Direction: Upward')
            print('Bound: {:<.3g}'.format(self.pbounds[pname][1]))
            print(' '*80)
            parub[pname], upstates, upvars, upobj, upflag = self.step_CI(
                pname, dr='up', stepfrac=stepfrac
            )
            states_dict = {**states_dict, **upstates}
            # _var_df = pd.concat([_var_df, upvars])
            _var_df = {**_var_df, **upvars}
            _obj_dict = {**_obj_dict, **upobj}
            _flag_dict = {**_flag_dict, **upflag}

            # step to lower limit
            print(' '*80)
            print('Parameter: {:s}'.format(pname))
            print('Direction: Downward')
            print('Bound: {:<.3g}'.format(self.pbounds[pname][0]))
            print(' '*80)
            self.plist[pname].set_value(self.popt[pname])
            parlb[pname], dnstates, dnvars, dnobj, dnflag = self.step_CI(
                pname, dr='down', stepfrac=stepfrac
            )
            states_dict = {**states_dict, **dnstates}
            # _var_df = pd.concat([_var_df, dnvars])
            _var_df = {**_var_df, **dnvars}
            _obj_dict = {**_obj_dict, **dnobj}
            _flag_dict = {**_flag_dict, **dnflag}
            
            # reset variable
            self.plist[pname].set_value(self.popt[pname])
            self.plist[pname].unfix()

        # Get CIs for indexed parameters
        for pname in filter(lambda x: self.pindexed[x], self.pnames):
            # manually change parameter of interest
            for idx in self.pidx[pname]:
                self.plist[pname][idx].fix()
                parub[pname] = {}
                parlb[pname] = {}

                # step to upper limit
                print(' '*80)
                print('Parameter: {:s}'.format(pname))
                print('Index: {:d}'.format(idx))
                print('Direction: Upward')
                print('Bound: {:<.3g}'.format(self.pbounds[pname][idx][1]))
                print(' '*80)
                parub[pname][idx], upstates, upvars, upobj, upflag = self.step_CI(
                    pname, idx=idx, dr='up', stepfrac=stepfrac
                )
                states_dict = {**states_dict, **upstates}
                # _var_df = pd.concat([_var_df, upvars])
                _var_df = {**_var_df, **upvars}
                _obj_dict = {**_obj_dict, **upobj}
                _flag_dict = {**_flag_dict, **upflag}

                # step to lower limit
                print(' '*80)
                print('Parameter: {:s}'.format(pname))
                print('Index: {:d}'.format(idx))
                print('Direction: Downward')
                print('Bound: {:<.3g}'.format(self.pbounds[pname][idx][0]))
                print(' '*80)
                self.setval(pname, self.popt[pname])
                parlb[pname][idx], dnstates, dnvars, dnobj, dnflag = self.step_CI(
                    pname, idx=idx, dr='down', stepfrac=stepfrac
                )
                states_dict = {**states_dict, **dnstates}
                # _var_df = pd.concat([_var_df, dnvars])
                _var_df = {**_var_df, **dnvars}
                _obj_dict = {**_obj_dict, **dnobj}
                _flag_dict = {**_flag_dict, **dnflag}

                # reset variable
                self.setval(pname, self.popt[pname])
                self.plist[pname][idx].unfix()

        # assign profile likelihood bounds to PLEpy object
        self.parub = parub
        self.parlb = parlb
        self.var_df = _var_df
        self.obj_dict = _obj_dict
        self.flag_dict = _flag_dict
        return {'Lower Bound': parlb, 'Upper Bound': parub}

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

    def plot_PL(self, show=True, fname=None):
        # Experimental plotting - doesn't currently work
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import seaborn as sns

        nPars = len(self.pnames)
        nrow = nPars
        ncol = nPars
        sns.set(style='darkgrid', rc={'axes.facecolor': '#757575'})
        PLfig, axs = plt.subplots(nrows=nrow, ncols=ncol, sharey=True)
        PLfig.set_figwidth(11)
        PLfig.set_figheight(6)
        pnames = sorted(self.pnames)

        for i in range(nrow):
            pname = pnames[i]
            pkeys = sorted(filter(lambda x: x.split('_')[0] == pname,
                                  self.var_df.index.values))
            pdata = self.var_df.loc[pkeys]
            pdata = pdata.sort_values(pname)
            ob = [np.log(self.obj_dict[key]) for key in pkeys]
            pl = [self.var_df[pname][key] for key in pkeys]
            ob = [x for y, x in sorted(zip(pl, ob))]
            pl = sorted(pl)
            cg = [a - self.popt[pname] for a in pl]
            cmap = mpl.cm.seismic
            norm = mpl.colors.Normalize(vmin=-max(np.abs(cg)),
                                        vmax=max(np.abs(cg)))

            for j in range(ncol):
                if i == j:
                    if nPars == 1:
                        ax = axs
                    else:
                        ax = axs[i, j]
                    ax.scatter(pl, ob, c=cg, cmap=cmap, norm=norm)
                    chibd = np.log(self.obj) + chi2.isf(self.alpha, 1)/2
                    ax.scatter(self.popt[pname], np.log(self.obj), c='g')
                    ax.plot([pl[0], pl[-1]], [chibd, chibd], color='k')
                    ax.set_xlabel(pname +' Value', fontsize=10)
                else:
                    ax = axs[i, j]
                    ax.scatter(pdata[pnames[j]], ob, c=cg, cmap=cmap,
                               norm=norm)
                    ax.scatter(self.popt[pnames[j]], np.log(self.obj), c='g')
                    ax.set_xlabel(pnames[j] + ' Value', fontsize=10)
                if j == 0:
                    ax.set_ylabel('Objective Value', fontsize=10)
        PLfig.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(fname, dpi=600)
            plt.close("all")
        return PLfig


    def plot_simplePL(self, show=True, fname=None):
        # Plot just the PLs for each parameter - currently non-functional
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
                                  self.var_df.index.values))
            pdata = self.var_df.loc[pkeys]
            pdata = pdata.sort_values(pname)
            ob = [np.log(self.obj_dict[key]) for key in pkeys]
            pl = [self.var_df[pname][key] for key in pkeys]
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
        if show:
            plt.show()
        else:
            plt.savefig(fname, dpi=600)
        return PL_fig

    def plot_dual(self, maxdtheta=3, sep_index=True, show=True, fname=None):
        # This one works :)
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Get variable data in plotting format
        m = ['.', '^', 'x', 's']
        df = pd.DataFrame(self.var_df).T
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
        nrow = np.floor(2*nPars/3)
        if nrow < 1:
            nrow = 1
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
            i += 1
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(fname, dpi=600)
        return dual_fig

    def to_json(self, filename):
        # save PL data to a json file
        atts = ['alpha', 'parub', 'parlb', 'var_df', 'obj_dict', 'flag_dict']
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
