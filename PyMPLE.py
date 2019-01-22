import json
import numpy as np
import pandas as pd
from numpy import copy
from scipy.stats.distributions import chi2
from pyomo.dae import *
from pyomo.environ import *


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

        def pprint(pname, inum, ierr, ipval, istep, ifreq=20):
            dash = '='*90
            head = ' Iter. | Error | Par. Value | Stepsize | Par. Name'
            iform = ' {:^5d} | {:^5.3f} | {:>10.4g} | {:>8.3g} | {:<49s}'
            iprint = iform.format(inum, ierr, ipval, istep, pname)
            if inum % ifreq == 0:
                print(*[dash, head, dash], sep='\n')
                print(iprint)
            else:
                print(iprint)

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
        vdict = dict()
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
            iname = '_'.join([pname, dr, str(i)])
            try:
                riter = self.solver.solve(self.m)
                self.m.solutions.load_from(riter)

                err = 2*(np.log(value(self.m.obj)) - np.log(_obj_CI))
                vdict[iname] = {k: value(getattr(self.m, k))
                                for k in self.pnames}
                _obj_dict[iname] = value(self.m.obj)

                # adjust step size if convergence slow
                if i > 0:
                    prname = '_'.join([pname, dr, str(i-1)])
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
                pprint(pname, i, err, pardr, stepfrac*self.popt[pname])
                if err > etol:
                    print('Reached %s CI!' % (drer))
                    print('{:s} = {:.4g}'.format(dB, pardr))
                    v_dict = pd.DataFrame.from_dict(vdict, orient='index')
                    return pardr, states_dict, v_dict, _obj_dict
                elif i == ctol-1:
                    print('Maximum steps taken!')
                    if dr == 'up':
                        v_dict = pd.DataFrame.from_dict(vdict, orient='index')
                        return np.inf, states_dict, v_dict, _obj_dict
                    else:
                        v_dict = pd.DataFrame.from_dict(vdict, orient='index')
                        return -np.inf, states_dict, v_dict, _obj_dict

                nextdr += self.popt[pname]*stepfrac
                if dr == 'up':
                    bdreach = nextdr > bound
                else:
                    bdreach = nextdr < bound

                if bdreach:
                    print('Reached parameter %s bound!' % (drer))
                    print('{:s} = {:.4g}'.format(dB, pardr))
                    v_dict = pd.DataFrame.from_dict(vdict, orient='index')
                    return pardr, states_dict, v_dict, _obj_dict
                i += 1
            except Exception as e:
                z = e
                print(z)
                prname = '_'.join([pname, dr, str(i-1)])
                iname = '_'.join([pname, dr, str(i)])
                pardr = vdict[prname][pname]
                states_dict.pop(iname, None)
                vdict.pop(iname, None)
                _obj_dict.pop(iname, None)
                i = ctol
                print('Error occured!')
                print('{:s} set to {:.4g}'.format(dB, pardr))
                v_dict = pd.DataFrame.from_dict(vdict, orient='index')
                return pardr, states_dict, v_dict, _obj_dict

    def get_CI(self, maxSteps=100, alpha=0.05, stepfrac=0.01):

        # Get Confidence Intervals
        self.ctol = maxSteps
        self.alpha = alpha

        states_dict = dict()

        parub = dict(self.popt)
        parlb = dict(self.popt)
        _var_df = pd.DataFrame(columns=self.pnames)
        _obj_dict = dict()

        _obj_CI = value(self.obj)

        # Initialize parameters
        for pname in self.pnames:
            # manually change parameter of interest
            self.plist[pname].fix()

            # step to upper limit
            print(' '*90)
            print('Parameter: {:s}'.format(pname))
            print('Direction: Upward')
            print('Bound: {:<.3g}'.format(self.pbounds[pname][1]))
            print(' '*90)
            parub[pname], upstates, upvars, upobj = self.step_CI(
                pname, dr='up', stepfrac=stepfrac
            )
            states_dict = {**states_dict, **upstates}
            _var_df = pd.concat([_var_df, upvars])
            _obj_dict = {**_obj_dict, **upobj}

            # step to lower limit
            print(' '*90)
            print('Parameter: {:s}'.format(pname))
            print('Direction: Downward')
            print('Bound: {:<.3g}'.format(self.pbounds[pname][0]))
            print(' '*90)
            self.plist[pname].set_value(self.popt[pname])
            parlb[pname], dnstates, dnvars, dnobj = self.step_CI(
                pname, dr='down', stepfrac=stepfrac
            )
            states_dict = {**states_dict, **dnstates}
            _var_df = pd.concat([_var_df, dnvars])
            _obj_dict = {**_obj_dict, **dnobj}
            
            # reset variable
            self.plist[pname].set_value(self.popt[pname])
            self.plist[pname].unfix()

        # assign profile likelihood bounds to PyMPLE object
        self.parub = parub
        self.parlb = parlb
        self.var_df = _var_df
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

    def plot_PL(self, show=True, fname=None):
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
        atts = ['alpha', 'parub', 'parlb', 'var_df', 'obj_dict']
        sv_dict = {}
        for att in atts:
            sv_var = getattr(self, att)
            if type(sv_var) == pd.DataFrame:
                sv_var = sv_var.to_dict()
            sv_dict[att] = sv_var
        with open(filename, 'w') as f:
            json.dump(sv_dict, f)
    
    def load_json(self, filename):
        with open(filename, 'r') as f:
            sv_dict = json.load(f)
        for att in sv_dict.keys():
            if att == 'var_df':
                sv_dict[att] = pd.DataFrame.from_dict(sv_dict[att])
            setattr(self, att, sv_dict[att])
