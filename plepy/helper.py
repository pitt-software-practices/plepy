import numpy as np


def sigfig(n, sf: int):
    mag = int(np.floor(np.log10(n)))
    rf = sf - mag - 1
    return round(n, rf)


def recur_to_json(d: dict) -> dict:
    # recurssively convert dictionaries to compatible forms for JSON
    # serialization (keys must be strings)
    for key in list(d.keys()):
        if isinstance(d[key], dict):
            d[key] = recur_to_json(d[key])
    d2 = {str(key): d[key] for key in list(d.keys())}
    return d2


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


def sflag(results):
    # determine solver status for iteration & assign flag
    from pyomo.opt import SolverStatus, TerminationCondition

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
        profile likelihood data generated from PLEpy function,
        'get_PL()', has format:
        {'pname': {par_val: {keys: 'obj', 'par1', 'par2', etc.}}}
    clevel: float
        value of objective at confidence threshold

    Keywords
    --------
    pnames : list or str, optional
        parameter(s) to generate plots for, if 'all' will plot for all
        keys in outer level of dictionary, by default 'all'
    covar : list or str, optional
        parameter(s) to include covariance plots for, if 'all' will
        include all keys in outer level of dictionary, by default 'all'
    join : bool, optional
        place multiple profile likelihood plots on a single figure, by
        default False
    jmax : int, optional
        if join=True, the maximum number of plots to put in a single
        figure, by default 4
    disp: str, optional
        how to display generated figures:
        'show' will run command plt.show(),
        'save' will save figures using filename prefix specified in
            fprefix,
        'None' will not display figures and simply return their
            handles,
        by default 'show'
    fprefix: str, optional
        filename prefix to give figures if disp='save', by default
        'tmp_fig'
    **dispkwds: optional
        Keywords to pass to display function (either fig.show() or
        fig.savefig())
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # TODO: enable plotting of individual index values for indexed
    # variables

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
        # if the number of profiled parameters is not divisible by the
        # maximum number of subplot columns (jmax), make the first
        # figure generated contain the remainder
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
        # since we already plotted the remainder parameters, nleft
        # should be divisible by jmax now
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
