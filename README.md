# Profile Likelihood Estimator in Python (PLEpy)

**P**rofile **L**ikelihood **E**stimator in **Py**thon (PLEpy) is a python package for generating parameter likelihood profiles for Pyomo models. These profiles can be used to determine whether parameters are identifiable, practically non-identifiable or structurally non-identifiable. If the parameter is identifiable, confidence limits will be calculated for it. For more detail on the theory behind this method, see [Raue et al. (2009)][Raue2009].

[**Pyomo**][Pyomo] is an open-source modeling and optimization framework. It was chosen as a base for this tool due to its flexibility and user-friendliness. For documentation please refer to [www.pyomo.org/documentation](http://www.pyomo.org/documentation).

## Features

- Upper and lower non-linear confidence intervals (if they exist)
- Likelihood profile generation
- Optimal values of other model parameters for each point on likelihood profile

## Install

PLEpy can be installed using `pip`. Installation with `conda` is not yet supported, but will be in future versions. To install use the following command:
```
pip install plepy
```

Alternatively, this repository can be cloned:
```
git clone https://gitlab.com/monshap/hoc6h4cooh/
```
Make sure you manually add the local location of this package to your `PATH` and `PYTHONPATH` variables.
<!-- double check minimum variables you need to add this to -->

To test whether or not PLEpy was successfully installed, run one of the examples in the `/examples` folder. To run the rapid TEG example use the following `python` commands:
```python
python examples/rapidTEG/rapidTEG_example.py
```

## Quick Start
There are 6 key steps for using PLEpy:
1. Import the package
2. Define Pyomo model
3. Create PLEpy instance
4. Get confidence limits (if any)
5. Generate profiles between confidence limits
6. Plot results

### 1. Import the package
```python
from plepy import PLEpy
```

### 2. Define Pyomo model
User must define a Pyomo model with at least one variable and an objective function. Currenty, the **objective function must be named** `obj` (i.e. for a Pyomo model, `model`, it must be defined as `model.obj`). This will be fixed in future versions to enable flexible naming of objective function. For reference on how to build a Pyomo model, see [www.pyomo.org/documentation](www.pyomo.org/documentation).

### 3. Create a PLEpy instance
```python
pl_inst = PLEpy(model, ['par1', 'par2', ..., 'parN'], indices=None, **kwds)
```
where `model` is your Pyomo model and `'par1'` throught `'parN'` are the names of the parameters to be profiled. `indices` is an optional argument that provides key-value pairs of index values for any indexed parameters. For example usage, see `/examples/fiveshell/shell_example.py`. Other options can be passed in using keywords. Eventually, there will be an API.

### 4. Get confidence limits (if any)
```python
pl_inst.get_clims(pnames='all', alpha=0.05, acc=3)
```
The `.get_clims()` method will get upper and lower confidence limits (CLs) with power, `alpha`, for parameters listed in `pnames`. `pnames` can take the name of a single parameter, a list of parameters, or the value `'all'` (default), which will CLs for all parameters listed at creation of PLEpy instance (`'par1', 'par2', ..., 'parN'` in example above). `acc` specifies the number of significant digits in the resulting CLs. **Confidence limits are required for parameters before they can be profiled (step 5).**

### 5. Generate profiles between confidence limits
```python
pl_inst.get_PL(pnames='all', n=20, min_step=1e-3, dtol=0.2, save=False)
```
The `.get_PL()` method generates profile likelihood curves between either upper and lower CLs (or parameter bounds, if there were no CLs). As in step 4, `pnames` is can take the name of a single parameter, a list of parameters, or the value `'all'` (default). `n` specifies the minimum number of number of evaluations between each bound (increase to refine profile shape). On each interval (between evaluation points), if the change in $`|2ln(\textrm{Objective})|`$ is greater than `dtol` and the interval is greater than `min_step`, the midpoint will be evaluated. Profiling will stop once there are no more intervals with a profile change greater than `dtol` or all remaining intervals are smaller than `min_step`. If you would like to save these values to a JSON file, the `save` option can be set to `True` and an additonal keyword, `fname`, with a name for the file can be specified (default value is `'tmp_PLfile.json'`).

**Note**: This method requires upper and lower bounds for each parameter profiled (given by `pnames`). These can be generated using `.get_clims()` method (recommended) or manually specified as dictionaries (`pl_inst.parlb` and `pl_inst.parub` for lower and upper limits, respectively) with parameter names as keys and bounds as values.

### 6. Plot results
```python
pl_inst.plot_PL(**kwds)
```
Default usage ot the `.plot_PL()` method plots profiles for all parameters and all covariates on $`2 \times N`$ subplots, with a maximum of 4 columns per window. See example shown below. For more advanced usage, see [full API](/#).

![Profile Likelihood Plots](/tests/rapidTEG/rapidTEG_plots.png)


## References
1. A. Raue, C. Kreutz, T. Maiwald, J. Bachmann, M. Schilling, U. Klingmüller, J. Timmer. "Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood." *Bioinformatics* 25(15) (2009): 1923–1929. [https://doi.org/10.1093/bioinformatics/btp358][Raue2009]
2. Hart, William E., Carl D. Laird, Jean-Paul Watson, David L. Woodruff, Gabriel A. Hackebeil, Bethany L. Nicholson, and John D. Siirola. *Pyomo – Optimization Modeling in Python*. Second Edition.  Vol. 67. Springer, 2017. [https://doi.org/10.1007/978-3-319-58821-6][Hart2017Pyomo]

3. Hart, William E., Jean-Paul Watson, and David L. Woodruff. "Pyomo: modeling and solving mathematical programs in Python." *Mathematical Programming Computation* 3(3) (2011): 219-260. [https://doi.org/10.1007/s12532-011-0026-8][Hart2011Pyomo]

4. Nicholson, Bethany, John D. Siirola, Jean-Paul Watson, Victor M. Zavala, and Lorenz T. Biegler. "pyomo.dae: a modeling and automatic discretization framework for optimization with differential and algebraic equations." *Mathematical Programming Computation* 10(2) (2018): 187-223. [https://doi.org/10.1007/s12532-017-0127-0][Nicholson2018]

<!-- Links -->
[Raue2009]: https://doi.org/10.1093/bioinformatics/btp358
[Pyomo]: http://www.pyomo.org/
[Hart2017Pyomo]: https://doi.org/10.1007/978-3-319-58821-6
[Hart2011Pyomo]: https://doi.org/10.1007/s12532-011-0026-8
[Nicholson2018]: https://doi.org/10.1007/s12532-017-0127-0
