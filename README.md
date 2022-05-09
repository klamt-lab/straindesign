# Strain design package for COBRApy
Comprehensive package for computing strain design designs with the COBRApy toolbox. Supports MCS, MCS with nested optimization, OptKnock, RobustKnock and OptCouple, uses GPR-rule and network compression and allows for reaction and/or gene addition/removal/regulation.

This package uses the efmtool compression routine for reducing the network size during preprocessing (https://csb.ethz.ch/tools/software/efmtool.html).

## Installation:

The straindesign package is available on pip and conda. To install the latest release, run

```pip install straindesign```

or

```conda install -c cnapy straindesign```

### Developer Installation:
Download the repository and run

`pip install -e .`

in the main folder. Through the installation with -e, updates from a 'git pull' are at once available in your Python envrionment without the need for a reinstallation.

## Install additional solvers:
The cobra package is shipped with the GLPK solver. The more powerful commercial solvers IBM CPLEX and Gurobi may be used by cobra and the straindesign package. This makes sense in particular when using strain design algorithms like MCS, OptKnock etc. As another alternative solver, SCIP may be used. In the following, you will find installation instructions for the individual solvers.

### CPLEX
Together with Gurobi, CPLEX is the perfect choice for computing strain designs. Its stability and support of advanced features like indicator constraints and populating solution pools make it indispensible for genome-scale computations.

You will need an academic or commercial licence for CPLEX. Download and install the CPLEX suite and make sure that your CPLEX and Python versions are compatible. This step will not yet install CPLEX in your Python environment. Once the installation is completed, you may link your installation to your Python/conda environment. This is the next step.

Using the command line, navigate to your CPLEX installation path and into the Python folder. The path should look similar to 

`C:/Program Files/CPLEX210/python`

Make sure to activate the same Python/conda environment where `cobra` and `straindesign` are installed. Then call 

`python setup.py install`. 

Now CPLEX should be available for your computations.

The official instructions can be found here: https://www.ibm.com/docs/en/icos/22.1.0?topic=cplex-setting-up-python-api

### Gurobi
Similar to CPLEX, Gurobi offers a fast MILP solvers with the advanced features of indicator constraints and solution pooling. The installation steps are similar to the ones of CPLEX.

First, you will need an academic or commercial license and install the Gurobi solver software. Ensure that the versions of gurobi and Python versions are compatible, install Gurobi to your system and activate your license following the steps from the Gurobi manual. In the next step you will link your Gurobi installation to your Python/conda environment.

Using the command line, navigate to your CPLEX installation path and into the Python folder. The path should look similar to 

`C:/gurobi950/windows64`

Make sure to activate the same Python/conda environment where `cobra` and `straindesign` are installed. Then call 

`python setup.py install`.

If your `gurobipy` package does not work right away, additionally install the gurobi package from conda or PyPi via

`conda install -c gurobi gurobi`

or

`python -m pip install gurobipy`

Now Gurobi should be available for your computations.

The official instructions can be found here: https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-

### SCIP
Less powerfull than CPLEX and Gurobi, the open source solver SCIP still offers the solution of MILPs with indicator constraints, which gives it an edge above GLPK in terms of stability. If you want to use SCIP, you may install it via conda or pip:

`conda install -c conda-forge pyscipopt`

or

`python -m pip install pyscipopt`

Official website: https://github.com/scipopt/PySCIPOpt

## Examples:

Will be added soon...
