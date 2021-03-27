# optlang_enumerator
Module for enumerating multiple solutions to a MILP problem using the optlang framework.
Currently only the enumeration of constrained minimal cut sets is implemented.

Installation:

First you need to install the efmtool_link package (also available at https://github.com/cnapy-org).
Clone the repository, go into the top optlang_enumerator directory and install into your current Python environment with:

pip install .

Tip: If you use the -e option during installation then updates from a 'git pull' are at once available in your Python envrionment without the need for a reinstallation.

Example:

The ECC2comp.ipynb Jupyter notebook in the examples directory shows how to perform a basic MCS calculation.
