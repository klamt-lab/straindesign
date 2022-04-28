# Strain design package for COBRApy
Comprehensive package for computing strain design designs with the COBRApy toolbox. Supports MCS, MCS with nested optimization, OptKnock, RobustKnock and Optcouple, uses GPR-rule and network compression and allows for reaction and/or gene addition/removal/regulation.

This package uses the efmtool compression routine for reducing the network size during preprocessing (https://csb.ethz.ch/tools/software/efmtool.html).

## Installation:

The straindesign package is available on pip and conda. To install the latest release, run

```pip install straindesign```

First you need to install cobra and the efmtool_link package (also available at https://github.com/cnapy-org).
Clone the repository, go into the top straindesign directory and install into your current Python environment with:

### Developer Installation:
pip install .

Tip: If you use the -e option during installation then updates from a 'git pull' are at once available in your Python envrionment without the need for a reinstallation.

Examples:

Will be added soon...
