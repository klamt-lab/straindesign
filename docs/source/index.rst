.. role:: html(raw)
   :format: html

.. StrainDesign documentation master file, created by
   sphinx-quickstart on Thu May 19 13:21:03 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

====================================================================================
StrainDesign
====================================================================================
.. |logo| image:: https://github.com/klamt-lab/straindesign/blob/host_gifs/docs/logo.svg
  :target: https://straindesign.readthedocs.io/en/latest/
  :width: 50
  :alt: Icon

.. image:: https://img.shields.io/github/v/release/klamt-lab/straindesign.svg
   :target: https://github.com/klamt-lab/straindesign/releases
   :alt: Current Release

.. image:: https://img.shields.io/pypi/v/straindesign.svg
   :target: https://pypi.org/project/straindesign/
   :alt: Current PyPI Version
   
.. image:: https://anaconda.org/cnapy/straindesign/badges/version.svg
   :target: https://anaconda.org/cnapy/straindesign/
   :alt: Current Anaconda Version
   
.. image:: https://readthedocs.org/projects/straindesign/badge/?version=latest
   :target: https://straindesign.readthedocs.io/en/latest/
   :alt: Documentation Status
   
.. image:: https://img.shields.io/pypi/pyversions/straindesign.svg
   :target: https://pypi.org/project/straindesign/
   :alt: Supported Python Versions

.. image:: https://github.com/klamt-lab/straindesign/workflows/CI-test/badge.svg
    :target: https://github.com/klamt-lab/straindesign/workflows/CI-test]
    :alt: GitHub Actions CI-test Status]
   
.. image:: https://img.shields.io/pypi/l/straindesign.svg
   :target: https://www.gnu.org/licenses/old-licenses/lgpl-2.0.html
   :alt: Apache 2.0 License

.. image:: https://img.shields.io/badge/code%20style-yapf-blue
   :target: https://github.com/google/yapf
   :alt: YAPF
   

..
  .. image:: https://zenodo.org/badge/6510063.svg
     :target: https://zenodo.org/badge/latestdoi/6510063
     :alt: Zenodo DOI
     
A COBRApy\ :html:`<a href="#ref1"><sup>[1]</sup></a>`\ -based package for computational design of metabolic networks
====================================================================================================================

The comprehensive StrainDesign package for MILP-based strain design computation with the COBRApy toolbox supports MCS, MCS with nested optimization, OptKnock, RobustKnock and OptCouple, GPR-rule integration, gene and reaction knockouts and additions as well as regulatory interventions. The automatic lossless network and GPR compression allows strain design computations from genome-scale metabolic networks. Supported solvers are GLPK (available from COBRApy), CPLEX, Gurobi and SCIP. :html:`<br>` 

:html:`<a href="#installation">Installation instructions ...</a>`
:html:`<a href="#examples">Download Jupyter notebook examples ...</a>`

|pic1| |pic2| |pic3| 

.. |pic1| image:: https://raw.githubusercontent.com/klamt-lab/straindesign/host_gifs/docs/puzzle.svg
  :width: 25%
  :alt: Network interventions   
   
.. |pic2| image:: https://raw.githubusercontent.com/klamt-lab/straindesign/host_gifs/docs/network.svg
  :width: 30%
  :alt: Network interventions
  
.. |pic3| image:: https://raw.githubusercontent.com/klamt-lab/straindesign/host_gifs/docs/plot.gif
  :width: 40%
  :alt: Plot animation

Parts of the compression routine is done by efmtool's compression function (https://csb.ethz.ch/tools/software/efmtool.html\ :html:`<a href="#ref2"><sup>[2]</sup></a>`).

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   examples/JN_01_solver.ipynb
   examples/JN_02_analysis.ipynb
   examples/JN_03_plotting.ipynb
   examples/JN_04_strain_design_introduction.ipynb
   examples/JN_05_strain_design_mcs.ipynb
   examples/JN_06_strain_design_nested.ipynb
   examples/JN_07_network_design.ipynb
   examples/JN_08_compression.ipynb
   9_cnapy_integration
   api_reference

:html:`<a id="installation"></a>`\ Installation:
================================================

The StrainDesign package is available on pip and Anaconda. To install the latest release, run:

``pip install straindesign``

or

``conda install -c cnapy straindesign``

Developer Installation:
-----------------------

Download the repository and run

``pip install -e .``

in the main folder. Through the installation with -e, updates from a 'git pull' are at once available in your Python envrionment without the need for a reinstallation.

:html:`<a id="examples"></a>`\ Examples:
================================================

Computation examples are provided in the different chapters of this documentation. The original Jupyer notebook files are located in the StrainDesign package at `docs/source/examples <https://github.com/klamt-lab/straindesign/tree/main/docs/source/examples>`_.


References:
===========
:html:`<a id="ref1">[1]</a>` `Ebrahim, A., Lerman, J.A., Palsson, B.O. et al. COBRApy: COnstraints-Based Reconstruction and Analysis for Python. BMC Syst Biol 7, 74 (2013) <http://dx.doi.org/doi:10.1186/1752-0509-7-74>`_

:html:`<a id="ref2">[2]</a>` `Marco Terzer, J??rg Stelling, Large-scale computation of elementary flux modes with bit pattern trees, Bioinformatics, Volume 24, Issue 19, (2008), Pages 2229???2235, <https://doi.org/10.1093/bioinformatics/btn401>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
