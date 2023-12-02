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
   :target: https://readthedocs.org/projects/straindesign/builds/
   :alt: Documentation Status
   
.. image:: https://img.shields.io/pypi/pyversions/straindesign.svg
   :target: https://pypi.org/project/straindesign/
   :alt: Supported Python Versions

.. image:: https://github.com/klamt-lab/straindesign/workflows/CI-test/badge.svg
    :target: https://github.com/klamt-lab/straindesign/actions/workflows/CI-test.yml
    :alt: GitHub Actions CI-test Status
   
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

The comprehensive StrainDesign package for MILP-based strain design computation with the COBRApy toolbox supports MCS, MCS with nested optimization, OptKnock :html:`<a href="#ref2"><sup>[2]</sup></a>`, RobustKnock :html:`<a href="#ref3"><sup>[3]</sup></a>` and OptCouple :html:`<a href="#ref4"><sup>[4]</sup></a>`, GPR-rule integration, gene and reaction knockouts and additions as well as regulatory interventions. The automatic lossless network and GPR compression allows strain design computations from genome-scale metabolic networks. Supported solvers are GLPK (available from COBRApy), CPLEX, Gurobi and SCIP :html:`<a href="#ref5"><sup>[5]</sup></a>`. :html:`<br>` 

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

Parts of the compression routine is done by efmtool's compression function (https://csb.ethz.ch/tools/software/efmtool.html\ :html:`<a href="#ref6"><sup>[6]</sup></a>`).

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

JAVA_HOME path:
---------------

In some cases, installing the StrainDesign python package may fail with the error:

``JVMNotFoundException: No JVM shared library file (libjli.dylib) found. Try setting up the JAVA_HOME environment variable.``

In this case, make sure Java is installed correctly and the JAVA_HOME varialbe is set. `JAVA_HOME environment variable <https://www.baeldung.com/java-home-on-windows-7-8-10-mac-os-x-linux>`_

If you're on OS X and get the error

``OSError: [Errno 0] JVM DLL not found``

check that your `Java and the JPype library is set up correctly <https://github.com/jpype-project/jpype/issues/994>`_. The easiest way to avoid this error is to use conda to install StrainDesign.

:html:`<a id="examples"></a>`\ Examples:
================================================

Computation examples are provided in the different chapters of this documentation. The original Jupyer notebook files are located in the StrainDesign package at `docs/source/examples <https://github.com/klamt-lab/straindesign/tree/main/docs/source/examples>`_.

How to cite:
============

:html:`<a id="ref0"></a>` `Schneider P., Bekiaris P. S., von Kamp A., Klamt S. - StrainDesign: a comprehensive Python package for computational design of metabolic networks. Bioinformatics, btac632 (2022)  <https://doi.org/10.1093/bioinformatics/btac632>`_


.. toctree::
   :maxdepth: 4
   :caption: Contents:

   examples/JN_01_solver.ipynb
   examples/JN_02_analysis.ipynb
   examples/JN_03_plotting.ipynb
   examples/JN_04_strain_design_introduction.ipynb
   examples/JN_05_strain_design_mcs.ipynb
   examples/JN_06_strain_design_nested.ipynb
   examples/JN_08_compression.ipynb
   9_cnapy_integration
   api_reference
   
.. 
   examples/JN_07_network_design.ipynb

References:
===========
:html:`<a id="ref1">[1]</a>` `Ebrahim, A., Lerman, J.A., Palsson, B.O. et al. - COBRApy: COnstraints-Based Reconstruction and Analysis for Python. BMC Syst Biol 7, 74 (2013) <http://dx.doi.org/doi:10.1186/1752-0509-7-74>`_

:html:`<a id="ref2">[2]</a>` `Burgard, A. P., Pharkya, P., & Maranas, C. D. - Optknock: a bilevel programming framework for identifying gene knockout strategies for microbial strain optimization. Biotechnology and bioengineering, 84(6), 647–657 (2003) <https://doi.org/10.1002/bit.10803>`_

:html:`<a id="ref3">[3]</a>` `Tepper N., Shlomi T. - Predicting metabolic engineering knockout strategies for chemical production: accounting for competing pathways, Bioinformatics. Volume 26, Issue 4, Pages 536–543 (2010) <https://doi.org/10.1093/bioinformatics/btp704>`_

:html:`<a id="ref4">[4]</a>` `Jensen K., Broeken V., Lærke Hansen A.S., et al. - OptCouple: Joint simulation of gene knockouts, insertions and medium modifications for prediction of growth-coupled strain designs. Metabolic Engineering Communications, Volume 8 (2019) <https://doi.org/10.1016/j.mec.2019.e00087>`_

:html:`<a id="ref5">[5]</a>` `Bestuzheva K., Besançon M., Chen W.K. et al. - The SCIP Optimization Suite 8.0. Available at Optimization Online and as ZIB-Report 21-41, (2021) <https://doi.org/10.48550/arXiv.2112.08872>`_

:html:`<a id="ref6">[6]</a>` `Marco Terzer, Jörg Stelling, Large-scale computation of elementary flux modes with bit pattern trees, Bioinformatics, Volume 24, Issue 19, (2008), Pages 2229–2235, <https://doi.org/10.1093/bioinformatics/btn401>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

