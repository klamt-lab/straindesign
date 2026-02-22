Legacy Methods
==============

This page documents optional legacy functionality that requires additional
dependencies beyond the core StrainDesign installation.

Java-based EFMTool compression (``backend='efmtool_rref'``)
------------------------------------------------------------

The default compression backend is ``backend='sparse_rref'``, a pure Python
implementation with no extra dependencies. A legacy Java-based backend is
available for comparison or reproducibility purposes.

To use it, install the optional Java dependency::

    pip install straindesign[java]

or::

    pip install jpype1

Then pass ``backend='efmtool_rref'`` to :func:`~straindesign.compress_model`
or to ``compute_strain_designs`` via the ``backend`` keyword argument.

JAVA_HOME path
--------------

In some cases, using the ``efmtool_rref`` backend may fail with:

``JVMNotFoundException: No JVM shared library file (libjli.dylib) found. Try setting up the JAVA_HOME environment variable.``

In this case, make sure Java is installed correctly and the JAVA_HOME variable
is set. See `JAVA_HOME environment variable <https://www.baeldung.com/java-home-on-windows-7-8-10-mac-os-x-linux>`_
for platform-specific instructions.

If you're on OS X and get the error

``OSError: [Errno 0] JVM DLL not found``

check that your `Java and the JPype library is set up correctly <https://github.com/jpype-project/jpype/issues/994>`_.
The easiest way to avoid this error is to use conda to install StrainDesign and
Java together.
