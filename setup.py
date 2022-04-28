from setuptools import setup,find_packages

setup(name='straindesign',
    version='0.1',
    url='https://github.com/klamt-lab/straindesign.git',
    description="Computational strain design package for the COBRApy framework",
    author='Philipp Schneider',
    author_email='zgddtgt@gmail.com',
    license='Apache License 2.0',
    python_requires='>=3.7',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'sympy', 'cobra', 'optlang', 'efmtool_link', 'swiglpk'],
    zip_safe=False)
