from setuptools import setup

setup(name='mcs',
    version='0.1',
    description='Computation of MCS.',
    url='https://github.com/VonAlphaBisZulu/mcs.git',
    author='Philipp Schneider',
    author_email='zgddtgt@gmail.com',
    license='Apache License 2.0',
    packages=['mcs'],
    install_requires=['numpy', 'scipy', 'cobra', 'optlang', 'efmtool_link', 'sympy', 'swiglpk'],
    zip_safe=False)