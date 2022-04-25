from setuptools import setup

setup(name='straindesign',
    version='0.1',
    description='Computation of Strain Designs.',
    url='https://github.com/VonAlphaBisZulu/straindesign.git',
    author='Philipp Schneider',
    author_email='zgddtgt@gmail.com',
    license='Apache License 2.0',
    packages=['straindesign'],
    install_requires=['numpy', 'scipy', 'cobra', 'optlang', 'efmtool_link', 'sympy', 'swiglpk','pickle','json','re'],
    zip_safe=False)
