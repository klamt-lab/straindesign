from setuptools import setup,find_packages

setup(name='straindesign',
    version='1.1',
    url='https://github.com/klamt-lab/straindesign.git',
    description="Computational strain design package for the COBRApy framework",
    long_description="Computational strain design package for the COBRApy framework, offering standard and advanced tools for the analysis and redesign of biological networks",
    long_description_content_type="text/plain",
    author='Philipp Schneider',
    author_email='zgddtgt@gmail.com',
    license='Apache License 2.0',
    python_requires='>=3.7',
    package_data={'straindesign': ['efmtool.jar']},
    packages=find_packages(),
    install_requires=['cobra','jpype1','scipy','matplotlib'],
    zip_safe=False)
