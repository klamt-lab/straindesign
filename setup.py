from setuptools import setup, find_packages
import pkg_resources
from setuptools.command.install import install
import subprocess
import sys

# This is ugly but necessary, because jpype wouldn't install from conda on macos
class CustomInstall(install):
    def run(self):
        # Ensure jpype1 is installed via pip
        try:
            pkg_resources.get_distribution('jpype1')
        except pkg_resources.DistributionNotFound:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'jpype1'])
        
        # Continue with the standard install
        install.run(self)

setup(
    name="straindesign",
    version="1.10",
    url="https://github.com/klamt-lab/straindesign.git",
    description="Computational strain design package for the COBRApy framework",
    long_description=
    "Computational strain design package for the COBRApy framework, offering standard and advanced tools for the analysis and redesign of biological networks",
    long_description_content_type="text/plain",
    author="Philipp Schneider",
    author_email="zgddtgt@gmail.com",
    license="Apache License 2.0",
    python_requires=">=3.7",
    package_data={"straindesign": ["efmtool.jar"]},
    packages=find_packages(),
    install_requires=["cobra", "jpype1", "scipy", "matplotlib", "psutil"],
    project_urls={
        "Bug Reports": "https://github.com/klamt-lab/straindesign/issues",
        "Source": "https://github.com/klamt-lab/straindesign/",
        "Documentation": "https://straindesign.readthedocs.io/en/latest/index.html"
    },
    classifiers=[
        "Intended Audience :: Science/Research", "Development Status :: 3 - Alpha", "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", "Programming Language :: Python :: 3.9", "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11", "Natural Language :: English", "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
    keywords=["metabolism", "constraint-based", "mixed-integer", "strain design"],
    zip_safe=False,
)
