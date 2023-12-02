import os
import subprocess
import sys
import platform
from setuptools import setup, find_packages

def set_java_home():
    if platform.system() == 'Darwin':  # Check if OS is macOS
        try:
            # Attempt to find Java home
            java_home_output = subprocess.check_output(['/usr/libexec/java_home'], universal_newlines=True)
            java_home_path = java_home_output.strip()
            os.environ['JAVA_HOME'] = java_home_path  # Set for the current process
            print(f"JAVA_HOME set to {java_home_path}")
        except subprocess.CalledProcessError:
            sys.exit("Java not found. Please install Java and set JAVA_HOME.")
    else:
        # Handle other OS or assume JAVA_HOME is set
        if not os.environ.get('JAVA_HOME'):
            sys.exit("JAVA_HOME environment variable is not set. Please set JAVA_HOME.")

set_java_home()

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
