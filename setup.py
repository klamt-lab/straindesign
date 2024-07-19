from setuptools import setup, find_packages

setup(
    name="straindesign",
    version="1.12",
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
        "Intended Audience :: Science/Research", "Development Status :: 3 - Alpha", "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", "Programming Language :: Python :: 3.11", "Programming Language :: Python :: 3.12",
        "Natural Language :: English", "Operating System :: OS Independent", "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
    keywords=["metabolism", "constraint-based", "mixed-integer", "strain design"],
    zip_safe=False,
)
