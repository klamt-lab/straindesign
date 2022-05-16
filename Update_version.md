This is only needed for manual version upgrading. A versioning script is availabe in GitHub actions.

## In any case:
1. Update version number in `conda-recipe/meta.yaml`. Update dependencies/versions and description if necessary
2. Update version number in `setup.py`. Update dependencies/versions and description and `requirements.txt` if necessary.
3. Create a new tag/release on GitHub with matching version number (e.g. `v0.1`)

## Building PyPi package

### Prerequisites
1. Install newest version of pip (`python -m pip install --upgrade pip` or `conda update pip`)
2. Install twain (`pip install twain` or `conda install twain`)

### Build and upload package

1. Navigate to package folder and build package with `python setup.py sdist bdist_wheel`
2. Clean up dist folder (remove old version builds)
3. Upload package source and wheel to PyPi via `twine upload dist/*` (Use PyPi credencials)

## Building Conda package

### Prerequisites
1. Create a conda environment to build the package (e.g. `conda create -n straindesign-build`)
2. Activate environment (`conda activate straindesign-build`)
3. Install requirements (`conda install anaconda-client conda-build`)

### Build and upload package

1. Navigate to package folder and build package with `conda-build conda-recipe/. -c conda-forge --croot conda-bld`
2. Clean up conda-bld folder (remove old version builds)
3. `anaconda login`
4. `anaconda upload -u cnapy conda-bld/noarch/straindesign*`
5. Clean up with conda
