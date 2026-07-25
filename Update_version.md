Releases are normally cut by the `Build and Upload Python Package` GitHub action,
which is triggered manually and takes the new version number as its input. It
bumps the version, formats the repository, creates the tag and release, and
uploads to PyPI and the `cnapy` Anaconda channel. The steps below are the manual
equivalent, for the case where the action cannot be used.

## In any case:
1. Update the version number in `pyproject.toml`, `conda-recipe/meta.yaml` and
   `docs/source/conf.py`. `python .github/update_version.py <file> <version>`
   does this per file and fails if the file declares no version, so a silent
   miss is not possible. Update dependencies/versions and the description in
   `pyproject.toml` and `requirements.txt` if necessary.
2. Create a new tag/release on GitHub with a matching version number (e.g. `v0.1`)

## Building the PyPI package

### Prerequisites
1. Install the newest version of pip (`python -m pip install --upgrade pip` or `conda update pip`)
2. Install the build frontend and twine (`pip install --upgrade build twine`)

### Build and upload package

1. Clean the dist folder (remove old version builds)
2. Navigate to the package folder and build with `python -m build`, which produces
   both the source distribution and the wheel
3. Verify the metadata renders on PyPI with `twine check dist/*`
4. Upload the source distribution and wheel via `twine upload dist/*` (use PyPI credentials)

## Building the Conda package

### Prerequisites
1. Create a conda environment to build the package (e.g. `conda create -n straindesign-build`)
2. Activate the environment (`conda activate straindesign-build`)
3. Install the requirements (`conda install anaconda-client conda-build`)

### Build and upload package

1. Navigate to the package folder and build with `conda-build conda-recipe/. -c conda-forge --croot conda-bld`
2. Clean up the conda-bld folder (remove old version builds)
3. `anaconda login`
4. `anaconda upload -u cnapy conda-bld/noarch/straindesign*`
5. Clean up with conda
