"""Test if package imports successfully."""

import pytest


def test1():
    import cobra
    from pathlib import Path
    from cobra.io import read_sbml_model
    import straindesign
    model_path = (Path(cobra.__path__[0]) / "data" / "textbook.xml.gz")
    model = read_sbml_model(str(model_path.resolve()))
    straindesign.fba(model)
