"""Test that the public API downstream packages import stays importable.

Being unused inside this repository is not evidence that a public name is unused:
`lineqlist2str` had no caller here and was deleted as dead code, but CNApy imports
it at startup in gui_elements/strain_design_dialog.py, so the removal broke the
application. These names are re-checked here because grepping this repository
cannot see that.
"""
import importlib

import pytest

# Imported by CNApy; see cnapy/gui_elements/strain_design_dialog.py and siblings.
CNAPY_IMPORTS = {
    "straindesign": [
        "avail_solvers",
        "compute_strain_designs",
        "fba",
        "lineq2list",
        "lineqlist2str",
        "linexpr2dict",
        "linexprdict2str",
        "plot_flux_space",
        "SDModule",
        "select_solver",
        "yopt",
    ],
    "straindesign.parse_constr": ["lineq2list", "linexpr2dict", "linexprdict2str"],
    "straindesign.names": ["CPLEX", "GLPK", "GUROBI", "SCIP"],
    "straindesign.strainDesignSolutions": ["SDSolutions"],
}


@pytest.mark.parametrize("module,names", sorted(CNAPY_IMPORTS.items()))
def test_public_names_are_importable(module, names):
    """Names that downstream packages import must remain importable."""
    mod = importlib.import_module(module)
    missing = [n for n in names if not hasattr(mod, n)]
    assert not missing, f"{module} no longer exports: {', '.join(missing)}"


def test_lineqlist2str_formats_an_inequality():
    """lineqlist2str renders [lhs, sign, rhs] as a string, as CNApy displays it."""
    from straindesign import lineqlist2str

    assert lineqlist2str([{"a": 3.0, "b": -1.0}, "<=", 2.0]) == "3.0 a - 1.0 b <= 2.0"
    assert lineqlist2str([{}, "<=", 2.0]) == "<= 2.0"
    assert lineqlist2str([{}, "", ""]) == ""
