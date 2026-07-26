"""Test that the public API downstream packages import stays importable.

Being unused inside this repository is not evidence that a public name is unused:
`lineqlist2str` had no caller here and was deleted as dead code, but CNApy imports
it at startup in gui_elements/strain_design_dialog.py, so the removal broke the
application. These names are re-checked here because grepping this repository
cannot see that.
"""
import ast
import importlib
import io
import tarfile
import urllib.request
import warnings

import pytest

CNAPY_TARBALL = "https://github.com/cnapy-org/CNApy/archive/refs/heads/master.tar.gz"

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


def _straindesign_names_imported_by(source):
    """Yield (module, name) for every straindesign import in a source file."""
    with warnings.catch_warnings():
        # Parsing someone else's source can raise SyntaxWarning for things like
        # invalid escape sequences. Those are CNApy's to fix, not signal here.
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.split(".")[0] == "straindesign":
                for alias in node.names:
                    if alias.name != "*":
                        yield node.module, alias.name


@pytest.mark.downstream
def test_cnapy_master_imports_still_exist():
    """Every straindesign name CNApy imports must still be exported.

    CNApy is read rather than installed: it depends on Qt, a JVM via jpype, and
    CPLEX, none of which belong in this matrix. Reading the source catches names
    CNApy has newly imported, which the hardcoded list above cannot.
    """
    try:
        with urllib.request.urlopen(CNAPY_TARBALL, timeout=120) as response:
            payload = response.read()
    except Exception as exc:  # offline, rate-limited, or the branch moved
        pytest.skip(f"could not fetch CNApy source: {exc}")

    wanted = set()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".py"):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            wanted.update(_straindesign_names_imported_by(handle.read().decode("utf-8", "replace")))

    assert wanted, "found no straindesign imports in CNApy; the parser or the URL is wrong"

    missing = []
    for module, name in sorted(wanted):
        mod = importlib.import_module(module)
        if not hasattr(mod, name):
            missing.append(f"{module}.{name}")
    assert not missing, ("CNApy imports names this package no longer exports: " + ", ".join(missing))


def test_lineqlist2str_formats_an_inequality():
    """lineqlist2str renders [lhs, sign, rhs] as a string, as CNApy displays it."""
    from straindesign import lineqlist2str

    assert lineqlist2str([{"a": 3.0, "b": -1.0}, "<=", 2.0]) == "3.0 a - 1.0 b <= 2.0"
    assert lineqlist2str([{}, "<=", 2.0]) == "<= 2.0"
    assert lineqlist2str([{}, "", ""]) == ""
