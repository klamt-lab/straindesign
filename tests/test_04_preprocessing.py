"""Test if all strain design functions run correctly."""
from .test_01_load_models_and_solvers import *
import straindesign as sd
from numpy import inf


@pytest.mark.timeout(15)
def test_gpr_extension_compression1(model_gpr):
    gkocost = {
        'g1': 1.0,
        'g2': 1.0,
        'g4': 3.0,
        'g5': 2.0,
        'g8': 1.0,
        'g9': 1.0,
    }
    gkicost = {'g3': 1.0, 'g6': 1.0, 'g7': 1.0}
    sd.extend_model_gpr(model_gpr, use_names=False)
    cmp_map = sd.compress_model(model_gpr)
    assert (len(model_gpr.reactions) == 16)
    gkocost, gkicost, cmp_map = sd.compress_ki_ko_cost(gkocost, gkicost, cmp_map)
    assert (len(gkocost) == 4)
    assert (len(gkicost) == 3)


@pytest.mark.timeout(15)
def test_gpr_extension_compression2(model_gpr):
    gkocost = {
        'g1': 1.0,
        'g2': 1.0,
        'g4': 3.0,
        'g5': 2.0,
        'g8': 1.0,
        'g9': 1.0,
    }
    gkicost = {'g3': 1.0, 'g6': 1.0, 'g7': 1.0}
    sd.extend_model_gpr(model_gpr, use_names=False)
    assert (len(model_gpr.reactions) == 40)
    sd.compress_model(model_gpr, set(('g5', 'g9')))
    assert (len(model_gpr.reactions) == 18)
