import numpy as np
from scipy import sparse
from straindesign.strainDesignMILP import _drop_non_minimal


def test_supersets_are_dropped_and_counted():
    rows = [[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 1]]
    z = sparse.csr_matrix(np.array(rows))
    kept, dropped = _drop_non_minimal(z)
    assert dropped == 2                                     # {0,1,2} > {0,1}; {0,1,3} > {0,1} and > {3}
    assert sorted(frozenset(kept[i].indices.tolist()) for i in range(kept.shape[0])) == \
        sorted([frozenset({0, 1}), frozenset({0, 2}), frozenset({3})])


def test_minimal_set_is_untouched():
    z = sparse.csr_matrix(np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]))
    kept, dropped = _drop_non_minimal(z)
    assert dropped == 0 and kept.shape[0] == 3
