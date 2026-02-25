import sys, time
print("Python:", sys.version)
import cobra
print("cobra imported")
m = cobra.io.read_sbml_model('tests/iMLcore.xml')
print(f"iMLcore: {len(m.reactions)} rxns, {len(m.metabolites)} mets")

t0 = time.perf_counter()
c1 = m.copy()
print(f"model.copy(): {(time.perf_counter()-t0)*1000:.0f} ms")

from cobra.util import create_stoichiometric_matrix
from scipy import sparse
t0 = time.perf_counter()
S = sparse.csr_matrix(create_stoichiometric_matrix(m))
print(f"create_stoichiometric_matrix(): {(time.perf_counter()-t0)*1000:.0f} ms  shape={S.shape}")

print("Done")
