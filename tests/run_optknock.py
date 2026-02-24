"""Time just optknock to get baseline CPLEX timing."""
import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "tests/test_performance.py",
     "-v", "--tb=short",
     "-k", "optknock"],
    capture_output=True, text=True,
    cwd="C:/Users/phili/OneDrive/Dokumente/Python/straindesign",
)
output = result.stdout + result.stderr
# print last part
lines = output.splitlines()
for line in lines:
    if any(x in line for x in ["PASSED","FAILED","ERROR","test_","elapsed","sol","status","====","---"]):
        print(line)
