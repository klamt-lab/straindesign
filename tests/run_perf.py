"""Run the quick performance suite and print summary."""
import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "tests/test_performance.py",
     "-v", "--tb=short"],
    capture_output=True, text=True,
    cwd="C:/Users/phili/OneDrive/Dokumente/Python/straindesign",
)
output = result.stdout + result.stderr
# Show last 6000 chars
print(output[-6000:] if len(output) > 6000 else output)
sys.exit(result.returncode)
