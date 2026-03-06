import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

result = subprocess.run(
    [sys.executable, "load_datasets.py"],
    capture_output=False,
    cwd=str(PROJECT_ROOT)
)
sys.exit(result.returncode)
