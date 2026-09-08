"""Read-only release inputs and separate generated outputs."""
from pathlib import Path
import os
REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'data'
REFERENCE = REPO / 'reference'
def run_path(experiment):
    root = Path(os.environ.get('PAD_OUTPUT_ROOT', str(REPO / 'runs'))).resolve() / experiment
    for path in (root, root/'models', root/'cache'):
        path.mkdir(parents=True, exist_ok=True)
    return root
