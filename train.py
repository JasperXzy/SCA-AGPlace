import sys
from pathlib import Path


def _setup_paths():
    repo_root = Path(__file__).resolve().parent
    candidates = [repo_root, repo_root / "src"]
    for path in reversed(candidates):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_setup_paths()

from mag_vlaq.lightning.cli import main as cli_main

if __name__ == "__main__":
    cli_main()
