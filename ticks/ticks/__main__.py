import sys
from pathlib import Path

# Allow running directly from the repo without installing ibs_annotate.
_repo = Path(__file__).resolve().parent.parent.parent
_ibs = _repo / "ibs_annotate"
if _ibs.exists() and str(_ibs) not in sys.path:
    sys.path.insert(0, str(_ibs))

from .cli import main

if __name__ == "__main__":
    main()
