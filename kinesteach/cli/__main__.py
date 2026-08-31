"""`python -m kinesteach.cli`, which is what scripts/kinesteach runs."""
import sys

from .main import main

sys.exit(main())
