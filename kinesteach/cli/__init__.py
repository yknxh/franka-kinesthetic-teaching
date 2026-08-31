"""The command line, as a package.

    main.py       teach · process · replay · list · report · webui
    calibrate.py  workspace · payload-sweep · payload-fit
    common.py     shared plumbing: config, connection, checked homing

A directory rather than three `cli_*` modules, which is what the rest of the
package already does for a group (`backend/`, `webui/`). `calibrate` rather
than `payload` so it does not shadow `kinesteach.payload`, and because the two
things it does -- walking the safe region, weighing the unregistered load --
are both setup the robot needs before a demonstration, not part of one.
"""
from .main import build_parser, main

__all__ = ["main", "build_parser"]
