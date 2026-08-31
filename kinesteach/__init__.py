"""Kinesthetic teaching for the Franka Research 3.

    kinesteach/
      config, dataset, record, validate    what an episode is and whether it worked
      teach, replay, process, kinematics   the pipeline
      envelope, payload                    the safe region and the load in it
      safety                               nothing starts a policy without a way to stop it
      backend/                             the only place that knows about polymetis
      cli/, webui/                         the two front ends
"""

#: Stamped into every episode's metadata, so a recording says which version of
#: this tool produced it. Lives here rather than in `record.py`: it describes
#: the package, not the module that happens to write it down.
KINESTEACH_VERSION = "0.1.0"

__all__ = ["KINESTEACH_VERSION"]
