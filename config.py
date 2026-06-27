"""
Shared configuration for the TC1Analysis tools.

DATA_ROOT must point at the CASH hourly-SAC archive, laid out as
    <DATA_ROOT>/<year>/<month>/<day>/<hour>.sac      e.g. 2022/9/1/0.sac

It is NOT part of this repository (it is large raw data). Set the CASH_ARCHIVE
environment variable to your own path, or edit the default below.
"""
import os

_env = os.environ.get("CASH_ARCHIVE")
_default = os.path.expanduser("~/jamaseisData/CASH")
_sample = os.path.join(os.path.dirname(__file__), "sample_data", "CASH")
# Use CASH_ARCHIVE if set; else the full local archive if present; else the
# bundled sample dataset (so a fresh clone runs out of the box).
DATA_ROOT = _env or (_default if os.path.isdir(_default) else _sample)

# Station location (Cashmere, Christchurch, New Zealand)
CASH_LAT, CASH_LON = -43.567, 172.622
