"""
Shared configuration for the TC1Analysis tools.

DATA_ROOT must point at the CASH hourly-SAC archive, laid out as
    <DATA_ROOT>/<year>/<month>/<day>/<hour>.sac      e.g. 2022/9/1/0.sac

It is NOT part of this repository (it is large raw data). Set the CASH_ARCHIVE
environment variable to your own path, or edit the default below.
"""
import os

DATA_ROOT = os.environ.get("CASH_ARCHIVE", os.path.expanduser("~/jamaseisData/CASH"))

# Station location (Cashmere, Christchurch, New Zealand)
CASH_LAT, CASH_LON = -43.567, 172.622
