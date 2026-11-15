"""Canonical KPI and numeric beam schema."""
import re

KPIS = ("DLPRB", "DLThpVol", "DLThpTime", "MR_number")
SEGMENTS = {"train": ("train", "0w-5w"), "week6": ("test", "5w-6w"),
            "week11": ("test", "10w-11w")}


def parse_beam_id(beam_id: str) -> tuple[int, int, int]:
    """Parse a strict bs_cell_beam identifier into numeric components."""
    if not re.fullmatch(r"\d+_\d+_\d+", beam_id):
        raise ValueError(f"Invalid beam ID: {beam_id!r}")
    return tuple(int(part) for part in beam_id.split("_"))


def csv_path(kpi: str, segment: str) -> str:
    """Return repository-relative CSV path for one KPI and segment."""
    folder, period = SEGMENTS[segment]
    return f"data/{folder}/{kpi}_{folder}_{period}.csv"
