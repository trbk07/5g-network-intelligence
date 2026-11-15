"""Descriptive data audits; no imputation, filtering or model fitting."""
import numpy as np

from src.data.panel import Panel
from src.data.schema import KPIS, parse_beam_id


def audit_missing(panel: Panel) -> dict:
    """Return missing ratios per KPI and per beam."""
    mask = np.isnan(panel.values)
    return {"per_kpi": dict(zip(KPIS, mask.mean(axis=(0, 1)).tolist())),
            "per_beam": mask.mean(axis=(0, 2)).tolist()}


def audit_zeros(panel: Panel) -> dict:
    """Return zero/active ratios using observed values as denominator."""
    result = {}
    for i, kpi in enumerate(KPIS):
        values = panel.values[:, :, i]
        observed = values[~np.isnan(values)]
        result[kpi] = {"observed": len(observed),
                       "zero_ratio": float(np.mean(observed == 0)) if observed.size else None,
                       "active_ratio": float(np.mean(observed > 0)) if observed.size else None}
    return result


def audit_range(panel: Panel) -> dict:
    """Report finite range, quantiles and negative counts without deleting values."""
    result = {}
    for i, kpi in enumerate(KPIS):
        values = panel.values[:, :, i]
        observed = values[np.isfinite(values)]
        result[kpi] = {"negative_count": int((observed < 0).sum()),
                       "quantiles": np.quantile(observed, [0, .5, .9, .95, 1]).tolist()
                       if observed.size else None}
    return result


def audit_duplicates(panel: Panel) -> dict:
    """Report key duplicates; builder rejects these before alignment."""
    return {"duplicate_time_keys": len(panel.time) - len(np.unique(panel.time)),
            "duplicate_beam_ids": len(panel.beam_ids) - len(set(panel.beam_ids)),
            "source_files": panel.metadata.get("source_audit", {})}


def check_mapping(panel: Panel) -> dict:
    """Check identifiers against the documented 30 × 3 × 32 network topology."""
    ids = [parse_beam_id(beam) for beam in panel.beam_ids]
    expected = {(bs, cell, beam) for bs in range(30) for cell in range(3) for beam in range(32)}
    return {"complete_expected_mapping": set(ids) == expected,
            "within_expected_bounds": all(item in expected for item in ids),
            "base_stations": len({item[0] for item in ids}),
            "cells": len({item[:2] for item in ids}), "beams": len(ids)}


def audit_beams(panel: Panel) -> dict:
    """Return topology counts and all-missing/constant beam counts."""
    target = panel.values[:, :, 0]
    constant = sum(len(np.unique(col[np.isfinite(col)])) == 1 for col in target.T)
    return {**check_mapping(panel), "all_missing_target_beams": int(np.isnan(target).all(0).sum()),
            "constant_target_beams": constant}


def check_temporal_boundary(train: Panel, week6: Panel, *, verified: bool = False,
                            evidence: str = "") -> dict:
    """Require documented continuity and exact source indices before concatenation."""
    structure_ok = (np.array_equal(train.time, np.arange(840))
                    and np.array_equal(week6.time, np.arange(168))
                    and train.beam_ids == week6.beam_ids)
    allowed = bool(verified and evidence.strip() and structure_ok)
    return {"continuity_verified": allowed, "structure_ok": bool(structure_ok),
            "evidence": evidence, "week6_time_offset": 840 if allowed else None,
            "reason": "Documented continuity and indices verified" if allowed else
            "Use internal holdout: continuity has not passed evidence and structure checks"}
