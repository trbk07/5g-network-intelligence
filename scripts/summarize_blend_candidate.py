"""Independent fixed-epoch, forecast, metric and CPU/GPU audit; no fitting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.finalize_blend_candidate import fixed_epochs
from scripts.run_risk import risk_records
from scripts.summarize_hurdle_research import score_rows
from src.data.io import sha256
from src.data.schema import parse_beam_id
from src.evaluation.bootstrap import paired_cluster_ci, paired_multihorizon_ci
from src.evaluation.origins import evaluation_origins
from src.evaluation.validation import validation_inputs
from src.models.blend_research_forecaster import BlendResearchForecaster
from src.models.forecast_blend import blend_forecasts
from src.utils.config import load_config


def summarize(config_path="configs/blend_candidate_audit.yaml", base_path="configs/temporal_linear.yaml",
              artifact_path="artifacts/blend_candidate", source_path="artifacts/temporal_linear",
              report_prefix="blend_candidate", component="linear", forecaster_class=BlendResearchForecaster):
    """Verify a fixed candidate; architecture-specific loading uses an explicit class."""
    cfg=load_config(config_path)
    base=load_config(base_path)
    root=Path(artifact_path)
    model=forecaster_class(root,"artifacts/hurdle_candidate")
    if model.protocol!=cfg:
        raise ValueError("Fixed audit protocol differs")
    if sha256(Path("reports/forecasts/hurdle_candidate_val.npz"))!=cfg["hurdle_replay_sha256"]:
        raise ValueError("Reference hurdle replay differs")
    source=Path(source_path)
    for name,digest in json.loads((source/"manifest.json").read_text())["files"].items():
        if sha256(source/name)!=digest:
            raise ValueError("Original stopping metadata/checkpoint checksum differs")
    for variant,meta in model.metadata.items():
        development=[json.loads((source/fold["name"]/variant/"meta.json").read_text()) for fold in base["folds"]]
        if (meta["base_protocol"]!=base or meta["identity"]["epochs"]!=fixed_epochs(development)
                or meta["development_manifest_sha256"]!=sha256(source/"manifest.json")):
            raise ValueError("Fixed epochs differ from original development stopping counts")
    panel,_=validation_inputs(load_config("configs/data.yaml"))
    replay=risk_records(Path(f"reports/forecasts/{report_prefix}_val.npz"))
    hurdle=risk_records(Path("reports/forecasts/hurdle_candidate_val.npz"))
    ensemble=risk_records(Path("reports/forecasts/improvement_audit.npz"))
    transfer=risk_records(Path("reports/forecasts/multisource_pilot_val.npz"))
    original=risk_records(Path("reports/forecasts/lightgbm_val.npz"))
    if set(replay)!={1,6,24}:
        raise ValueError("All declared horizons required")
    rows,parity,intervals=[],[],[]
    pairs={name:[] for name in ("hurdle_median","earlier_ensemble","pm03_nr3500")}
    scales=[]
    clusters=np.array([parse_beam_id(beam)[0] for beam in panel.beam_ids])
    expected={f"selected_{component}","hurdle_median",f"hurdle_{component}_blend"}
    for h,group in replay.items():
        if set(group)!=expected:
            raise ValueError("Every component and blend replay required")
        origins=evaluation_origins(503,671,h)
        actual=panel.values[origins+h,:,0]
        for name,record in group.items():
            np.testing.assert_array_equal(record["actual"],actual)
            np.testing.assert_array_equal(record["origins"],origins)
            if record["beam_ids"]!=panel.beam_ids:
                raise ValueError("Full original beam axis required")
            rows.extend(score_rows(actual,{name:record["predicted"]},"reused_validation",h))
        np.testing.assert_array_equal(group["hurdle_median"]["predicted"],hurdle[h]["hurdle_median"]["predicted"])
        coefficient=model.selection["choices"][str(h)][f"{component}_weight"]
        expected_prediction=blend_forecasts(group["hurdle_median"]["predicted"],group[f"selected_{component}"]["predicted"],coefficient)
        np.testing.assert_array_equal(group[f"hurdle_{component}_blend"]["predicted"],expected_prediction)
        earlier=ensemble[h]["improved_ensemble"]
        np.testing.assert_array_equal(earlier["actual"],actual)
        np.testing.assert_array_equal(earlier["origins"],origins)
        if earlier["beam_ids"]!=panel.beam_ids:
            raise ValueError("Earlier ensemble beam population differs")
        rows.extend(score_rows(actual,{"earlier_ensemble":earlier["predicted"]},"reused_validation",h))
        external=transfer[h]["pm03_nr3500"]
        for key in ("actual","origins"):
            np.testing.assert_array_equal(external[key],group[f"hurdle_{component}_blend"][key])
        if external["beam_ids"]!=panel.beam_ids:
            raise ValueError("External-transfer comparison population differs")
        rows.extend(score_rows(actual,{"pm03_nr3500":external["predicted"]},"reused_validation",h))
        observed=np.isfinite(actual)
        c1=original[h]["lightgbm_c1"]
        np.testing.assert_array_equal(c1["actual"],actual)
        np.testing.assert_array_equal(c1["origins"],origins)
        if c1["beam_ids"]!=panel.beam_ids:
            raise ValueError("C1 normalization population differs")
        scales.append(float(np.abs(c1["predicted"][observed]-actual[observed]).mean()))
        for name,prediction in (("hurdle_median",group["hurdle_median"]["predicted"]),
                                ("earlier_ensemble",earlier["predicted"]),("pm03_nr3500",external["predicted"])):
            candidate=group[f"hurdle_{component}_blend"]["predicted"]
            pair=(actual,candidate,prediction,observed)
            pairs[name].append(pair)
            intervals.append({"reference":name,"horizon":h,"unit":"original_MAE_difference",
                              **paired_cluster_ci(actual,candidate,prediction,clusters,observed)})
        for origin in (503,551,599):
            index=int(np.flatnonzero(origins==origin)[0])
            beams=np.array([0,len(panel.beam_ids)//2,len(panel.beam_ids)-1])
            predictions=model.predict_components(panel.values[origin-167:origin+1,beams,:],origin,h)
            for name,predicted in predictions.items():
                saved=group[name]["predicted"][index,beams]
                tolerance=1e-8 if name=="hurdle_median" else 1e-5
                np.testing.assert_allclose(predicted,saved,atol=tolerance,rtol=0)
                parity.append({"horizon":h,"origin":origin,"model":name,"beams_checked":len(beams),
                               "tolerance":tolerance,"max_error":float(np.max(np.abs(predicted-saved)))})
    for name,comparisons in pairs.items():
        intervals.append({"reference":name,"horizon":"mean_normalized","unit":"C1_MAE_normalized_difference",
                          **paired_multihorizon_ci(comparisons,clusters,scales)})
    frame=pd.DataFrame(rows)
    declared=pd.read_csv(f"reports/{report_prefix}_validation.csv")
    declared=declared[declared.load_tier=="All"].set_index(["model","horizon"])
    reconstructed=frame[frame.model.isin(expected)].set_index(["model","horizon"])
    columns=["mae","wape","n_valid","zero_fp_rate"]
    pd.testing.assert_frame_equal(reconstructed[columns].sort_index(),declared[columns].sort_index(),check_dtype=False,atol=1e-12,rtol=1e-12)
    frame.to_csv(f"reports/{report_prefix}_comparison.csv",index=False)
    pd.DataFrame(parity).to_csv(f"reports/{report_prefix}_cpu_parity.csv",index=False)
    pd.DataFrame(intervals).to_csv(f"reports/{report_prefix}_cluster_intervals.csv",index=False)
    Path(f"reports/{report_prefix}_verification.json").write_text(json.dumps({
        "fixed_epochs_verified":True,"selection_hash_verified":True,"hurdle_replay_hash_verified":True,
        "full_blend_replay_arithmetic_verified":True,"all_metrics_reconstructed":True,
        "history_only_beam_predictions_checked":sum(p["beams_checked"] for p in parity),
        "max_parity_error":max(p["max_error"] for p in parity),"holdout_used":False,"production_promoted":False,
        "claim_scope":"reused-validation audit only, no generalization or no-overfitting guarantee"},indent=2))
    print(frame[["model","horizon","mae","wape","n_valid","zero_fp_rate"]].to_string(index=False))
    print(pd.DataFrame(intervals)[["reference","horizon","difference","lower","upper"]].to_string(index=False))


def main():
    summarize()


if __name__=="__main__":
    main()
