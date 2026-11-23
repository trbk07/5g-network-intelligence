"""Unfitted-state serialization and fixed arithmetic, with no optimizer/fitting."""

import json

import numpy as np
import pytest
import torch

from src.data.io import sha256
from src.features.gru_inputs import GRUPreprocessor
from src.models.temporal_linear import TemporalLinear
from src.models.temporal_mlp import TemporalResidualMLP
from src.models.mlp_blend_research_forecaster import MLPBlendResearchForecaster
import src.models.blend_research_forecaster as module


class FixedHurdle:
    """Analytical reference fixture, not a fitted model."""

    def __init__(self,root):
        pass

    def predict_at(self,history,origin,horizon):
        return np.full(history.shape[1],2.)


def fixture(root,epoch=3,mlp=False):
    component="mlp" if mlp else "linear"
    selection={"choices":{str(h):{"variant":"A",f"{component}_weight":0.2} for h in (1,6,24)}}
    (root/"selection.json").write_text(json.dumps(selection,indent=2))
    protocol={"fit_end":503,"audit_end":671,"selection_sha256":sha256(root/"selection.json")}
    if mlp:
        protocol.update(expected_selected_variants=["A"],expected_fixed_epochs={"A":4},mlp_manifest_sha256="initialized-fixture")
    (root/"protocol.json").write_text(json.dumps(protocol))
    directory=root/"A"
    directory.mkdir()
    identity={"architecture":f"temporal-{component}-refit-v1","variant":"A","fit_end":503,"epochs":4,"seed":17}
    (directory/"meta.json").write_text(json.dumps({"identity":identity,"audit_protocol":protocol,"holdout_used":False,
        "base_protocol":{"hidden_size":64,"dropout":0.1},"development_manifest_sha256":"initialized-fixture"}))
    GRUPreprocessor(np.zeros(1),np.zeros(1),np.ones(1),("0_0_0",),503,"A").save(directory/"preprocessor.npz")
    model=TemporalResidualMLP(1,64,0.1) if mlp else TemporalLinear(1)
    torch.save({"identity":identity,"epoch":epoch,"model":model.state_dict()},directory/"last.pt")
    files={p.relative_to(root).as_posix():sha256(p) for p in root.rglob("*") if p.is_file()}
    (root/"manifest.json").write_text(json.dumps({"files":files}))


@pytest.mark.parametrize("mlp",[False,True])
def test_initialized_components_and_fixed_mixture_roundtrip(tmp_path,monkeypatch,mlp):
    fixture(tmp_path,mlp=mlp)
    monkeypatch.setattr(module,"HurdleForecaster",FixedHurdle)
    loader=MLPBlendResearchForecaster if mlp else module.BlendResearchForecaster
    model=loader(tmp_path,tmp_path)
    component="mlp" if mlp else "linear"
    history=np.ones((168,2,4))
    components=model.predict_components(history,503,6)
    np.testing.assert_allclose(components[f"selected_{component}"],[1,1],atol=1e-5)
    np.testing.assert_allclose(components["hurdle_median"],[2,2])
    np.testing.assert_allclose(components[f"hurdle_{component}_blend"],[1.8,1.8],atol=1e-5)


def test_wrong_final_epoch_is_not_silently_selected(tmp_path,monkeypatch):
    fixture(tmp_path,epoch=2)
    monkeypatch.setattr(module,"HurdleForecaster",FixedHurdle)
    with pytest.raises(ValueError,match="fixed epoch"):
        module.BlendResearchForecaster(tmp_path,tmp_path)


def test_changed_selection_is_rejected_before_model_loading(tmp_path):
    fixture(tmp_path)
    (tmp_path/"selection.json").write_text('{"choices":{}}')
    with pytest.raises(ValueError,match="checksum"):
        module.BlendResearchForecaster(tmp_path,tmp_path)
