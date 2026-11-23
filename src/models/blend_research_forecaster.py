"""Fixed 80/20 research blend; CPU history-only inference, no production routing."""

import json
from pathlib import Path

import numpy as np
import torch

from src.data.io import sha256
from src.features.gru_inputs import GRUPreprocessor
from src.models.forecast_blend import blend_forecasts
from src.models.hurdle_forecaster import HurdleForecaster
from src.models.temporal_linear import TemporalLinear


class BlendResearchForecaster:
    component = "linear"

    def build_model(self, variant, metadata):
        return TemporalLinear(1 if variant=="A" else 4)

    def __init__(self, root, hurdle_root):
        root=Path(root)
        for name,digest in json.loads((root/"manifest.json").read_text())["files"].items():
            if sha256(root/name)!=digest:
                raise ValueError("Fixed blend artifact checksum mismatch")
        self.protocol=json.loads((root/"protocol.json").read_text())
        self.selection=json.loads((root/"selection.json").read_text())
        if (self.protocol["fit_end"]!=503 or self.protocol["audit_end"]!=671
                or sha256(root/"selection.json")!=self.protocol["selection_sha256"]):
            raise ValueError("Fixed blend protocol/selection differs")
        self.hurdle=HurdleForecaster(hurdle_root)
        self.models,self.preprocessors,self.metadata={},{},{}
        for variant in {c["variant"] for c in self.selection["choices"].values()}:
            if variant not in ("A","B"):
                raise ValueError("Unsupported fixed blend variant")
            directory=root/variant
            meta=json.loads((directory/"meta.json").read_text())
            identity=meta["identity"]
            if (identity["fit_end"]!=503 or identity["architecture"]!=f"temporal-{self.component}-refit-v1"
                    or identity["variant"]!=variant or meta["audit_protocol"]!=self.protocol or meta["holdout_used"]):
                raise ValueError("Fixed linear refit provenance differs")
            preprocessor=GRUPreprocessor.load(directory/"preprocessor.npz")
            if preprocessor.fit_end!=503 or preprocessor.variant!=variant:
                raise ValueError("Fixed refit preprocessing differs")
            state=torch.load(directory/"last.pt",map_location="cpu",weights_only=True)
            if state["identity"]!=identity or state["epoch"]+1!=identity["epochs"]:
                raise ValueError("Last checkpoint does not match specified fixed epoch")
            model=self.build_model(variant,meta).eval()
            model.load_state_dict(state["model"],strict=True)
            self.models[variant],self.preprocessors[variant],self.metadata[variant]=model,preprocessor,meta

    def predict_components(self, history, origin, horizon):
        history=np.asarray(history)
        if (history.ndim!=3 or history.shape[0]!=168 or history.shape[2]!=4
                or not history.shape[1] or np.isinf(history).any()
                or not isinstance(origin,(int,np.integer)) or origin<167 or horizon not in (1,6,24)):
            raise ValueError("Supported history-only origin, horizon and 168-row panel required")
        choice=self.selection["choices"][str(horizon)]
        variant=choice["variant"]
        preprocessor=self.preprocessors[variant]
        sequence=preprocessor.transform(history,variant).transpose(1,0,2)
        phase=preprocessor.static_at(np.full(history.shape[1],origin),np.arange(history.shape[1]))
        with torch.inference_mode():
            value=self.models[variant](torch.as_tensor(sequence),torch.as_tensor(phase))[:,(1,6,24).index(horizon)].numpy()
        linear=np.maximum(0,value*preprocessor.scale[0]+preprocessor.mean[0])
        hurdle=self.hurdle.predict_at(history,origin,horizon)
        return {f"selected_{self.component}":linear,"hurdle_median":hurdle,
                f"hurdle_{self.component}_blend":blend_forecasts(hurdle,linear,choice[f"{self.component}_weight"])}

    def predict_at(self, history, origin, horizon):
        return self.predict_components(history,origin,horizon)[f"hurdle_{self.component}_blend"]
