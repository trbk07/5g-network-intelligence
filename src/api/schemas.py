"""Explicit wire contract: JSON null means missing; historical thresholds are not forecast bands."""

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class ForecastRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"example": {"origin": 671, "beam_ids": ["0_0_0"]}}
    )
    origin: StrictInt
    beam_ids: list[str] = Field(min_length=1, max_length=32)

    @field_validator("beam_ids")
    @classmethod
    def unique_beams(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("Beam IDs must be distinct")
        return value


class HistoryPoint(BaseModel):
    time: int
    value: float | None


class Prediction(BaseModel):
    horizon: int
    target_index: int
    point: float | None
    actual: float | None
    flags: list[str]


class BeamForecast(BaseModel):
    beam_id: str
    history: list[HistoryPoint]
    predictions: list[Prediction]


class ForecastResponse(BaseModel):
    origin: int
    model_version: str
    is_replay: bool
    beams: list[BeamForecast]


class RiskPrediction(Prediction):
    status: str
    valid_mask: bool
    reason: str | None


class BeamRisk(BeamForecast):
    historical_warning_p90: float | None
    historical_critical_p95: float | None
    threshold_fit_end: int
    predictions: list[RiskPrediction]


class RiskResponse(ForecastResponse):
    beams: list[BeamRisk]
