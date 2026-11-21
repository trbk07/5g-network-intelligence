"""Five read-only endpoints; all inference shares one loaded, verified Predictor."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.schemas import ForecastRequest, ForecastResponse, RiskResponse
from src.inference.predictor import Predictor

logger = logging.getLogger(__name__)


def create_app(predictor_factory=None):
    """An injectable loader keeps error contracts and tiny fixtures testable without training."""

    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = None
        try:
            app.state.predictor = (
                predictor_factory()
                if predictor_factory
                else Predictor(os.getenv("ASSET_DIR", "demo_assets"))
            )
        except Exception:
            logger.exception("Replay assets unavailable; readiness endpoints will report 503")
        yield
        app.state.predictor = None

    app = FastAPI(
        title="5G Network Intelligence",
        version="1.0.0",
        lifespan=lifespan,
        description="Historical offline replay. No training or live telemetry in requests.",
    )

    def ready():
        model = getattr(app.state, "predictor", None)
        if model is None:
            raise HTTPException(
                503,
                "Verified model/replay assets unavailable. Run the documented asset setup and restart.",
            )
        return model

    @app.get("/health")
    def health():
        model = ready()
        return {
            "status": "ready",
            "model_ready": True,
            "data_ready": True,
            "is_replay": True,
            "model_version": model.version,
            "beams_loaded": len(model.store.index),
        }

    @app.get("/model-info")
    def model_info():
        model = ready()
        result = model.info()
        result["benchmark"] = model.store.manifest.get("benchmark", [])
        return result

    @app.get("/replay/meta")
    def replay_meta():
        return ready().store.meta()

    def infer(request, risk=False):
        model = ready()
        try:
            return (model.predict_risk if risk else model.predict)(request.beam_ids, request.origin)
        except KeyError:
            raise HTTPException(404, "Unknown beam in loaded demo subset") from None
        except ValueError:
            raise HTTPException(
                422, "Invalid origin or unavailable input/output for this replay request"
            ) from None
        except Exception:
            logger.exception("Inference failed")
            raise HTTPException(503, "Model inference temporarily unavailable") from None

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(request: ForecastRequest):
        return infer(request)

    @app.post("/risk", response_model=RiskResponse)
    def risk(request: ForecastRequest):
        return infer(request, True)

    return app


app = create_app()
