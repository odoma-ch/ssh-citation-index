"""HTTP service for the fine-tuned GLiNER2 citation-linking decider.

For one parsed reference and the raw search results of one index, the model picks
the candidate that identifies the cited publication, or ``none``.

    uvicorn services.linking_decider.app:app --host 0.0.0.0 --port 8000

Environment:
    DECIDER_MODEL_PATH        local checkpoint directory (skips the Hub download)
    DECIDER_MODEL_ID          Hub repo (private: needs HF_TOKEN)
    DECIDER_MODEL_REVISION    Hub revision, default "main"
    DECIDER_MODEL_SHA256      expected model.safetensors SHA-256; startup fails on mismatch
    DECIDER_NONE_LOGIT_BIAS   constant added to the none logit (selected on validation)
    DECIDER_DEVICE            torch device, default "cpu"
    OMP_NUM_THREADS           set to the pod's CPU limit
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .formatting import (
    INDEXES,
    MAX_CANDIDATES,
    bias_none_probabilities,
    build_request,
    validate_probabilities,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "yurui983/gliner2.5-multi-decide-citation-linking"
# v4_select_v1: bias +7.737 was selected on validation for exactly these weights.
DEFAULT_SHA256 = "20deaed79508d22c1a65986d47ae319031596e53360f27b77ff057b953189406"
DEFAULT_NONE_LOGIT_BIAS = 7.737

Classify = Callable[[str, dict], dict]  # (text, labels) -> label probabilities


@dataclass(frozen=True)
class Decider:
    classify: Classify
    info: dict


class DecideRequest(BaseModel):
    reference: dict[str, Any] = Field(
        ..., description="Parsed reference (citation_index Reference fields)"
    )
    raw_reference: str = Field(
        ..., min_length=1, description="Original citation string"
    )
    index: Literal[INDEXES]  # type: ignore[valid-type]
    candidates: list[dict[str, Any]] = Field(
        ...,
        max_length=MAX_CANDIDATES,
        description="Raw connector search results, in rank order",
    )


class DecideResponse(BaseModel):
    choice: int | None = Field(
        ..., description="Position of the chosen candidate, or null for none"
    )
    score: float = Field(..., description="Bias-adjusted score of the decision")
    scores: list[float] = Field(..., description="Bias-adjusted candidate scores")
    none_score: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint() -> tuple[Path, str]:
    local = os.environ.get("DECIDER_MODEL_PATH")
    if local:
        return Path(local), local
    from huggingface_hub import snapshot_download

    model_id = os.environ.get("DECIDER_MODEL_ID", DEFAULT_MODEL_ID)
    revision = os.environ.get("DECIDER_MODEL_REVISION", "main")
    path = snapshot_download(model_id, revision=revision)
    return Path(path), f"{model_id}@{revision}"


def load_decider() -> Decider:
    """Load the checkpoint once; refuse weights the bias was not selected for."""
    import torch
    from gliner2 import AutoExtractor
    from gliner2.classification import ClassificationSchema, Classifier

    checkpoint, source = resolve_checkpoint()
    weights_sha = sha256_file(checkpoint / "model.safetensors")
    expected = os.environ.get("DECIDER_MODEL_SHA256", DEFAULT_SHA256)
    if expected and weights_sha != expected:
        raise RuntimeError(
            f"model.safetensors SHA-256 {weights_sha} != expected {expected}; "
            "set DECIDER_MODEL_SHA256 and DECIDER_NONE_LOGIT_BIAS for new weights"
        )
    device = os.environ.get("DECIDER_DEVICE", "cpu")
    model = AutoExtractor.from_pretrained(str(checkpoint)).to(device).eval()
    classifier = Classifier(model).eval()
    # ponytail: one request at a time per pod; scale with replicas, or batch
    # requests here if a GPU is ever attached.
    lock = threading.Lock()

    def classify(text: str, labels: dict) -> dict:
        schema = ClassificationSchema().single("link", labels)
        with lock, torch.inference_mode():
            answer = classifier.classify(text, schema)
        return dict(answer.probabilities("link"))

    info = {
        "model": source,
        "sha256": weights_sha,
        "device": device,
        "threads": torch.get_num_threads(),
    }
    logger.info("Linking decider loaded: %s", info)
    return Decider(classify=classify, info=info)


def none_logit_bias() -> float:
    return float(os.environ.get("DECIDER_NONE_LOGIT_BIAS", DEFAULT_NONE_LOGIT_BIAS))


def create_app(load: Callable[[], Decider] = load_decider) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["decider"] = load()
        state["bias"] = none_logit_bias()
        yield

    app = FastAPI(title="Citation linking decider", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            **state["decider"].info,
            "none_logit_bias": state["bias"],
        }

    @app.post("/v1/decide", response_model=DecideResponse)
    def decide(body: DecideRequest) -> DecideResponse:
        if not body.candidates:
            # Training rejects empty pools without inference.
            return DecideResponse(choice=None, score=1.0, scores=[], none_score=1.0)
        try:
            text, labels = build_request(
                body.reference, body.raw_reference, body.index, body.candidates
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            raw = state["decider"].classify(text, labels)
            validate_probabilities(raw, set(labels))
            probabilities = bias_none_probabilities(raw, state["bias"])
        except Exception:
            logger.exception("Decider inference failed (index=%s)", body.index)
            raise
        decision = max(probabilities, key=probabilities.get)
        scores = [probabilities[f"candidate_{i}"] for i in range(1, len(labels))]
        return DecideResponse(
            choice=None if decision == "none" else int(decision.split("_")[1]) - 1,
            score=probabilities[decision],
            scores=scores,
            none_score=probabilities["none"],
        )

    return app


app = create_app()
